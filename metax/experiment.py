"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import enum
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from tqdm import tqdm

import metax
import wandb
from metax import data, utils
from metax.data.base import MetaDataset
from metax.learner import MetaLearnerState


class CallbackEvent(enum.Enum):
    START = enum.auto()
    STEP = enum.auto()
    END = enum.auto()


class FewshotExperiment:
    def __init__(
        self,
        meta_learner: metax.learner.MetaLearner,
        meta_batch_size: int,
        steps_outer: int,
        steps_inner_test: int,
        metatrainset: data.Dataloader,
        metavalidset: data.Dataloader,
        metatestset: data.Dataloader,
        metaoodset: Optional[data.Dataloader] = None,
        metauxsets: Optional[Dict[str, data.Dataloader]] = None,
        logbook: Optional[Any] = None,
    ):
        self.meta_learner = meta_learner
        self.metatrainset = metatrainset
        self.metavalidset = metavalidset
        self.metatestset = metatestset
        self.metaoodset = metaoodset
        self.metauxsets = metauxsets
        self.meta_batch_size = meta_batch_size
        self.steps_outer = steps_outer
        self.steps_inner_test = steps_inner_test

        self.eval_every_n_steps = 1000
        self.logbook = logbook
        self.callbacks = defaultdict(list)

    def add_callback(self, onevent: CallbackEvent, callback_fn: Callable):
        """
        Add a callback function triggered on the specified event.

        Args:
            onevent: the `CallbackEvent` at which to trigger the callback
            callback: callback function that should take as inputs (rng, ctx, MetaState)
                where ctx is the current context of the Experiment (i.e. self)
        """
        self.callbacks[onevent].append(partial(callback_fn, ctx=self))

    def trigger_callback(self, rng, onevent: CallbackEvent, meta_state: MetaLearnerState):
        metrics = dict()
        for callback_fn in self.callbacks[onevent]:
            rng, rng_callback = jax.random.split(rng)
            metrics.update(callback_fn(rng=rng_callback, meta_state=meta_state))

        return metrics

    def log(self, log_dict):
        """
        Write logs to logger of current experimental state and metrics returned by step

        Args:
            log_dict: dict of metrics to be logged
        """

        def preprocess_singleton(value):
            if hasattr(value, "shape") and len(value.shape) > 0:
                # Log non-scalar metrics as histograms unless it contains nans
                if not np.any(np.isnan(value)):
                    return wandb.Histogram(np.array(value))
                else:
                    return np.nan
            elif isinstance(value, jnp.ndarray):
                return np.array(value)
            else:
                return value

        self.logbook.log(jtu.tree_map(preprocess_singleton, log_dict))

    def reset(self, rng, dummy_input: jnp.ndarray):
        """
        Reset ExperimentRunnerState, allows passing a checkpoint.
        """
        (rng_init,) = jax.random.split(rng, 1)
        meta_state_init = self.meta_learner.reset(rng_init, dummy_input)

        return meta_state_init

    def run(self, rng, meta_state: MetaLearnerState):
        """
        Repeatedly calling step, log and save with option to run in jit mode without
        intermediate logging/saving.
        """
        rng_start, rng_step, rng_end = jax.random.split(rng, 3)

        # Trigger callbacks on CallbackEvent.START
        if self.logbook is not None:
            metrics_start = self.trigger_callback(rng_start, CallbackEvent.START, meta_state)
            self.log(metrics_start)

        # Run meta-training loop
        meta_state = self._run(rng_step, meta_state)

        # Trigger callbacks on CallbackEvent.END
        if self.logbook is not None:
            metrics_end = self.trigger_callback(rng_end, CallbackEvent.END, meta_state)
            self.log(metrics_end)

        return meta_state

    def _run(self, rng, meta_state: MetaLearnerState):
        best_valid_loss_outer, best_meta_state = jnp.finfo(jnp.float32).max, meta_state

        for step, batch_train in enumerate(tqdm(self.metatrainset)):
            rng, rng_train, rng_valid, rng_test, rng_ood, rng_callback = jax.random.split(rng, 6)
            meta_state, metric_train = self.step(rng_train, meta_state, batch_train)
            metric_train = utils.prepend_keys(metric_train, "train")

            # Logging and checkpointing
            if step % self.eval_every_n_steps == 0 or step == len((self.metatrainset)) - 1:
                # Collect metrics
                metric_valid = self.eval(rng_valid, meta_state, self.metavalidset)
                metric_valid = utils.prepend_keys(metric_valid, "valid")

                metric_test = self.eval(rng_test, meta_state, self.metatestset)
                metric_test = utils.prepend_keys(metric_test, "test")

                if self.metaoodset is not None:
                    metric_ood = self.eval(rng_ood, meta_state, self.metaoodset)
                    metric_ood = utils.prepend_keys(metric_ood, "ood")
                else:
                    metric_ood = dict()

                if self.metauxsets is not None:
                    metric_aux = dict()
                    for mode, auxloader in self.metauxsets.items():
                        m = self.eval(rng_ood, meta_state, auxloader)
                        metric_aux.update(utils.prepend_keys(m, mode))
                else:
                    metric_aux = dict()

                metric_step = self.trigger_callback(rng_callback, CallbackEvent.STEP, meta_state)

                # Log metrics
                log_dict = {
                    **metric_train, **metric_valid, **metric_ood, **metric_test, **metric_step, **metric_aux
                }
                # NOTE: "inner" metrics are not scalar so we average them
                log_dict.update({
                    "{}".format(k): jnp.mean(v)
                    for (k, v) in log_dict.items() if "inner" in k
                })
                self.log(log_dict)

                # Checkpointing ("early stopping")
                if jnp.mean(metric_valid["valid_loss_outer"]) < best_valid_loss_outer:
                    best_valid_loss_outer = jnp.mean(metric_valid["valid_loss_outer"])
                    best_meta_state = meta_state

        return best_meta_state

    @partial(jax.jit, static_argnames="self")
    def step(self, rng, meta_state: MetaLearnerState, meta_batch: MetaDataset):
        meta_state, metrics = self.meta_learner.update(rng, meta_state, meta_batch)

        return meta_state, metrics

    @partial(jax.jit, static_argnames="self")
    def eval_step(self, rng, meta_state, meta_batch):
        _, metrics = self.meta_learner.eval(rng, meta_state, meta_batch, self.steps_inner_test)

        return metrics

    def eval(self, rng, meta_state: MetaLearnerState, meta_dataset: MetaDataset):
        rngs = jax.random.split(rng, len(meta_dataset))
        metrics_list = [
            self.eval_step(rng_t, meta_state, meta_batch)
            for rng_t, meta_batch in zip(rngs, meta_dataset)
        ]
        metrics = jtu.tree_map(lambda *args: jnp.stack((args)), *metrics_list)
        metrics = jtu.tree_map(lambda x: jnp.mean(x, axis=0), metrics)

        return metrics
