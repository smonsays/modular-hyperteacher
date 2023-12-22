"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from metax.data import Dataset, batch_generator
from metax.utils import append_keys

from .base import MetaLearnerInnerGradientDescent


class FastSlowState(NamedTuple):
    hparams: NamedTuple
    params: NamedTuple
    hstate: NamedTuple
    state: NamedTuple
    optim_outer: optax.OptState
    optim_inner: optax.OptState


class FastSlow(MetaLearnerInnerGradientDescent):
    """
    Simple baseline where meta-parameters and base-parameters use different learning rates
    but are updated simultaneously. The meta-batch is consumed sequentially.
    """

    def __init__(
        self,
        meta_model,
        batch_size,
        steps_inner,
        optim_fn_inner,
        optim_fn_outer,
        reset_inner,
    ):
        super().__init__(meta_model, batch_size, steps_inner, optim_fn_inner)
        self.optim_fn_outer = optim_fn_outer
        self.reset_inner = reset_inner

    def reset(self, rng, sample_input):
        rng_hparams, rng_params = jax.random.split(rng)

        hparams_init, hstate_init = self.meta_model.reset_hparams(rng_hparams, sample_input)
        params_init, state_init = self.meta_model.reset_params(
            rng_params, hparams_init, hstate_init, sample_input
        )
        optim_inner_init = self.optim_fn_inner.init(params_init)
        optim_outer_init = self.optim_fn_outer.init(hparams_init)

        meta_state_init = FastSlowState(
            hparams=hparams_init,
            params=params_init,
            hstate=hstate_init,
            state=state_init,
            optim_outer=optim_outer_init,
            optim_inner=optim_inner_init,
        )

        return meta_state_init

    def update(self, rng, meta_state, metadataset):
        rng_scan, rng_loss, rng_pred = jax.random.split(rng, 3)

        carry, metrics = jax.lax.scan(self._update_task, [rng_scan, meta_state], metadataset)
        _, meta_state = carry
        metrics = jtu.tree_map(partial(jnp.mean, axis=0), metrics)

        return meta_state, metrics

    def _update_task(self, carry, metadataset):
        rng, meta_state = carry
        rng_next, rng_data, rng_reset, rng_scan = jax.random.split(rng, 4)

        # Combine data and batch
        dataset = Dataset(
            x=jnp.concatenate((metadataset.train.x, metadataset.test.x), axis=0),
            y=jnp.concatenate((metadataset.train.y, metadataset.test.y), axis=0),
        )
        dataset_batched = batch_generator(rng_data, dataset, self.steps_inner, self.batch_size)

        # Reset parameters and inner optimizer
        if self.reset_inner:
            params, state = self.meta_model.reset_params(
                rng_reset, meta_state.hparams, meta_state.hstate, dataset.x
            )
            optim_inner = self.optim_fn_inner.init(params)
        else:
            params = meta_state.params
            state = meta_state.state
            optim_inner = meta_state.optim_inner

        # Perform a few update steps on the data
        carry_init = [
            rng_scan,
            state,
            meta_state.hstate,
            params,
            meta_state.hparams,
            optim_inner,
            meta_state.optim_outer,
        ]
        carry, metrics = jax.lax.scan(self._train_step, carry_init, dataset_batched)
        _, state, hstate, params, hparams, optim_inner, optim_outer = carry

        # Average gradnorm for outer steps of tasks for compatibility with other MetaLearner s
        # and define loss_outer (which technically doesn't exist) to be the last inner loss
        metrics["gradnorm_outer"] = jnp.mean(metrics["gradnorm_outer"])
        metrics["loss_outer"] = metrics["loss_inner"][-1]

        new_meta_state = FastSlowState(
            hparams=hparams,
            params=params,
            hstate=hstate,
            state=state,
            optim_outer=optim_outer,
            optim_inner=optim_inner,
        )

        return [rng_next, new_meta_state], metrics

    def _loss_fn(self, rng, state, hstate, params, hparams, batch):
        rng_pred, rng_loss = jax.random.split(rng)

        pred, (state, hstate) = self.meta_model(
            rng_pred, state, hstate, params, hparams, batch.x, is_training=True
        )
        # NOTE: Only uses the inner loss function (outer_loss only used for evaluation)
        loss, metrics = self.meta_model.loss_fn_inner(
            rng=rng_loss,
            pred=pred,
            target=batch.y,
            params=params,
            hparams=hparams,
            state=state,
            hstate=hstate,
            info=batch.info,
        )

        return loss, ((state, hstate), metrics)

    def _train_step(self, carry, batch):
        rng, state, hstate, params, hparams, optim_inner, optim_outer = carry
        rng_next, rng_grad = jax.random.split(rng)

        grad_fn = jax.grad(self._loss_fn, argnums=(3, 4), has_aux=True)
        (grads, hgrads), ((state, hstate), metrics_inner) = grad_fn(
            rng_grad, state, hstate, params, hparams, batch
        )

        params_update, optim_inner = self.optim_fn_inner.update(grads, optim_inner, params)
        params = optax.apply_updates(params, params_update)

        hparams_update, optim_outer = self.optim_fn_outer.update(hgrads, optim_outer, hparams)
        hparams = optax.apply_updates(hparams, hparams_update)

        metrics = {
            **append_keys(metrics_inner, "inner"),
            "gradnorm_inner": optax.global_norm(grads),
            "gradnorm_outer": optax.global_norm(hgrads),
        }

        return [rng_next, state, hstate, params, hparams, optim_inner, optim_outer], metrics


class BatchedFastSlow(FastSlow):
    """
    Batched version of FastSlow where updates on all tasks are performed in parallel,
    synchronising theta after each step.
    """
    def __init__(
        self,
        meta_model,
        batch_size,
        steps_inner,
        optim_fn_inner,
        optim_fn_outer,
        reset_inner,
        meta_batch_size,
    ):
        super().__init__(
            meta_model,
            batch_size,
            steps_inner,
            optim_fn_inner,
            optim_fn_outer,
            reset_inner,
        )
        self.meta_batch_size = meta_batch_size
        self._train_step_batched = jax.vmap(
            self._train_step, in_axes=([0, 0, None, 0, None, 0, None], 0)
        )
        self.reset_params_batch = jax.vmap(
            self.meta_model.reset_params, in_axes=(0, None, None, None)
        )
        self.optim_fn_inner_init_batch = jax.vmap(self.optim_fn_inner.init)

    def reset(self, rng, sample_input):
        rng_hparams, rng_params = jax.random.split(rng)

        hparams_init, hstate_init = self.meta_model.reset_hparams(rng_hparams, sample_input)
        rngs_params = jax.random.split(rng_params, self.meta_batch_size)
        params_init, state_init = self.reset_params_batch(
            rngs_params, hparams_init, hstate_init, sample_input
        )

        optim_inner_init = self.optim_fn_inner_init_batch(params_init)
        optim_outer_init = self.optim_fn_outer.init(hparams_init)

        meta_state_init = FastSlowState(
            hparams=hparams_init,
            params=params_init,
            hstate=hstate_init,
            state=state_init,
            optim_outer=optim_outer_init,
            optim_inner=optim_inner_init,
        )

        return meta_state_init

    def update(self, rng, meta_state, metadataset):
        rng_data, rng_loss, rng_reset, rng_scan = jax.random.split(rng, 4)

        # Combine data and batch
        @partial(jax.vmap, out_axes=1)
        def combine_and_batch(rng, metadataset):
            data_combined = Dataset(
                x=jnp.concatenate((metadataset.train.x, metadataset.test.x), axis=0),
                y=jnp.concatenate((metadataset.train.y, metadataset.test.y), axis=0),
            )
            return batch_generator(rng, data_combined, self.steps_inner, self.batch_size)

        rng_data = jax.random.split(rng_data, self.meta_batch_size)
        dataset_batched = combine_and_batch(rng_data, metadataset)

        # Reset parameters and inner optimizer
        if self.reset_inner:
            rngs_reset = jax.random.split(rng_reset, self.meta_batch_size)
            params, state = self.reset_params_batch(
                rngs_reset, meta_state.hparams, meta_state.hstate, metadataset.train.x[0]
            )
            optim_inner = self.optim_fn_inner_init_batch(params)
        else:
            state = meta_state.state
            params = meta_state.params
            optim_inner = meta_state.optim_inner

        # Perform updates on all tasks in parallel synchronising theta after each step
        rngs_scan = jax.random.split(rng_scan, self.meta_batch_size)
        carry_init = [
            rngs_scan,
            state,
            meta_state.hstate,
            params,
            meta_state.hparams,
            optim_inner,
            meta_state.optim_outer,
        ]
        carry, metrics = jax.lax.scan(self._update_step, carry_init, dataset_batched)
        _, state, hstate, params, hparams, optim_inner, optim_outer = carry

        # Average gradnorm_outer over inner_steps for compatibility with other MetaLearner s
        # and define loss_outer (which technically doesn't exist) to be the last inner loss
        metrics["gradnorm_outer"] = jnp.mean(metrics["gradnorm_outer"], axis=0)
        metrics["loss_outer"] = metrics["loss_inner"][-1]
        if "acc_inner" in metrics:
            metrics["acc_outer"] = metrics["acc_inner"][-1]

        new_meta_state = FastSlowState(
            hparams=hparams,
            params=params,
            hstate=hstate,
            state=state,
            optim_outer=optim_outer,
            optim_inner=optim_inner,
        )

        return new_meta_state, metrics

    def _update_step(self, carry, batch):
        (
            rng_next,
            state,
            hstate,
            params,
            hparams,
            optim_inner,
            optim_outer,
        ), metrics = self._train_step_batched(carry, batch)

        def mean_with_same_dtype(x, axis):
            dtype = jnp.dtype(x)
            mean = jnp.mean(x, axis=axis)

            return jax.lax.convert_element_type(mean, dtype)

        # HACK: Averaging over the optimizer state might result in unexpected behaviour
        # HACK: Averaging might change dtype (e.g. int to float), this simply casts it back
        hparams = jax.tree_map(partial(mean_with_same_dtype, axis=0), hparams)
        hstate = jax.tree_map(partial(mean_with_same_dtype, axis=0), hstate)
        optim_outer = jax.tree_map(partial(mean_with_same_dtype, axis=0), optim_outer)
        metrics = jax.tree_map(partial(jnp.mean, axis=0), metrics)

        return [rng_next, state, hstate, params, hparams, optim_inner, optim_outer], metrics


class FederatedFastSlow(FastSlow):
    """
    Federated version of FastSlow where updates on all tasks are performed in parallel, only
    synchronising theta after all `steps_inner` are done.
    """
    def __init__(
        self,
        meta_model,
        batch_size,
        steps_inner,
        optim_fn_inner,
        optim_fn_outer,
        reset_inner,
    ):
        super().__init__(
            meta_model,
            batch_size,
            steps_inner,
            optim_fn_inner,
            optim_fn_outer,
            reset_inner,
        )
        self.batch_update_task = jax.vmap(self._update_task, in_axes=([0, None], 0))

    def update(self, rng, meta_state, metadataset):

        rng_loss, rng_update = jax.random.split(rng, 2)

        meta_batch_size = len(metadataset.train.x)
        rngs_update = jax.random.split(rng_update, meta_batch_size)
        (_, meta_state_batch), metrics = self.batch_update_task(
            [rngs_update, meta_state], metadataset
        )

        # NOTE: We also average over the params and optim_inner but this might not be desired
        # HACK: Averaging over the optimizer state might result in unexpected behaviour
        # HACK: Averaging might change dtype (e.g. int to float), this simply casts it back
        meta_state_dtypes = jtu.tree_map(jnp.dtype, meta_state)
        meta_state = jtu.tree_map(partial(jnp.mean, axis=0), meta_state_batch)
        meta_state = jtu.tree_map(jax.lax.convert_element_type, meta_state, meta_state_dtypes)
        metrics = jtu.tree_map(partial(jnp.mean, axis=0), metrics)

        return meta_state, metrics
