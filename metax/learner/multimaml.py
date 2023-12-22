"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from collections import namedtuple
from functools import partial
from typing import NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from metax.data import Dataset
from metax.data.base import MultitaskDataset
from metax.models import Setcoder
from metax.utils import append_keys, tree_index

from .base import MetaLearnerState
from .maml import ModelAgnosticMetaLearning


class OracleMultiMetaParams(NamedTuple):
    shared: NamedTuple
    task: NamedTuple


class OracleMultiModelAgnosticMetaLearning(ModelAgnosticMetaLearning):
    def __init__(
        self,
        meta_model,
        batch_size,
        steps_inner,
        optim_fn_inner,
        optim_fn_outer,
        first_order,
        num_experts,
    ):
        super().__init__(
            meta_model, batch_size, steps_inner, optim_fn_inner, optim_fn_outer, first_order
        )
        self.num_experts = num_experts

    def outer_loss(self, rng, hstate, hparams, metadataset, steps_adapt):
        rng_adapt, rng_loss, rng_pred, rng_reset = jax.random.split(rng, 4)

        # Select task-specific hparams using task-id to reset params
        hparams_task = tree_index(hparams.task, metadataset.train.task_id)
        hstate_task = tree_index(hstate.task, metadataset.train.task_id)
        params_init, state_init = self.meta_model.reset_params(
            rng_reset, hparams_task, hstate_task, metadataset.train.x
        )

        # Use task-shared hparams for adapt and __call__
        (state, params), metrics_inner = self.adapt(
            rng_adapt,
            state_init,
            hstate.shared,
            params_init,
            hparams.shared,
            metadataset.train,
            steps_adapt,
        )
        # NOTE: is_training=True in combination with batch_norm would make this transductive,
        #       i.e. p(answer | query, all_other_points_in_query).
        # NOTE: If is_training=True, batch_norm stats accumulated during training
        #       will not be used regardless of the decay rate.
        pred, (state, hstate_shared) = self.meta_model(
            rng_pred,
            state,
            hstate.shared,
            params,
            hparams.shared,
            metadataset.test.x,
            is_training=False,
        )
        loss, metrics_outer = self.meta_model.loss_fn_outer(
            rng=rng_loss,
            pred=pred,
            target=metadataset.test.y,
            params=params,
            hparams=hparams.shared,
            state=state,
            hstate=hstate_shared,
            info=metadataset.test.info,
        )

        hstate = OracleMultiMetaParams(shared=hstate_shared, task=hstate.task)

        metrics = {
            **append_keys(metrics_outer, "outer"),
            **metrics_inner,
        }
        aux = ((state, params), (state_init, params_init), hstate)

        return loss, (aux, metrics)

    def reset(self, rng, sample_input):
        rng_hparams, rng_task = jax.random.split(rng)

        # Generate unique set of hparams per task + a shared set (this wastes a bit of memory)
        rngs_task = jax.random.split(rng_task, self.num_experts)
        reset_hparams_vmap = jax.vmap(self.meta_model.reset_hparams, in_axes=(0, None))
        hparams_task, hstate_task = reset_hparams_vmap(rngs_task, sample_input)
        hparams_shared, hstate_shared = self.meta_model.reset_hparams(rng_hparams, sample_input)

        hparams = OracleMultiMetaParams(shared=hparams_shared, task=hparams_task)
        hstate = OracleMultiMetaParams(shared=hstate_shared, task=hstate_task)

        optim_model = self.optim_fn_outer.init(hparams)

        return MetaLearnerState(hparams=hparams, optim=optim_model, hstate=hstate)


class MultiMetaParams(NamedTuple):
    encoder: NamedTuple
    model: NamedTuple


class MultiModelAgnosticMetaLearning(ModelAgnosticMetaLearning):
    def __init__(
        self,
        meta_model,
        batch_size,
        steps_inner,
        optim_fn_inner,
        optim_fn_outer,
        first_order,
        encoder,
        encoder_dim,
        input_shape,
        output_shape,
        num_classes: Optional[int],
    ):
        super().__init__(
            meta_model, batch_size, steps_inner, optim_fn_inner, optim_fn_outer, first_order
        )
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Filter hparams used for init to be generated by the setcoder
        hparams, _ = self.meta_model.reset_hparams(
            jax.random.PRNGKey(0), jnp.empty((1, *input_shape))
        )
        hparams_init_dict = {key: val for key, val in hparams._asdict().items() if "_init" in key}
        hparams_init = namedtuple("HparamsInit", hparams_init_dict)(**hparams_init_dict)
        hparams_init_flat, ravel_fn = jax.flatten_util.ravel_pytree(hparams_init)

        self.setcoder = Setcoder(len(hparams_init_flat), ravel_fn, encoder, encoder_dim, num_classes)

    def outer_loss(self, rng, hstate, hparams, metadataset, steps_adapt):
        rng_adapt, rng_encoder, rng_loss, rng_pred, rng_reset = jax.random.split(rng, 5)

        # Add task-batch dimension as set models are defined on batches of sets
        trainset = jtu.tree_map(partial(jnp.expand_dims, axis=0), metadataset.train)

        # Generate task-specific hparams using setcoder to reset params
        hparams_task, hstate_encoder = self.setcoder.apply(
            hparams.encoder, hstate.encoder, rng_encoder, trainset, is_training=True
        )
        params_init, state_init = self.meta_model.reset_params(
            rng_reset, hparams_task, hstate.model, metadataset.train.x
        )

        # Use task-shared hparams for adapt and __call__
        (state, params), metrics_inner = self.adapt(
            rng_adapt,
            state_init,
            hstate.model,
            params_init,
            hparams.model,
            metadataset.train,
            steps_adapt,
        )
        # NOTE: is_training=True in combination with batch_norm would make this transductive,
        #       i.e. p(answer | query, all_other_points_in_query).
        # NOTE: If is_training=True, batch_norm stats accumulated during training
        #       will not be used regardless of the decay rate.
        pred, (state, hstate_model) = self.meta_model(
            rng_pred,
            state,
            hstate.model,
            params,
            hparams.model,
            metadataset.test.x,
            is_training=False,
        )
        loss, metrics_outer = self.meta_model.loss_fn_outer(
            rng=rng_loss,
            pred=pred,
            target=metadataset.test.y,
            params=params,
            hparams=hparams.model,
            state=state,
            hstate=hstate_model,
            info=metadataset.test.info,
        )

        hstate = MultiMetaParams(encoder=hstate_encoder, model=hstate_model)

        metrics = {
            **append_keys(metrics_outer, "outer"),
            **metrics_inner,
        }
        aux = ((state, params), (state_init, params_init), hstate)

        return loss, (aux, metrics)

    def reset(self, rng, sample_input: Union[Dataset, MultitaskDataset]):
        rng_encoder, rng_hparams = jax.random.split(rng)
        hparams_model, hstate_model = self.meta_model.reset_hparams(rng_hparams, sample_input)
        sample_input_set = Dataset(
            x=jnp.expand_dims(sample_input, axis=0),
            y=jnp.empty((1, sample_input.shape[0], *self.output_shape)),
        )
        hparams_encoder, hstate_encoder = self.setcoder.init(
            rng_encoder, sample_input_set, is_training=True
        )

        hparams = MultiMetaParams(encoder=hparams_encoder, model=hparams_model)
        hstate = MultiMetaParams(encoder=hstate_encoder, model=hstate_model)

        optim_model = self.optim_fn_outer.init(hparams)

        return MetaLearnerState(hparams=hparams, optim=optim_model, hstate=hstate)

class CheatingModelAgnosticMetaLearning(ModelAgnosticMetaLearning):

    def adapt(self, rng, state, hstate, params, hparams, dataset, steps):
        """
        Adapts params and state on dataset given hparams and hstate.
        """

        params = self.meta_model.cheat(params, dataset.info)

        return (state, params), dict()
