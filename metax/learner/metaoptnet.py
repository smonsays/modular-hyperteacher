"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from metax import data, utils
from metax.module.anil import AlmostNoInnerLoop, AlmostNoInnerLoopParams
from metax.utils.utils import append_keys

from .base import MetaLearner, MetaLearnerState


class MetaOptNet(MetaLearner):
    def __init__(self, meta_model, optim_fn_outer, l2_reg):
        super().__init__(meta_model)
        assert isinstance(meta_model, AlmostNoInnerLoop)
        self.optim_fn_outer = optim_fn_outer
        self.l2_reg = l2_reg

    def adapt(self, rng, state, hstate, params, hparams, dataset, steps=None):
        """
        Adapts params and state on dataset given hparams and hstate.
        """
        rng_loss, rng_apply = jax.random.split(rng)
        features, state_body = self.meta_model.body.apply(
            hparams.body, hstate.body, rng_apply, dataset.x, is_training=True
        )

        # Append ones to features for bias (intercept)
        num_samples = dataset.y.shape[0]
        ones = jnp.ones((num_samples, 1), dtype=features.dtype)
        features = jnp.append(features, ones, axis=-1)

        # Solve least squares problem
        A = jnp.matmul(features, features.T) + self.l2_reg * jnp.eye(num_samples)
        params_fit = jnp.matmul(features.T, jnp.linalg.solve(A, dataset.y))

        # Extract updated weights and bias
        params_head = params.head
        params_head["linear_block/~/linear"]["w"] = params_fit[:-1, :]
        params_head["linear_block/~/linear"]["b"] = params_fit[-1, :]
        params = AlmostNoInnerLoopParams(head=params_head)

        _, (_, log_metric) = self.inner_loss(rng_loss, state, hstate, params, hparams, dataset)

        return (state, params), append_keys(log_metric, "inner_final")

    def update(self, rng, meta_state, metadataset: data.MetaDataset):
        def batch_outer_loss(rng, hstate, hparams, metadataset):
            rngs = jax.random.split(rng, utils.tree_length(metadataset))
            outer_loss_vmap = jax.vmap(self.outer_loss, in_axes=(0, None, None, 0, None))
            loss, ((_, _, hstate), metrics) = outer_loss_vmap(
                rngs, hstate, hparams, metadataset, None
            )
            return jnp.mean(loss), (hstate, metrics)

        grad_fn = jax.grad(batch_outer_loss, argnums=2, has_aux=True)
        hgrads, (hstate, metrics) = grad_fn(
            rng, meta_state.hstate, meta_state.hparams, metadataset
        )

        # HACK: Averaging over the model state might result in unexpected behaviour
        # HACK: Averaging might change dtype (e.g. int to float), this simply casts it back
        hstate_dtypes = jtu.tree_map(jnp.dtype, hstate)
        hstate = jtu.tree_map(partial(jnp.mean, axis=0), hstate)
        hstate = jtu.tree_map(jax.lax.convert_element_type, hstate, hstate_dtypes)

        hparams_update, optim_state = self.optim_fn_outer.update(
            hgrads, meta_state.optim, meta_state.hparams
        )
        hparams = optax.apply_updates(meta_state.hparams, hparams_update)

        metrics = {
            "gradnorm_outer": optax.global_norm(hgrads),
            **jtu.tree_map(partial(jnp.mean, axis=0), metrics),
        }

        return MetaLearnerState(hparams=hparams, optim=optim_state, hstate=hstate), metrics
