"""
Copyright (c) Simon Schug, Anja Surina
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

from metax.data import MultitaskDataset
from metax.utils.utils import append_keys

from .base import MetaLearnerInnerGradientDescent


class MultitaskLearnerState(NamedTuple):
    hparams: NamedTuple
    params: NamedTuple
    hstate: NamedTuple
    state: NamedTuple
    optim_outer: optax.OptState
    optim_inner: optax.OptState


class MultitaskLearner(MetaLearnerInnerGradientDescent):
    """
    MultitaskLearner using MetaLearner API
    """
    def __init__(
        self,
        meta_model,
        batch_size,
        steps_inner,
        optim_fn_inner,
        optim_fn_outer,
        num_tasks,
        random_params_init,
    ):
        super().__init__(meta_model, batch_size, steps_inner, optim_fn_inner)
        self.optim_fn_outer = optim_fn_outer
        self.num_tasks = num_tasks
        self.random_params_init = random_params_init

    def reset(self, rng, sample_input):
        """
        Initialize MultitaskLearnerState.
        """
        rng_init_shared, rng_init_specific = jax.random.split(rng, 2)

        hparams_init, hstate_init, optim_outer_init = self.reset_hparams(
            rng_init_shared, sample_input
        )
        params_init, state_init, optim_inner_init = self.reset_params(
            rng_init_specific, hparams_init, hstate_init, sample_input
        )

        return MultitaskLearnerState(
            hparams=hparams_init,
            params=params_init,
            hstate=hstate_init,
            state=state_init,
            optim_outer=optim_outer_init,
            optim_inner=optim_inner_init,
        )

    def reset_params(self, rng, hparams, hstate, sample_input):
        """
        Reset task-specific parameters.
        """
        rng_params_init, rng_params_get = jax.random.split(rng, 2)
        rngs_params_init = jax.random.split(rng_params_init, self.num_tasks)
        rngs_params_get = jax.random.split(rng_params_get, self.num_tasks)

        if self.random_params_init:
            # TODO(@simon): We might want to do this sequentially as allocating a
            #               task-shared param for each task takes a lot of memory for bigger models
            # Create placeholder hparams for each task to initialise all params differently
            batched_reset_hparams = jax.vmap(self.meta_model.reset_hparams, in_axes=(0, None))
            (dummy_hparams, dummy_hstate) = batched_reset_hparams(rngs_params_init, sample_input)

            batched_reset_params = jax.vmap(self.meta_model.reset_params, in_axes=(0, 0, 0, None))
            params_init, state_init = batched_reset_params(
                rngs_params_get, dummy_hparams, dummy_hstate, sample_input
            )
        else:
            # Reset all task-specific params based on the given hparams
            batched_reset_params = jax.vmap(
                self.meta_model.reset_params, in_axes=(0, None, None, None)
            )
            params_init, state_init = batched_reset_params(
                rngs_params_get, hparams, hstate, sample_input
            )

        optim_inner_init = jax.vmap(self.optim_fn_inner.init)(params_init)

        return params_init, state_init, optim_inner_init

    def reset_hparams(self, rng, sample_input):
        """
        Reset shared parameters.
        """
        rng_init, _ = jax.random.split(rng)
        hparams_init, hstate_init = self.meta_model.reset_hparams(rng_init, sample_input)
        optim_outer_init = self.optim_fn_outer.init(hparams_init)

        return hparams_init, hstate_init, optim_outer_init

    def update(self, rng, multi_state: MultitaskLearnerState, dataset: MultitaskDataset):
        """
        Update MultitaskLearnerState on the given dataset batch.
        """
        # Compute grads wrt params, hparams of multi_loss on given dataset
        grad_fn = jax.grad(self.multi_loss_fn, argnums=(1, 2), has_aux=True)

        params_selected = jax.tree_map(lambda x: x[dataset.task_id], multi_state.params)
        state_selected = jax.tree_map(lambda x: x[dataset.task_id], multi_state.state)
        optim_inner_state_selected = jax.tree_map(lambda x: x[dataset.task_id], multi_state.optim_inner)

        (grads_params_selected, grads_hparams), ((state_selected, hstate), metrics) = grad_fn(
            rng,
            params_selected,
            multi_state.hparams,
            state_selected,
            multi_state.hstate,
            dataset,
        )

        # Apply updates to params (specific)
        params_update, optim_inner_state_selected = jax.vmap(self.optim_fn_inner.update)(
            grads_params_selected, optim_inner_state_selected, params_selected
        )
        params_selected = optax.apply_updates(params_selected, params_update)

        params = jax.tree_map(lambda x, y: x.at[dataset.task_id].set(y), multi_state.params, params_selected)
        state = jax.tree_map(lambda x, y: x.at[dataset.task_id].set(y), multi_state.state, state_selected)
        optim_inner_state = jax.tree_map(lambda x, y: x.at[dataset.task_id].set(y), multi_state.optim_inner, optim_inner_state_selected)

        # Apply updates to hparams(shared)
        hparams_update, optim_outer_state = self.optim_fn_outer.update(
            grads_hparams, multi_state.optim_outer, multi_state.hparams
        )
        hparams = optax.apply_updates(multi_state.hparams, hparams_update)

        metrics = {
            # To keep metric shapes consistent with other meta-learnes expand
            "gradnorm_inner": jnp.expand_dims(optax.global_norm(grads_params_selected), axis=-1),
            "gradnorm_outer": optax.global_norm(grads_hparams),
            **metrics,
        }

        return (
            MultitaskLearnerState(
                params=params,
                hparams=hparams,
                state=state,
                hstate=hstate,
                optim_inner=optim_inner_state,
                optim_outer=optim_outer_state,
            ),
            metrics,
        )

    def multi_loss_fn(
        self,
        rng,
        params_selected,
        hparams,
        state_selected,
        hstate,
        dataset: MultitaskDataset,
    ):
        """
        Compute loss_fn_inner on task_specific params corresponding to dataset.task_id
        """
        rng_loss, _ = jax.random.split(rng)

        def single_loss_fn(rng, state, hstate, params, hparams, dataset):
            rng_loss, rng_pred = jax.random.split(rng)
            pred, (state, hstate) = self.meta_model(
                rng_pred, state, hstate, params, hparams, dataset.x, True
            )
            loss, metrics = self.meta_model.loss_fn_inner(
                rng_loss,
                pred,
                dataset.y,
                params,
                hparams,
                state,
                hstate,
                dataset.info,
            )

            return loss, ((state, hstate), metrics)

        # NOTE: vmap here prevents batch norm from communicating across tasks
        rngs_loss = jax.random.split(rng_loss, len(dataset.y))
        batched_loss = jax.vmap(single_loss_fn, in_axes=(0, 0, None, 0, None, 0))
        losses, ((state_new, hstate_new), metrics) = batched_loss(
            rngs_loss,
            state_selected,
            hstate,
            params_selected,
            hparams,
            dataset,
        )

        # HACK: Average states for shared params
        state_dtypes = jtu.tree_map(jnp.dtype, hstate_new)
        hstate_new = jtu.tree_map(partial(jnp.mean, axis=0), hstate_new)
        hstate_new = jtu.tree_map(jax.lax.convert_element_type, hstate_new, state_dtypes)

        metrics = {
            "loss_outer": jnp.mean(losses),
            # To keep metric shapes consistent with other meta-learnes keep "inner_step" dim
            **append_keys(jtu.tree_map(lambda x: jnp.mean(x, keepdims=True), metrics), "inner"),
        }

        return jnp.mean(losses), ((state_new, hstate_new), metrics)
