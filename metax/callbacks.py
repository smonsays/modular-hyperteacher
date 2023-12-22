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
from jax.flatten_util import ravel_pytree

from metax import utils
from metax.learner.base import MetaLearnerState
from metax.models.regression import RidgeRegression
from metax.utils.pytree import tree_length
from metax.utils.utils import dict_combine, flatcat


def callback_hparams(rng, ctx, meta_state: MetaLearnerState):
    return {"hparams": utils.flatten_dict(meta_state.hparams)}


def get_callback_params(metavalidset):

    dataset_batch = next(iter(metavalidset))
    dataset = jtu.tree_map(lambda x: x[0], dataset_batch)

    @partial(jax.jit, static_argnames=("ctx"))
    def callback_params(rng, ctx, meta_state: MetaLearnerState):
        """
        Log adapted params and the amount they changed
        """
        hstate, hparams = meta_state.hstate, meta_state.hparams

        def adapt_params(rng):
            _, (((_, params_adapted), (_, params_init), _), _) = ctx.meta_learner.outer_loss(
                rng, hstate, hparams, dataset, ctx.steps_inner_test
            )
            return params_init, params_adapted

        params_init, params_adapted = adapt_params(rng)
        params_change = jtu.tree_map(lambda x, y: x - y, params_adapted, params_init)

        return {
            "params_init": utils.flatten_dict(params_init._asdict()),
            "params_change": utils.flatten_dict(params_change._asdict())
        }
    return callback_params


def get_callback_decode_modules(metatestset, metaoodset):
    meta_batch_test = next(iter(metatestset))
    meta_batch_ood = next(iter(metaoodset))

    @partial(jax.jit, static_argnames=("ctx"))
    def callback_decode_modules(rng, ctx, meta_state: MetaLearnerState):
        hparams, hstate = meta_state.hparams, meta_state.hstate

        # Get raveled, adapted params for both valid and ood tasks
        @jax.vmap
        def adapt_params(rng, dataset):
            _, (((state_adapted, params_adapted), (state_init, params_init), _), _) = ctx.meta_learner.outer_loss(
                rng, hstate, hparams, dataset, ctx.steps_inner_test
            )

            return ravel_pytree(params_adapted)[0]

        rng_adapt_test, rng_adapt_ood, rng_reg = jax.random.split(rng, 3)
        rngs_adapt_test = jax.random.split(rng_adapt_test, tree_length(meta_batch_test))
        params_adapted_test = adapt_params(rngs_adapt_test, meta_batch_test)

        rngs_adapt_ood = jax.random.split(rng_adapt_ood, tree_length(meta_batch_ood))
        params_adapted_ood = adapt_params(rngs_adapt_ood, meta_batch_ood)

        # Linearly regress module embeddings based on adapted params for each teacher layer separately
        x_train, y_train = params_adapted_test, meta_batch_test.train.info["embeddings"][:, 0, :]
        x_test, y_test = params_adapted_ood, meta_batch_ood.train.info["embeddings"][:, 0, :]

        # from sklearn.linear_model import Ridge
        # from sklearn.multioutput import MultiOutputRegressor
        # from sklearn.pipeline import make_pipeline
        # from sklearn.preprocessing import StandardScaler
        # log_dict = dict()
        # for layer in range(y_train.shape[1]):
        #     reg = make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=1.0)))
        #     reg = reg.fit(x_train, y_train[:, layer])
        #     coeff = reg.score(x_test, y_test[:, layer])
        #     log_dict[f"decoder_ridge_coeff_layer_{layer}"] = coeff

        @partial(jax.vmap, in_axes=(None, None, None, -1, -1))  # vmap over layers
        @partial(jax.vmap, in_axes=(None, None, None, -1, -1))  # vmap over experts
        def ridge_r2_score(rng, x_train, x_test, y_train, y_test):
            reg_jax = RidgeRegression(feature_dim=params_adapted_test.shape[1], l2_reg=1.0, intercept=True)
            params = reg_jax.init(rng, x_train)
            params = reg_jax.fit(params, x_train, y_train)
            return reg_jax.score(params, x_test, y_test)

        coeffs = ridge_r2_score(rng_reg, x_train, x_test, y_train, y_test)

        log_dict = dict()
        for expert in range(coeffs.shape[0]):
            for layer in range(coeffs.shape[1]):
                log_dict[f"decoder_r2_layer_{layer}_expert_{expert}"] = coeffs[expert, layer]

        return log_dict

    return callback_decode_modules


def get_callback_analyze_embedding(metavalidset):

    meta_batch = next(iter(metavalidset))
    num_tasks = tree_length(meta_batch)

    @partial(jax.jit, static_argnames=("ctx"))
    def callback_analyze_embedding(rng, ctx, meta_state: MetaLearnerState):

        rng_adapt, rng_target = jax.random.split(rng, 2)
        hparams, hstate = meta_state.hparams, meta_state.hstate
        meta_model = ctx.meta_learner.meta_model

        @jax.vmap
        def adapt_params(rng, dataset):
            _, (((state_adapted, params_adapted), (state_init, params_init), _), _) = ctx.meta_learner.outer_loss(
                rng, hstate, hparams, dataset, ctx.steps_inner_test
            )
            return (state_init, params_init), (state_adapted, params_adapted)

        @jax.vmap
        def get_target_params(rng, state, params):
            rng_input, rng_hidden, rng_output = jax.random.split(rng, 3)

            # Input hypernetwork
            params_input_hnet = dict_combine(hparams.bank_input, params.embedding_input)
            state_hnet_input = dict_combine(hstate.bank_input, state.embedding_input)
            params_target_input, _ = meta_model.hnet_input.apply(
                params_input_hnet, state_hnet_input, rng_input, is_training=True
            )

            # Hidden hypernetwork
            params_hidden_hnet = dict_combine(hparams.bank_hidden, params.embedding_hidden)
            state_hnet_hidden = dict_combine(hstate.bank_hidden, state.embedding_hidden)
            params_target_hidden, _ = meta_model.hnet_hidden.apply(
                params_hidden_hnet, state_hnet_hidden, rng_hidden, is_training=True
            )
            # Output hypernetwork
            params_output_hnet = dict_combine(hparams.bank_output, params.embedding_output)
            state_hnet_output = dict_combine(hstate.bank_output, state.embedding_output)
            params_target_output, _ = meta_model.hnet_output.apply(
                params_output_hnet, state_hnet_output, rng_output, is_training=True
            )

            return flatcat(params_target_input), flatcat(params_target_hidden), flatcat(params_target_output)

        rngs_adapt = jax.random.split(rng_adapt, num_tasks)
        (state_init, params_init), (state_adapted, params_adapted) = adapt_params(rngs_adapt, meta_batch)

        rngs_target = jax.random.split(rng_target, num_tasks)
        tparams_init_input, tparams_init_hidden, tparams_init_output = get_target_params(
            rngs_target, state_init, params_init
        )
        tparams_adapted_input, tparams_adapted_hidden, tparams_adapted_output = get_target_params(
            rngs_target, state_adapted, params_adapted
        )

        def average_correlation(embedding):
            """ Average cross-correlation """
            embedding_flat = jtu.tree_flatten(embedding)[0][0].reshape(num_tasks, -1)
            corr_matrix = jnp.corrcoef(embedding_flat)
            mask = jnp.tril(jnp.ones_like(corr_matrix), k=-1)
            return jnp.sum(corr_matrix * mask) / jnp.sum(mask)

        def average_cossim(embedding):
            """ Average cross-cosine-similarity """
            embedding_flat = jtu.tree_flatten(embedding)[0][0].reshape(num_tasks, -1)
            cos_sim_fn = jax.vmap(
                jax.vmap(optax.cosine_similarity, in_axes=(0, None)), in_axes=(None, 0)
            )
            cossim_matrix = cos_sim_fn(embedding_flat, embedding_flat)
            mask = jnp.tril(jnp.ones_like(cossim_matrix), k=-1)
            return jnp.sum(cossim_matrix * mask) / jnp.sum(mask)

        def average_cossim_vs_init(embedding, embedding_init):
            """ Average cosine similiarity with init """
            embedding_flat = jtu.tree_flatten(embedding)[0][0].reshape(num_tasks, -1)
            embedding_init_flat = jtu.tree_flatten(embedding_init)[0][0].reshape(num_tasks, -1)[0]

            @jax.vmap
            def cos_sim_fn(e):
                return optax.cosine_similarity(e, embedding_init_flat)

            return jnp.mean(cos_sim_fn(embedding_flat))

        def is_zero(pytree):
            return jtu.tree_map(lambda x: 1.0 * jnp.isclose(x, 0), pytree)

        def not_zero(pytree):
            return jtu.tree_map(lambda x: 1.0 * ~jnp.isclose(x, 0), pytree)

        log_dict = {
            "embed_cossim_input": average_cossim(params_adapted.embedding_input),
            "embed_cossim_hidden": average_cossim(params_adapted.embedding_hidden),
            "embed_cossim_output": average_cossim(params_adapted.embedding_output),
            "embed_diff_cossim_input": average_cossim_vs_init(
                params_adapted.embedding_input, params_init.embedding_input
            ),
            "embed_diff_cossim_hidden": average_cossim_vs_init(
                params_adapted.embedding_hidden, params_init.embedding_hidden
            ),
            "embed_diff_cossim_output": average_cossim_vs_init(
                params_adapted.embedding_output, params_init.embedding_output
            ),
            "embed_zeros_input": average_cossim(is_zero(params_adapted.embedding_input)),
            "embed_zeros_hidden": average_cossim(is_zero(params_adapted.embedding_hidden)),
            "embed_zeros_output": average_cossim(is_zero(params_adapted.embedding_output)),
            "embed_nonzeros_input": average_cossim(not_zero(params_adapted.embedding_input)),
            "embed_nonzeros_hidden": average_cossim(not_zero(params_adapted.embedding_hidden)),
            "embed_nonzeros_output": average_cossim(not_zero(params_adapted.embedding_output)),
            "tparams_cossim_input": average_cossim(tparams_adapted_input),
            "tparams_cossim_hidden": average_cossim(tparams_adapted_hidden),
            "tparams_cossim_output": average_cossim(tparams_adapted_output),
            "tparams_diff_cossim_input": average_cossim_vs_init(
                tparams_adapted_input, tparams_init_input
            ),
            "tparams_diff_cossim_hidden": average_cossim_vs_init(
                tparams_adapted_hidden, tparams_init_hidden
            ),
            "tparams_diff_cossim_output": average_cossim_vs_init(
                tparams_adapted_output, tparams_init_output
            ),
        }

        return log_dict

    return callback_analyze_embedding
