"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Dict, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp

from metax import energy, models
from metax.utils import dict_filter
from metax.utils.utils import dict_combine

from .base import MetaModule


class MetaHypernetworkMetaParams(NamedTuple):
    bank_input: Dict
    bank_hidden: Dict
    bank_output: Dict
    embedding_init_input: Dict
    embedding_init_hidden: Dict
    embedding_init_output: Dict


class MetaHypernetworkParams(NamedTuple):
    embedding_input: Dict
    embedding_hidden: Dict
    embedding_output: Dict
    bias: Dict


class MetaHypernetworkMetaState(NamedTuple):
    bank_input: Dict
    bank_hidden: Dict
    bank_output: Dict


class MetaHypernetworkState(NamedTuple):
    embedding_input: Dict
    embedding_hidden: Dict
    embedding_output: Dict
    target_network_input: Dict
    target_network_hidden: Dict
    target_network_output: Dict


class MetaHypernetwork(MetaModule):
    def __init__(
        self,
        loss_fn_inner,
        loss_fn_outer,
        target_network_input,
        target_network_hidden,
        target_network_output,
        *,
        input_shape,
        output_dim,
        hidden_dim,
        num_templates,
        chunking,
        weight_generator,
        embedding_nonlinearity,
        embedding_dropout,
        embedding_norm_stop_grad,
        embedding_normalization,
        embedding_constant_init,
        hnet_init,
        l1_reg,
        l2_reg,
        zero_threshold,
        fast_bias,
    ) -> None:
        super().__init__(loss_fn_inner, loss_fn_outer)

        if l1_reg is not None:
            self.loss_fn_inner += energy.LNorm(
                reg_strength=l1_reg,
                param_keys=["embedding_input", "embedding_hidden", "embedding_output"],
                order=1,
                reduction="sum",
            )

        if l2_reg is not None:
            self.loss_fn_inner += energy.LNorm(
                reg_strength=l2_reg,
                param_keys=["embedding_input", "embedding_hidden", "embedding_output"],
                order=2,
                reduction="sum",
            )

        self.target_network_input = target_network_input
        self.target_network_hidden = target_network_hidden
        self.target_network_output = target_network_output
        self.embedding_constant_init = embedding_constant_init
        self.zero_threshold = zero_threshold
        self.output_dim = output_dim
        self.fast_bias = fast_bias

        # Generate placeholder parameters for each target network
        params_target_input, _ = self.target_network_input.init(
            jax.random.PRNGKey(0), jnp.empty((1, *input_shape)), is_training=True
        )
        params_target_hidden, _ = self.target_network_hidden.init(
            jax.random.PRNGKey(0), jnp.empty((1, hidden_dim)), is_training=True
        )
        params_target_output, _ = self.target_network_output.init(
            jax.random.PRNGKey(0), jnp.empty((1, hidden_dim)), is_training=True
        )

        # Initialize haiku hypernetwork models
        self.hnet_input = models.Hypernetwork(
            params_target=params_target_input,
            type=weight_generator,
            chunking=False,
            num_templates=num_templates,
            embedding_nonlinearity=embedding_nonlinearity,
            embedding_dropout=embedding_dropout,
            embedding_norm_stop_grad=embedding_norm_stop_grad,
            embedding_normalization=embedding_normalization,
            embedding_constant_init=embedding_constant_init,
            weight_generator_init=hnet_init,
        )

        self.hnet_hidden = models.Hypernetwork(
            params_target=params_target_hidden,
            type=weight_generator,
            chunking=chunking,
            num_templates=num_templates,
            embedding_nonlinearity=embedding_nonlinearity,
            embedding_dropout=embedding_dropout,
            embedding_norm_stop_grad=embedding_norm_stop_grad,
            embedding_normalization=embedding_normalization,
            embedding_constant_init=embedding_constant_init,
            weight_generator_init=hnet_init,
        )

        self.hnet_output = models.Hypernetwork(
            params_target=params_target_output,
            type=weight_generator,
            chunking=chunking,
            num_templates=num_templates,
            embedding_nonlinearity=embedding_nonlinearity,
            embedding_dropout=embedding_dropout,
            embedding_norm_stop_grad=embedding_norm_stop_grad,
            embedding_normalization=embedding_normalization,
            embedding_constant_init=embedding_constant_init,
            weight_generator_init=hnet_init,
        )

        if self.fast_bias:
            self.bias_output = hk.without_apply_rng(hk.transform(lambda x: hk.Bias()(x)))

    def postprocess_param(self, params, hparams):
        if self.reg_strength is None or self.soft_l1_thr == 0:
            return params
        if self.reg_type == "l1":
            for key in self.reg_param_keys:
                params = params._replace(
                    **{
                        key: jax.tree_util.tree_map(
                            lambda p: p * (jnp.abs(p) > self.soft_l1_thr * self.reg_strength),
                            getattr(params, key),
                        )
                    }
                )
        elif self.reg_type == "imaml_l1":
            for key, hkey in self.reg_params_key_map.items():
                params = params._replace(
                    **{
                        key: jax.tree_util.tree_map(
                            lambda p, p_i: jnp.where(
                                jnp.abs(p - p_i) > self.soft_l1_thr * self.reg_strength, p, p_i
                            ),
                            getattr(params, key),
                            getattr(hparams, hkey),
                        )
                    }
                )
        return params

    def __call__(self, rng, state, hstate, params, hparams, input, is_training):
        rng_in1, rng_in2, rng_hid1, rng_hid2, rng_out1, rng_out2 = jax.random.split(rng, 6)

        if self.zero_threshold > 0:
            # Zero out embedding smaller than the threshold, but with straight through gradient.
            def threshold_st(p):
                return p + jax.lax.stop_gradient(p * (jnp.abs(p) > self.zero_threshold) - p)

            for key in ["embedding_input", "embedding_hidden", "embedding_output"]:
                params = params._replace(
                    **{key: jax.tree_util.tree_map(threshold_st, getattr(params, key))}
                )

        # Input hypernetwork + target network
        params_input_hnet = dict_combine(hparams.bank_input, params.embedding_input)
        state_hnet_input = dict_combine(hstate.bank_input, state.embedding_input)
        params_target_input, state_hnet_input = self.hnet_input.apply(
            params_input_hnet, state_hnet_input, rng_in1, is_training
        )
        x, state_target_input = self.target_network_input.apply(
            params_target_input, state.target_network_input, rng_in2, input, is_training
        )
        # Hidden hypernetwork + target network
        params_hidden_hnet = dict_combine(hparams.bank_hidden, params.embedding_hidden)
        state_hnet_hidden = dict_combine(hstate.bank_hidden, state.embedding_hidden)
        params_target_hidden, state_hnet_hidden = self.hnet_hidden.apply(
            params_hidden_hnet, state_hnet_hidden, rng_hid1, is_training
        )
        x, state_target_hidden = self.target_network_hidden.apply(
            params_target_hidden, state.target_network_hidden, rng_hid2, x, is_training
        )
        # Output hypernetwork + target network
        params_output_hnet = dict_combine(hparams.bank_output, params.embedding_output)
        state_hnet_output = dict_combine(hstate.bank_output, state.embedding_output)
        params_target_output, state_hnet_output = self.hnet_output.apply(
            params_output_hnet, state_hnet_output, rng_out1, is_training
        )
        out, state_target_output = self.target_network_output.apply(
            params_target_output, state.target_network_output, rng_out2, x, is_training
        )

        if self.fast_bias:
            out = self.bias_output.apply(params.bias, out)

        state = MetaHypernetworkState(
            embedding_input=dict_filter(state_hnet_input, "embedding"),
            embedding_hidden=dict_filter(state_hnet_hidden, "embedding"),
            embedding_output=dict_filter(state_hnet_output, "embedding"),
            target_network_input=state_target_input,
            target_network_hidden=state_target_hidden,
            target_network_output=state_target_output,
        )
        hstate = MetaHypernetworkMetaState(
            bank_input=dict_filter(state_hnet_input, "bank"),
            bank_hidden=dict_filter(state_hnet_hidden, "bank"),
            bank_output=dict_filter(state_hnet_output, "bank"),
        )

        return out, (state, hstate)

    def reset_hparams(self, rng, sample_input):
        rng_input, rng_hidden, rng_output = jax.random.split(rng, 3)

        # Init each hypernetwork, split embedding from weight_bank params
        params_input, state_input = self.hnet_input.init(rng_input, is_training=True)
        hparams_bank_input = dict_filter(params_input, "bank")
        hparams_embedding_input = dict_filter(params_input, "embedding")
        hstate_bank_input = dict_filter(state_input, "bank")

        params_hidden, state_hidden = self.hnet_hidden.init(rng_hidden, is_training=True)
        hparams_bank_hidden = dict_filter(params_hidden, "bank")
        hparams_embedding_hidden = dict_filter(params_hidden, "embedding")
        hstate_bank_hidden = dict_filter(state_hidden, "bank")

        params_output, state_output = self.hnet_output.init(rng_output, is_training=True)
        hparams_bank_output = dict_filter(params_output, "bank")
        hparams_embedding_output = dict_filter(params_output, "embedding")
        hstate_bank_output = dict_filter(state_output, "bank")

        hparams = MetaHypernetworkMetaParams(
            bank_input=hparams_bank_input,
            bank_hidden=hparams_bank_hidden,
            bank_output=hparams_bank_output,
            embedding_init_input=hparams_embedding_input,
            embedding_init_hidden=hparams_embedding_hidden,
            embedding_init_output=hparams_embedding_output,
        )

        hstate = MetaHypernetworkMetaState(
            bank_input=hstate_bank_input,
            bank_hidden=hstate_bank_hidden,
            bank_output=hstate_bank_output,
        )

        return hparams, hstate

    def reset_params(self, rng, hparams, hstate, sample_input):
        rng_input, rng_hidden, rng_output, rng_bias = jax.random.split(rng, 4)

        # Init each hypernetwork to extract a fresh state for the embeddings and the target networks
        params_input, state_input = self.hnet_input.init(rng_input, is_training=True)
        state_embed_input = dict_filter(state_input, "embedding")
        _, state_target_input = self.target_network_input.init(
            rng_input, sample_input, is_training=True
        )

        params_hidden, state_hidden = self.hnet_hidden.init(rng_hidden, is_training=True)
        state_embed_hidden = dict_filter(state_hidden, "embedding")
        _, state_target_hidden = self.target_network_hidden.init(
            rng_hidden, sample_input, is_training=True
        )

        params_output, state_output = self.hnet_output.init(rng_output, is_training=True)
        state_embed_output = dict_filter(state_output, "embedding")
        _, state_target_output = self.target_network_output.init(
            rng_output, sample_input, is_training=True
        )
        if self.fast_bias:
            params_bias = self.bias_output.init(
                rng_bias, jnp.empty((sample_input.shape[0], self.output_dim))
            )
        else:
            params_bias = dict()

        if self.embedding_constant_init:
            # If the embedding should remain constant, use the re-initialized value of the meta model.
            hparams_embedding_input = dict_filter(params_input, "embedding")
            hparams_embedding_hidden = dict_filter(params_hidden, "embedding")
            hparams_embedding_output = dict_filter(params_output, "embedding")
        else:
            hparams_embedding_input = hparams.embedding_init_input
            hparams_embedding_hidden = hparams.embedding_init_hidden
            hparams_embedding_output = hparams.embedding_init_output

        params = MetaHypernetworkParams(
            embedding_input=hparams_embedding_input,
            embedding_hidden=hparams_embedding_hidden,
            embedding_output=hparams_embedding_output,
            bias=params_bias,
        )

        state = MetaHypernetworkState(
            embedding_input=state_embed_input,
            embedding_hidden=state_embed_hidden,
            embedding_output=state_embed_output,
            target_network_input=state_target_input,
            target_network_hidden=state_target_hidden,
            target_network_output=state_target_output,
        )

        return params, state
