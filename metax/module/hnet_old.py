"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import math
from functools import partial
from operator import add, floordiv
from typing import Dict, NamedTuple

import flax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from metax import energy, models
from metax.utils import dict_filter, is_tuple_of_ints

from .base import MetaModule


class HypernetworkMetaParams(NamedTuple):
    bank: Dict
    body_init: Dict
    embedding_init: Dict
    head_init: Dict


class HypernetworkParams(NamedTuple):
    body: Dict
    embedding: Dict
    head: Dict


class HypernetworkMetaState(NamedTuple):
    bank: Dict
    head: Dict


class HypernetworkState(NamedTuple):
    body: Dict
    embedding: Dict
    head: Dict


class Hypernetwork(MetaModule):
    """
    Old version of hnet implementation supporting neuron-wise chunking
    and adapting the biases as fast parameters
    """
    def __init__(
        self,
        loss_fn_inner,
        loss_fn_outer,
        body,
        num_templates,
        chunk_shape,
        input_shape,
        output_dim,
        adapt_head,
        reg_strength,
        weight_generator,
        embed_nonlinearity,
        double_embedding,
        hnet_init,
        params_key="w",
    ):
        super().__init__(loss_fn_inner, loss_fn_outer)
        self.body = body
        self.num_templates = num_templates
        self.chunk_shape = chunk_shape
        self.adapt_head = adapt_head
        self.reg_strength = reg_strength
        self.params_key = params_key

        if reg_strength is not None:
            # Use iMAML regularizer towards meta-learned init
            key_map = {"body": "body_init", "embedding": "embedding_init"}

            if adapt_head:
                key_map["head"] = "head_init"

            self.loss_fn_inner += energy.iMAML(
                reg_strength=reg_strength,
                key_map=key_map,
                reduction="sum"
            )

        # Infer shapes of body params to be generated
        params_shape_dtype, _ = jax.eval_shape(
            partial(body.init, is_training=True),
            jax.random.PRNGKey(0),
            jnp.empty((1, *input_shape)),
        )
        params_shape = jtu.tree_map(jnp.shape, params_shape_dtype)
        params_shape = dict_filter(params_shape, self.params_key)  # NOTE: only generate weights
        num_chunks, dim_chunks = Hypernetwork.get_chunk_sizes(params_shape, chunk_shape)

        self.params_shape = params_shape
        if weight_generator == "linear":
            self.bank = models.LinearBank(params_shape, num_templates, dim_chunks, hnet_init)
        elif weight_generator == "mlp":
            self.bank = models.MLPBank(params_shape, num_templates, dim_chunks, hidden_dims=[50, ], init=hnet_init)
        else:
            raise ValueError

        self.embedding = models.Embedding(
            num_chunks, num_templates, embed_nonlinearity, double_embedding
        )
        self.head = models.Linear(output_dim)

    @staticmethod
    def get_chunk_sizes(params_shape, chunk_shape):
        if chunk_shape is None:
            # No chunking means all params are generated at once
            num_chunks = 1
            dim_chunks_tree = jtu.tree_map(math.prod, params_shape, is_leaf=is_tuple_of_ints)
            dim_chunks = jtu.tree_reduce(add, dim_chunks_tree)

        else:
            # Assuming chunk sizes divides all param shapes without remainder
            def num_chunks_fn(shape):
                return math.prod(jtu.tree_map(floordiv, shape, chunk_shape))

            num_chunks_tree = jtu.tree_map(num_chunks_fn, params_shape, is_leaf=is_tuple_of_ints)
            num_chunks = jtu.tree_reduce(add, num_chunks_tree)
            dim_chunks = math.prod(chunk_shape)

        return num_chunks, dim_chunks

    def __call__(self, rng, state, hstate, params, hparams, input, is_training):

        rng_features, rng_head = jax.random.split(rng)

        features, (state_bank, state_body, state_embed) = self.features(
            rng_features, state, hstate, params, hparams, input, is_training
        )

        # NOTE: Both hstate and state contain state_head, which one is used depends on `adapt_head`
        if self.adapt_head:
            out, state_head = self.head.apply(
                params.head, state.head, rng_head, features, is_training
            )
        else:
            out, state_head = self.head.apply(
                hparams.head_init, hstate.head, rng_head, features, is_training
            )

        state = HypernetworkState(state_body, state_embed, state_head)
        hstate = HypernetworkMetaState(state_bank, state_head)

        return out, (state, hstate)

    def features(self, rng, state, hstate, params, hparams, input, is_training):
        rng_bank, rng_body, rng_embed = jax.random.split(rng, 3)

        # Generate params
        embed, state_embed = self.embedding.apply(params.embedding, state.embedding, rng_embed)
        weights, state_bank = self.bank.apply(hparams.bank, hstate.bank, rng_bank, embed)

        # Combine with non-generated params
        params_body = flax.traverse_util.unflatten_dict({
            **flax.traverse_util.flatten_dict(weights),
            **flax.traverse_util.flatten_dict(params.body)
        })

        # Run forward of target network
        features, state_body = self.body.apply(
            params_body, state.body, rng_body, input, is_training
        )

        return features, (state_bank, state_body, state_embed)

    def reset_hparams(self, rng, sample_input):
        rng_bank, rng_body, rng_embed, rng_head = jax.random.split(rng, 4)

        hparams_embed, state_embed = self.embedding.init(rng_embed)
        embed, _ = self.embedding.apply(hparams_embed, state_embed, rng_embed)

        hparams_bank, state_bank = self.bank.init(rng_bank, embed)

        # Track all params of body but weights as regular params with a learned init
        params_body, state_body = self.body.init(rng_body, sample_input, is_training=True)
        hparams_body = dict_filter(params_body, key=self.params_key, all_but_key=True)

        dummy_features, _ = self.body.apply(
            params_body, state_body, rng_body, sample_input, is_training=True
        )
        hparams_head, state_head = self.head.init(rng_head, dummy_features, is_training=True)

        hparams = HypernetworkMetaParams(hparams_bank, hparams_body, hparams_embed, hparams_head)
        hstate = HypernetworkMetaState(state_bank, state_head)

        return hparams, hstate

    def reset_params(self, rng, hparams, hstate, sample_input):
        # Get params
        if self.adapt_head:
            params_head = hparams.head_init
        else:
            params_head = dict()

        params = HypernetworkParams(
            body=hparams.body_init,
            embedding=hparams.embedding_init,
            head=params_head
        )

        # Get states
        rng_body, rng_embed, rng_head = jax.random.split(rng, 3)
        _, state_embed = self.embedding.init(rng_embed)
        params_body, state_body = self.body.init(rng_body, sample_input, is_training=True)
        dummy_features, _ = self.body.apply(
            params_body, state_body, rng_body, sample_input, is_training=True
        )
        _, state_head = self.head.init(rng_head, dummy_features, is_training=True)

        state = HypernetworkState(body=state_body, embedding=state_embed, head=state_head)

        return params, state
