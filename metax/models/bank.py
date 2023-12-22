"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import math
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp


class Embedding(hk.Module):
    def __init__(
        self,
        num_embeddings,
        num_templates,
        nonlinearity,
        dropout_rate: Optional[float] = None,
        normalization=False,
        normalization_stop_grad=False,
        constant_init=False,
        double_embedding=False,
        name=None,
    ):
        super().__init__(name=name)
        self.num_embeddings = num_embeddings
        self.num_templates = num_templates
        self.nonlinearity = nonlinearity
        self.dropout_rate = dropout_rate
        self.normalization = normalization
        self.normalization_stop_grad = normalization_stop_grad
        self.use_double_embedding = double_embedding
        self.constant_init = constant_init

    @property
    def embeddings(self):
        if self.constant_init:
            initializer = hk.initializers.Constant(1.0)
        else:
            initializer = hk.initializers.VarianceScaling(1.0, "fan_in", "uniform")

        return hk.get_parameter(
            name="embeddings",
            shape=[self.num_embeddings, self.num_templates],
            init=initializer
        )

    @property
    def double_embedding(self):
        return hk.get_parameter(
            name="double_embedding",
            shape=[self.num_embeddings, self.num_templates],
            init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),  # lecun_uniform
        )

    def __call__(self, is_training: bool):
        x = self.embeddings

        if self.nonlinearity == "l1_ball":
            import jaxopt
            x = jax.vmap(jaxopt.projection.projection_l1_ball, in_axes=1, out_axes=1)(x)
        elif self.nonlinearity == "relu":
            x = jax.nn.relu(x)
        elif self.nonlinearity == "softmax":
            x = jax.nn.softmax(x, axis=1)
        else:
            assert self.nonlinearity is None or self.nonlinearity == "linear"

        if self.use_double_embedding:
            x = x * self.double_embedding

        if self.normalization:
            scaling = math.sqrt(self.num_templates) / jnp.linalg.norm(x, axis=1, keepdims=True)
            if self.normalization_stop_grad:
                scaling = jax.lax.stop_gradient(scaling)
            # Normalize embedding to have an elementwise norm of 1
            x = x * scaling

        if self.dropout_rate is not None:
            dropout_rate = self.dropout_rate if is_training else 0.0
            x = hk.dropout(hk.next_rng_key(), dropout_rate, x)

        return x


class Bank(hk.Module):
    def __init__(self, num_templates, chunk_dim, name=None):
        super().__init__(name=name)
        self.num_templates = num_templates
        self.chunk_dim = chunk_dim


class LinearBank(Bank):
    def __init__(self, num_templates, chunk_dim, init="default", name=None):
        super().__init__(num_templates, chunk_dim, name)
        self.init = init

    def __call__(self, embedding):
        if self.init == "default":
            return hk.Linear(
                output_size=self.chunk_dim,
                with_bias=False,
                w_init=hk.initializers.TruncatedNormal(stddev=1.0 / math.sqrt(self.num_templates)),
            )(embedding)
        elif self.init == "bias_hyper_init":
            return hk.Linear(
                output_size=self.chunk_dim,
                with_bias=True,
                b_init=hk.initializers.TruncatedNormal(stddev=1.0 / math.sqrt(self.num_templates)),
                w_init=hk.initializers.Constant(0.0),
            )(embedding)


class MLPBank(Bank):
    def __init__(self, num_templates, chunk_dim, hidden_dims, init="default", name=None):
        super().__init__(num_templates, chunk_dim, name)
        self.init = init
        self.hidden_dims = hidden_dims

    def __call__(self, embedding):
        body = hk.nets.MLP(output_sizes=self.hidden_dims, activate_final=True)
        if self.init == "default":
            head = hk.Linear(
                output_size=self.chunk_dim,
                with_bias=False,
                w_init=hk.initializers.TruncatedNormal(
                    stddev=1.0 / math.sqrt(self.hidden_dims[-1])
                ),
            )
        elif self.init == "bias_hyper_init":
            head = hk.Linear(
                output_size=self.chunk_dim,
                with_bias=True,
                b_init=hk.initializers.TruncatedNormal(
                    stddev=1.0 / math.sqrt(self.hidden_dims[-1])
                ),
                w_init=hk.initializers.Constant(0.0),
            )

        return hk.Sequential([body, head])(embedding)


class DeepBank(Bank):
    def __init__(self, num_templates, chunk_dim, hidden_dim, init="default", name=None):
        super().__init__(num_templates, chunk_dim, name)
        self.init = init
        self.hidden_dim = hidden_dim

    def __call__(self, embedding):
        body = hk.Sequential(
            [
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                hk.Linear(self.hidden_dim),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.elu,
                hk.Linear(self.hidden_dim),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.elu,
                hk.Linear(self.hidden_dim),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.elu,
                hk.Linear(self.hidden_dim),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.elu,
            ]
        )
        if self.init == "default":
            head = hk.Linear(
                output_size=self.chunk_dim,
                with_bias=False,
                w_init=hk.initializers.TruncatedNormal(stddev=1.0 / math.sqrt(self.hidden_dim)),
            )
        elif self.init == "bias_hyper_init":
            head = hk.Linear(
                output_size=self.chunk_dim,
                with_bias=True,
                b_init=hk.initializers.TruncatedNormal(stddev=1.0 / math.sqrt(self.hidden_dim)),
                w_init=hk.initializers.Constant(0.0),
            )

        return hk.Sequential([body, head])(embedding)
