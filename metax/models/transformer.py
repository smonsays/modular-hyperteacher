"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class MultiHeadAttentionWithSkip(hk.Module):
    """Multi-headed attention (MHA) block with skip connections.

    This module is intended for attending over sequences of vectors.

    Rough sketch:
    - Compute keys (K), queries (Q), and values (V) as projections of inputs.
    - Attention weights are computed as W = softmax(QK^T / sqrt(key_size)).
    - Output is another projection of WV^T.

    For more detail, see the original Transformer paper:
      "Attention is all you need" https://arxiv.org/abs/1706.03762.

    Glossary of shapes:
    - T: Sequence length.
    - D: Vector (embedding) size.
    - H: Number of attention heads.
    """

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        *,
        w_init: Optional[hk.initializers.Initializer] = None,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        """Initialises the module.

        Args:
          num_heads: Number of independent attention heads (H).
          key_size: The size of keys (K) and queries used for attention.
          w_init: Initialiser for weights in the linear map.
          value_size: Optional size of the value projection (V). If None, defaults
            to the key size (K).
          model_size: Optional size of the output embedding (D'). If None, defaults
            to the key size multiplied by the number of heads (K * H).
          name: Optional name for this module.
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads

        if w_init is None:
            raise ValueError("Please provide a weight initializer: `w_init`.")

        self.w_init = w_init

    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more 'batch-like' leading dimensions.

        Args:
          query: Embeddings sequence used to compute queries; shape [..., T', D_q].
          key: Embeddings sequence used to compute keys; shape [..., T, D_k].
          value: Embeddings sequence used to compute values; shape [..., T, D_v].
          mask: Optional mask applied to attention weights; shape [..., H=1, T', T].

        Returns:
          A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [..., T', D'].
        """

        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        *leading_dims, sequence_length, _ = query.shape
        projection = self._linear_projection

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        query_heads = projection(query, self.key_size, "query")  # [T', H, Q=K]
        key_heads = projection(key, self.key_size, "key")  # [T, H, K]
        value_heads = projection(value, self.value_size, "value")  # [T, H, V]

        # Compute attention weights.
        attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
        attn_logits = attn_logits / np.sqrt(self.key_size).astype(key.dtype)
        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim}."
                )
            attn_logits = jnp.where(mask, attn_logits, -1e30)
        attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
        attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

        # Skip connection over the attention mechanism
        attn_skip = attn + query

        # Apply another projection with nonlinearity
        # NOTE: haiku treats any leading dims as batch dimensions
        final_projection = hk.Linear(self.model_size, w_init=self.w_init)
        out = jax.nn.relu(final_projection(attn_skip))

        # NOTE: set-transformer uses another skip connection here
        # out = out + attn_skip

        return out

    @hk.transparent
    def _linear_projection(
        self,
        x: jnp.ndarray,
        head_size: int,
        name: Optional[str] = None,
    ) -> jnp.ndarray:
        y = hk.Linear(self.num_heads * head_size, w_init=self.w_init, name=name)(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_heads, head_size))


class SetAttention(hk.Module):
    def __init__(self, key_size: int, num_heads: int, name: Optional[str] = None):
        super().__init__(name)
        self.key_size = key_size
        self.num_heads = num_heads

    def __call__(self, query, key):
        out = MultiHeadAttentionWithSkip(
            num_heads=self.num_heads,
            key_size=self.key_size,
            model_size=self.key_size * self.num_heads,
            w_init=hk.initializers.VarianceScaling(1.0),
        )(query, key, key)

        return out


class PoolingAttention(hk.Module):
    def __init__(
        self,
        dim_seeds: int,
        num_seeds: int,
        key_size: int,
        num_heads: int,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.dim_seeds = dim_seeds
        self.num_seeds = num_seeds
        self.mab = SetAttention(key_size, num_heads, name="mab")

    @property
    def seeds(self):
        return hk.get_parameter(
            name="seeds",
            shape=[1, self.num_seeds, self.dim_seeds],
            init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),  # glorot uniform
        )

    def __call__(self, x):
        return self.mab(jnp.repeat(self.seeds, repeats=x.shape[0], axis=0), x)


class Transformer(hk.Module):
    def __init__(self, output_size, key_size, num_heads, num_seeds=1, name: Optional[str] = None):
        super().__init__(name)
        self.output_size = output_size
        self.key_size = key_size
        self.num_heads = num_heads

    def __call__(self, x, is_training: bool):
        x = hk.Linear(self.num_heads * self.key_size)(x)
        x = SetAttention(num_heads=self.num_heads, key_size=self.key_size)(x, x)
        x = SetAttention(num_heads=self.num_heads, key_size=self.key_size)(x, x)
        x = hk.Flatten()(x)  # flatten heads and channels

        if self.output_size is not None:
            # linearly project from key_size * num_heads to output
            x = hk.Linear(self.output_size)(x)

        return x


class SetTransformer(hk.Module):
    def __init__(self, output_size, key_size, num_heads, num_seeds=1, name: Optional[str] = None):
        super().__init__(name)
        self.output_size = output_size
        self.key_size = key_size
        self.num_heads = num_heads
        self.num_seeds = num_seeds

    def __call__(self, x, is_training: bool):
        # encoder
        x = hk.Linear(self.num_heads * self.key_size)(x)
        x = SetAttention(key_size=self.key_size, num_heads=self.num_heads, name="sab")(x, x)

        # decoder
        x = PoolingAttention(
            dim_seeds=self.num_heads * self.key_size,
            num_seeds=self.num_seeds,
            key_size=self.key_size,
            num_heads=self.num_heads,
            name="pab",
        )(x)

        x = hk.Flatten()(x)  # flatten seeds and channels

        if self.output_size is not None:
            # linearly project from key_size * num_heads to output
            x = hk.Linear(self.output_size)(x)

        return x
