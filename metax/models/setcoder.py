"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Callable, Optional, Union

import haiku as hk
import jax
import jax.numpy as jnp

from metax.data import Dataset, MultitaskDataset

from .deepset import DeepSet
from .transformer import SetTransformer


class Setcoder(hk.Module):
    def __init__(self, output_dim: int, unravel_fn: Callable, encoder: str, encoder_dim: int, num_classes: Optional[int], name=None):
        super().__init__(name=name)
        self.output_dim = output_dim
        self.unravel_fn = unravel_fn
        self.num_classes = num_classes

        if encoder == "transformer":
            self.encoder = SetTransformer(
                output_size=None,
                key_size=encoder_dim,
                num_heads=encoder_dim,
                num_seeds=1
            )
        elif encoder == "deepsetmax":
            self.encoder = DeepSet(
                output_size=None, hidden_dim=encoder_dim, pool=jnp.max
            )
        elif encoder == "deepsetmean":
            self.encoder = DeepSet(
                output_size=None, hidden_dim=encoder_dim, pool=jnp.mean
            )
        else:
            raise ValueError

    def __call__(self, input: Union[Dataset, MultitaskDataset], is_training):

        if self.num_classes is not None:
            y = jax.nn.one_hot(input.y, self.num_classes)
        else:
            y = input.y

        input_cat = jnp.concatenate((input.x, y), axis=-1)
        z = self.encoder(input_cat, is_training=is_training).squeeze()
        z_proj = hk.Linear(self.output_dim)(z)

        return self.unravel_fn(z_proj)
