"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import chex
import haiku as hk
import jax


class DeepSet(hk.Module):
    def __init__(self, output_size, hidden_dim, pool):
        super().__init__()
        self.output_size = output_size
        self.encoder = hk.Sequential([
            # hk.Linear(hidden_dim),
            # jax.nn.relu,
            # hk.Linear(hidden_dim),
            # jax.nn.relu,
            # hk.Linear(hidden_dim),
            jax.nn.relu,
            hk.Linear(hidden_dim),
        ])
        self.pool = pool
        self.decoder = hk.Sequential([
            hk.Linear(hidden_dim),
            jax.nn.relu,
        ])

    def __call__(self, x, is_training: bool):
        # Input needs to be rank 3 for pooling operation to function properly
        chex.assert_rank(x, 3)

        x = self.encoder(x)
        x = self.pool(x, axis=1)
        x = self.decoder(x)

        if self.output_size is not None:
            x = hk.Linear(self.output_size)(x)

        return x.squeeze()
