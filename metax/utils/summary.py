from typing import Union

import jax.numpy as jnp
import numpy as np


class SummaryStats:
    def __init__(self, x: Union[np.ndarray, jnp.ndarray]) -> None:
        assert len(x.shape) < 3
        if len(x.shape) > 1:
            # Assuming leading dim is training steps, use final step to evaluate
            x = x[-1, :]

        self.mean = np.mean(x)
        self.std = np.std(x)
        self.sem = np.std(x) / np.prod(x.shape)
        self.min = np.min(x)
        self.max = np.max(x)

    def __repr__(self) -> str:
        return f'<mean={self.mean:.4f}, std={self.std:.4f}, sem={self.sem:.4f}, min={self.min:.4f}, max={self.max:.4f}>'
