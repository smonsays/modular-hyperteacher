"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Dict, NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from metax import energy

from .base import MetaModule


class LearnedL2MetaParams(NamedTuple):
    base_learner: Dict
    log_lambda: Dict


class LearnedL2Params(NamedTuple):
    base_learner: Dict


class LearnedL2State(NamedTuple):
    base_learner: Dict


class LearnedL2MetaState(NamedTuple):
    pass


class LearnedL2(MetaModule):
    def __init__(self, loss_fn_inner, loss_fn_outer, base_learner, l2_reg, fixed_init):
        super().__init__(loss_fn_inner=loss_fn_inner, loss_fn_outer=loss_fn_outer)
        self.base_learner = base_learner
        self.l2_reg = l2_reg
        self.fixed_init = fixed_init

        self.loss_fn_inner += energy.LearnedL2(
            key_map={"base_learner": "log_lambda"},
            reduction="sum"
        )

    def __call__(self, rng, state, hstate, params, hparams, input, is_training):
        output, state = self.base_learner.apply(
            params.base_learner, state.base_learner, rng, input, is_training
        )

        return output, (LearnedL2State(state), hstate)

    def reset_hparams(self, rng, sample_input):
        base_learner, _ = self.base_learner.init(rng, sample_input, is_training=True)
        log_lambda = jtu.tree_map(
            lambda x: jnp.log(self.l2_reg) * jnp.ones_like(x), base_learner
        )
        return LearnedL2MetaParams(base_learner, log_lambda), LearnedL2MetaState()

    def reset_params(self, rng, hparams, hstate, sample_input):

        _, base_learner_state = self.base_learner.init(rng, sample_input, is_training=True)

        if self.fixed_init:
            # Keep initialisation of parameters fixed instead of meta-learning it
            params = LearnedL2Params(jax.lax.stop_gradient(hparams.base_learner))
        else:
            params = LearnedL2Params(hparams.base_learner)

        return params, LearnedL2State(base_learner_state)
