"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import unittest
from functools import partial

import jax
import jax.numpy as jnp

from metax.data.envs.preference import CompositionalPreference


class CompositionalPreferenceTestCase(unittest.TestCase):
    rng = jax.random.PRNGKey(0)

    def test_experts(self):
        task_support = "disconnected"
        env = CompositionalPreference(
            num_preferences=8,
            num_features=4,
            num_objects=3,
            num_hot=2,
            continuous_combinations=False,
            discount=0.9,
            frac_ood=0.25,
            timelimit=10,
            task_support=task_support,
            seed=2022,
        )

        @partial(jax.jit, static_argnums=(1,))
        @partial(jax.vmap, in_axes=(0, None))
        def sample_embeddings(rng, mode):
            return env.reset_goal(rng, mode)[0]

        embeddings = sample_embeddings(jax.random.split(self.rng, 10000), "train")
        assert len(jnp.unique(embeddings, axis=0)) == len(env.preference_in_dist)

        embeddings = sample_embeddings(jax.random.split(self.rng, 10000), "test")
        assert len(jnp.unique(embeddings, axis=0)) == len(env.preference_in_dist)

        embeddings = sample_embeddings(jax.random.split(self.rng, 10000), "ood")
        assert len(jnp.unique(embeddings, axis=0)) == len(env.preference_out_dist)

    def test_demonstrate(self):
        env = CompositionalPreference(
            num_preferences=8,
            num_features=4,
            num_objects=3,
            num_hot=2,
            continuous_combinations=False,
            discount=0.9,
            frac_ood=0.25,
            timelimit=10,
            task_support="random",
            seed=2022,
        )

        @jax.jit
        @partial(jax.vmap, in_axes=(0, None))
        def optimal_trajectories_given_goal(rng, goal):
            state, emission = env.reset(rng, goal)
            return env.demonstrate(rng, state)

        for r in jax.random.split(self.rng, 10000):
            goal, _ = env.reset_goal(r, mode="train")
            traj, action_values = optimal_trajectories_given_goal(jax.random.split(r, 2000), goal)
            assert jnp.isfinite(action_values).all()
            assert jnp.isfinite(traj.observation).all()
