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

from metax.data.envs.grid import (MOVES, CompositionalGrid,
                                  CompositionalGridGoal)


class CompositionalGridTestCase(unittest.TestCase):
    rng = jax.random.PRNGKey(0)

    def test_reset(self):
        env = CompositionalGrid(
            grid_size := 7,
            num_interactions := 4,
            num_mazes := 2,
            num_objects := 3,
            num_distractors := 2,
            frac_ood := 0.2,
            task_support := "1_hot",
            seed := 2022,
        )
        state, emission = env.reset(rng=self.rng)
        # state, emission = jax.jit(env.reset)(rng=self.rng)
        assert state.timestep == 0
        assert emission.observation.shape == (grid_size, grid_size, num_objects + 2)
        assert jnp.all(jnp.concatenate((env.tasks_in_dist, env.tasks_out_dist)).shape == env.tasks_all.shape)
        assert len(jnp.unique(jnp.concatenate((env.tasks_in_dist, env.tasks_out_dist)), axis=1)) == len(env.tasks_all)

    def test_step(self):
        env = CompositionalGrid(
            grid_size := 7,
            num_interactions := 4,
            num_mazes := 6,
            num_objects := 5,
            num_distractors := 2,
            frac_ood := 0.2,
            task_support := "random",
            seed := 2022,
        )
        goal = CompositionalGridGoal(direction := 0, interaction := 1, maze := 2, object := 3)
        state, emission = env.reset(rng=self.rng, goal=goal)
        pos_goal, pos_agent = jnp.array([0, 0]), jnp.array([6, 3])
        assert jnp.all(state.positions[0] == pos_goal)
        assert jnp.all(state.positions[-1] == pos_agent)

        # Maze layout env.mazes[state.goal.maze]:
        # [g, 0, 0, 0, 0, 0, 0],
        # [0, 1, 1, 1, 1, 1, 0],
        # [0, 1, 0, 0, 0, 1, 0],
        # [0, 1, 1, 1, 0, 1, 0],
        # [0, 0, 0, 0, 0, 1, 0],
        # [0, 1, 1, 1, 1, 1, 0],
        # [0, 0, 0, a, 0, 0, 0]

        state, emission = env.step(rng=self.rng, env_state=state, action=MOVES.UP.value)
        assert state.timestep == 1
        assert jnp.all(state.positions[-1] == pos_agent)
        assert emission.observation[pos_agent[0], pos_agent[1], -1] == 1.0
        assert jnp.sum(emission.observation[:, :, -1] == 1) == 1
        assert emission.reward == 0.0

        state, emission = env.step(rng=self.rng, env_state=state, action=MOVES.LEFT.value)
        state, emission = env.step(rng=self.rng, env_state=state, action=MOVES.LEFT.value)
        state, emission = env.step(rng=self.rng, env_state=state, action=MOVES.LEFT.value)
        state, emission = env.step(rng=self.rng, env_state=state, action=MOVES.UP.value)
        state, emission = env.step(rng=self.rng, env_state=state, action=MOVES.UP.value)
        state, emission = env.step(rng=self.rng, env_state=state, action=MOVES.UP.value)
        state, emission = env.step(rng=self.rng, env_state=state, action=MOVES.UP.value)
        state, emission = env.step(rng=self.rng, env_state=state, action=MOVES.UP.value)
        state, emission = env.step(rng=self.rng, env_state=state, action=MOVES.UP.value)
        assert not state.done
        assert emission.reward == 0.0
        state, emission = env.step(rng=self.rng, env_state=state, action=len(MOVES) + goal.interaction)
        assert jnp.all(state.positions[0] == state.positions[-1])
        assert emission.reward == 1.0
        assert emission.done
        assert state.done

    def test_demonstrate(self):
        env = CompositionalGrid(
            grid_size := 7,
            num_interactions := 4,
            num_mazes := 5,
            num_objects := 4,
            num_distractors := 2,
            frac_ood := 0.2,
            task_support := "random",
            seed := 2022,
        )

        @partial(jax.vmap, in_axes=(0, None))
        def optimal_trajectories_given_goal(rng, goal):
            state, emission = env.reset(rng, goal)
            return env.demonstrate(rng, state)

        rngs = jax.random.split(self.rng, 128)
        goal = CompositionalGridGoal(direction := 0, interaction := 1, maze := 2, object := 3)
        goal_interaction = len(MOVES) + goal.interaction
        trajectories, actions = optimal_trajectories_given_goal(rngs, goal)

        assert jnp.allclose(jnp.sum(trajectories.reward, axis=1), 1.0)
        assert jnp.all(jnp.any(trajectories.done, axis=1))
        assert jnp.all(actions[trajectories.done] == goal_interaction)
        assert not jnp.any(actions[:, :-1][~trajectories.done[:, 1:]] == goal_interaction)
        assert jnp.all(jnp.sum(~trajectories.done, axis=-1) >= env.grid_size), "Goal should be at least grid_size steps away"

    def test_generate_random_maze(self):
        grid_list = [
            CompositionalGrid.generate_random_maze(grid_size=11, seed=i) for i in range(9)
        ]

        # Use matplotlib to visualize the mazes on a grid of images
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid

        fig = plt.figure()
        axs = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )

        for ax, im in zip(axs, grid_list):
            # Iterating over the grid returns the Axes.
            ax.imshow(~im)

        plt.show()
