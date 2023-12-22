"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import enum
import itertools
from functools import partial
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from chex import Array, PRNGKey
from flax import struct
from scipy.sparse.csgraph import shortest_path

from .base import Environment, EnvironmentInteraction


class MOVES(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class CompositionalGridGoal(NamedTuple):
    direction: int
    interaction: int
    maze: int
    object: int


@struct.dataclass
class CompositionalGridState:
    done: bool
    timestep: int
    distractors: Array
    positions: Array  # object (goal + distractor) and agent positions
    goal: CompositionalGridGoal


class CompositionalGrid(Environment):
    def __init__(
        self,
        grid_size: int,
        num_interactions: int,
        num_mazes: int,
        num_objects: int,
        num_distractors: int,
        frac_ood: float,
        task_support: str,
        seed: int,
    ) -> None:
        super().__init__()
        assert grid_size > 5, "grid_size must be greater than 5"

        self.grid_size = grid_size
        self.num_interactions = num_interactions
        self.num_directions = 4  # split grid into 4 quadrants for the goal position
        self.num_objects = num_objects
        self.num_mazes = num_mazes
        self.num_distractors = num_distractors
        self.frac_ood = frac_ood
        self.task_support = task_support
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)
        self.num_factors = 4  # direction, interaction, maze, object

        # Static matrices
        self._delta_position = jnp.concatenate((
            jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]]),  # up, right, down, left
            jnp.zeros((self.num_interactions, 2), dtype=jnp.int32),  # no movement for interaction
        ))
        size_low, size_high = grid_size // 2, (grid_size // 2) + grid_size % 2
        self._quadrants = jnp.stack((
            np.block([
                [np.ones((size_high, size_high)), np.zeros((size_high, size_low))],
                [np.zeros((size_low, size_high)), np.zeros((size_low, size_low))]
            ]),
            np.block([
                [np.zeros((size_high, size_high)), np.ones((size_high, size_low))],
                [np.zeros((size_low, size_high)), np.zeros((size_low, size_low))]
            ]),
            np.block([
                [np.zeros((size_high, size_high)), np.zeros((size_high, size_low))],
                [np.ones((size_low, size_high)), np.zeros((size_low, size_low))]
            ]),
            np.block([
                [np.zeros((size_high, size_high)), np.zeros((size_high, size_low))],
                [np.zeros((size_low, size_high)), np.ones((size_low, size_low))]
            ]),
        ))

        # Pregenerate possible goals and randomly split into in/out of distribution
        self.tasks_all = np.array(list(itertools.product(
            range(self.num_directions),
            range(self.num_interactions),
            range(self.num_mazes),
            range(self.num_objects),
        )))

        if self.task_support == "non_compositional":
            # in/out split with non-compositional support
            self.tasks_in_dist = np.array(list(itertools.product(
                range(self.num_directions - 1),  # hold out one goal quadrant from in_dist
                range(self.num_interactions),
                range(self.num_mazes),
                range(self.num_objects),
            )))

            @partial(np.vectorize, signature="(k),(n,k)->()")
            def elem_in_array(elem, array):
                return np.any(np.all(elem == array, axis=1))

            self.tasks_out_dist = self.tasks_all[~elem_in_array(self.tasks_all, self.tasks_in_dist)]

        elif "_hot" in self.task_support:
            num_hot = int(self.task_support.split("_")[0])
            mask = jnp.sum(self.tasks_all > 0, axis=1) <= num_hot
            self.tasks_in_dist = jnp.array(self.tasks_all[mask])
            self.tasks_out_dist = jnp.array(self.tasks_all[~mask])

        elif self.task_support == "random":
            self.tasks_all = jax.random.permutation(self.rng, self.tasks_all)
            self.num_ood = int(len(self.tasks_all) * self.frac_ood)
            self.tasks_in_dist = jnp.array(self.tasks_all[: -self.num_ood])
            self.tasks_out_dist = jnp.array(self.tasks_all[-self.num_ood:])

            # Make sure all features for every factor are present in the in-distribution tasks
            assert len(jnp.unique(self.tasks_in_dist[:, 0])) == self.num_directions
            assert len(jnp.unique(self.tasks_in_dist[:, 1])) == self.num_interactions
            assert len(jnp.unique(self.tasks_in_dist[:, 2])) == self.num_mazes
            assert len(jnp.unique(self.tasks_in_dist[:, 3])) == self.num_objects
        else:
            raise ValueError(f"Invalid task support: {self.task_support}")

        assert len(self.tasks_in_dist) > 0
        assert len(self.tasks_out_dist) > 0

        # Create random mazes
        if self.num_mazes > 0:
            self.mazes = jnp.stack([
                self.generate_random_maze(self.grid_size, seed=self.seed + i)
                for i in range(self.num_mazes)
            ])
        else:
            self.mazes = jnp.zeros((1, self.grid_size, self.grid_size))

        # Precompute optimal paths, this is potentially expensive for large grid sizes
        optimal_paths, shortest_paths = list(
            zip(*[self._precompute_optimal_paths(m) for m in self.mazes])
        )
        self.optimal_paths, shortest_paths = jnp.stack(optimal_paths), jnp.stack(shortest_paths)
        self.valid_goal_dist = shortest_paths >= self.grid_size

    @property
    def num_actions(self) -> int:
        return 4 + self.num_interactions

    @property
    def observation_shape(self) -> Tuple[int]:
        # encodes positions of agent, objects and walls
        return (self.grid_size, self.grid_size, self.num_objects + 2)

    def reset_goal(self, rng: PRNGKey, mode: str) -> Array:
        assert mode in ["ood", "test", "train"]
        if mode == "ood":
            task_code = jax.random.choice(rng, self.tasks_out_dist)
        else:
            task_code = jax.random.choice(rng, self.tasks_in_dist)

        task_id = jnp.ravel_multi_index(
            task_code,
            dims=(self.num_directions, self.num_interactions, self.num_mazes, self.num_objects),
            mode="wrap",
        )
        emb_dim = max(self.num_directions, self.num_interactions, self.num_mazes, self.num_objects)
        embedding = jax.nn.one_hot(task_code, emb_dim)

        return CompositionalGridGoal(*task_code), {"task_id": task_id, "embedding": embedding}

    def reset(
        self, rng: PRNGKey, goal: Optional[CompositionalGridGoal] = None
    ) -> Tuple[CompositionalGridState, EnvironmentInteraction]:
        """Resets the environment to a random, initial state"""
        rng_distractor, rng_pos1, rng_pos2, rng_pos3, rng_goal = jax.random.split(rng, 5)

        if goal is None:
            # Sample a goal from train distribution if None specified
            goal, _ = self.reset_goal(rng_goal, mode="train")

        # Sample distractor objects distinct from goal object
        distractors = jax.random.choice(
            key=rng_distractor,
            a=self.num_objects,
            shape=(self.num_distractors,),
            replace=True,
            p=1.0 - (jnp.arange(self.num_objects) == goal.object)
        )

        # Sample distinct, random positions for agent, distractors and the goal respecting direction
        position_goal = jax.random.choice(
            key=rng_pos2,
            a=np.array(list(itertools.product(range(self.grid_size), repeat=2))),
            shape=(1, ),
            p=((1.0 - self.mazes[goal.maze]) * self._quadrants[goal.direction]).reshape(-1),
        )
        goal_coord = self._coord_to_idx(position_goal[0][0], position_goal[0][1])
        position_agent = jax.random.choice(
            key=rng_pos1,
            a=np.array(list(itertools.product(range(self.grid_size), repeat=2))),
            shape=(1, ),
            p=((1.0 - self.mazes[goal.maze]).reshape(-1) * self.valid_goal_dist[goal.maze][goal_coord]),
        )
        positions_distractors = jax.random.choice(
            key=rng_pos3,
            a=np.array(list(itertools.product(range(self.grid_size), repeat=2))),
            shape=(self.num_distractors, ),
            replace=False,
            p=1.0 - self.mazes[goal.maze].reshape(-1),
        )

        positions = jnp.concatenate([position_goal, positions_distractors, position_agent])

        env_state = CompositionalGridState(
            done=False, timestep=0, distractors=distractors, positions=positions, goal=goal
        )
        emission = EnvironmentInteraction(
            observation=self.observe(env_state), reward=0.0, done=False, timestep=0
        )

        return env_state, emission

    def _step(
        self, rng: PRNGKey, env_state, action: Array
    ) -> Tuple[CompositionalGridState, EnvironmentInteraction]:
        pos_agent = env_state.positions[-1, :]

        # Check if agent reached goal (positive reward)
        goal_reached = jnp.logical_and(
            action == (len(MOVES) + env_state.goal.interaction),
            jnp.all(pos_agent == env_state.positions[0, :]),
        )
        reward = 1.0 * goal_reached

        # Move the agent to new position and check if valid
        pos_new = self._delta_position[action] + pos_agent
        pos_invalid = jnp.logical_or(
            jnp.logical_or(jnp.any(pos_new < 0), jnp.any(pos_new >= self.grid_size)),  # in grid?
            self.mazes[env_state.goal.maze][pos_new[0], pos_new[1]],  # in wall?
        )
        pos_new = jnp.where(pos_invalid, pos_agent, pos_new)

        # Update state
        positions = env_state.positions.at[-1].set(pos_new)
        env_state = CompositionalGridState(
            done=goal_reached,
            timestep=env_state.timestep + 1,
            distractors=env_state.distractors,
            positions=positions,
            goal=env_state.goal,
        )

        emission = EnvironmentInteraction(
            observation=self.observe(env_state),
            reward=reward,
            done=env_state.done,
            timestep=env_state.timestep,
        )

        return env_state, emission

    def observe(self, env_state: CompositionalGridState) -> Array:
        """
        Encode the environment state as an asrray of shape (grid_size, grid_size, num_factors * num_objects + 1).
        For each position in the grid, the code word has the following structure:
        [factor_0_feature_0, ..., factor_0_feature_n, ..., factor_n_feature_0, ..., factor_n_feature_n, wall?, agent?]
        """
        objects = jnp.concatenate([jnp.array([env_state.goal.object]), env_state.distractors])
        objects_hot = jax.nn.one_hot(objects, num_classes=self.num_objects)
        pos_objects, pos_agent = env_state.positions[0:-1, :], env_state.positions[-1, :]

        # Build the grid
        grid = jnp.zeros(self.observation_shape)
        grid = grid.at[
            jnp.expand_dims(pos_objects[:, 0], axis=1),
            jnp.expand_dims(pos_objects[:, 1], axis=1),
            :-2,
        ].set(jnp.expand_dims(objects_hot, axis=1))
        grid = grid.at[:, :, -2].set(self.mazes[env_state.goal.maze])  # walls encoded in penultimate channel
        grid = grid.at[pos_agent[0], pos_agent[1], -1].set(1.0)  # agent encoded in last channel

        return grid

    def _features_to_idx(self, features: Array) -> Array:
        """Converts features to a unique feature index"""
        idx = [factor * self.num_objects + feature for factor, feature in enumerate(features)]
        return jnp.array(idx)

    def _coord_to_idx(self, x, y):
        """Converts coordinates to a unique grid index"""
        return x * self.grid_size + y

    def _idx_to_coord(self, idx):
        """Converts a grid index to grid coordinates"""
        return idx // self.grid_size, idx % self.grid_size

    def demonstrate(
        self, rng: PRNGKey, env_state: CompositionalGridState
    ) -> EnvironmentInteraction:
        """Given a state, compute the optimal trajectory to the goal."""
        pos_agent, pos_goal = env_state.positions[-1, :], env_state.positions[0, :]
        idx_agent, idx_goal = self._coord_to_idx(*pos_agent), self._coord_to_idx(*pos_goal)
        optimal_actions = self.optimal_paths[env_state.goal.maze][idx_agent, idx_goal]

        # Fill placeholder actions with correct interaction
        mask_pad = (optimal_actions == -1)
        optimal_actions *= ~mask_pad
        optimal_actions += (len(MOVES) + env_state.goal.interaction) * mask_pad

        def env_step(carry, action):
            rng, env_state = carry
            rng, rng_step = jax.random.split(rng)
            env_state, emission = self.step(rng_step, env_state, action)
            return (rng, env_state), emission

        _, trajectory = jax.lax.scan(env_step, (rng, env_state), optimal_actions)

        # Append initial emission and remove last emission from trajectory
        initial_emission = EnvironmentInteraction(
            observation=self.observe(env_state),
            reward=0.0,
            done=False,
            timestep=0,
        )
        trajectory = jtu.tree_map(
            lambda x, y: jnp.concatenate((jnp.expand_dims(x, axis=0), y)),
            initial_emission, trajectory
        )
        trajectory = jtu.tree_map(lambda x: x[:-1], trajectory)

        return trajectory, optimal_actions

    def _precompute_optimal_paths(self, maze: Array):
        """Precompute the optimal trajectories for all possible states."""
        # Create an array that encodes the graph structure of the grid to compute all shortest paths
        coordinates, no_walls_coords = [], np.argwhere(maze == 0)
        for x, y in no_walls_coords:
            edges = []
            if x > 0 and not maze[x - 1, y]:
                edges.append([x - 1, y])
            if x < self.grid_size - 1 and not maze[x + 1, y]:
                edges.append([x + 1, y])
            if y > 0 and not maze[x, y - 1]:
                edges.append([x, y - 1])
            if y < self.grid_size - 1 and not maze[x, y + 1]:
                edges.append([x, y + 1])

            idx_curr = self._coord_to_idx(x, y)
            coordinates += [(idx_curr, self._coord_to_idx(i, k)) for (i, k) in edges]

        coordinates = np.array(coordinates)
        connectivity = np.zeros((self.grid_size**2, self.grid_size**2))
        connectivity[coordinates[:, 0], coordinates[:, 1]] = 1.0
        shortest_paths, predecessors = shortest_path(connectivity, return_predecessors=True)
        max_num_actions = (self.grid_size**2) - 1

        def get_path(predecessors, start, end):
            """Get the full path from the predecessor matrix."""
            path = [end]
            while path[-1] != start:
                path.append(predecessors[start, path[-1]])
            return path[::-1]

        def path_to_actions(path):
            """Convert path to actions."""
            # Pad with placeholder actions, need to be overwritten with correct interaction in self.demonstrate()
            actions = np.full((max_num_actions), -1)
            for i in range(len(path) - 1):
                x1, y1 = self._idx_to_coord(path[i])
                x2, y2 = self._idx_to_coord(path[i + 1])
                action = np.array([x2 - x1, y2 - y1])
                action = np.where(np.all(self._delta_position == action, axis=1))[0][0]
                actions[i] = action
            return np.array(actions)

        # Precompute optimal paths for all possible positions
        optimal_paths = -1 * np.ones(
            (self.grid_size**2, self.grid_size**2, max_num_actions), dtype=int
        )
        for start in no_walls_coords:
            for goal in no_walls_coords:
                start_idx, goal_idx = self._coord_to_idx(*start), self._coord_to_idx(*goal)
                path = get_path(predecessors, start_idx, goal_idx)
                actions = path_to_actions(path)
                optimal_paths[start_idx, goal_idx, :] = actions

        return jnp.array(optimal_paths), jnp.array(shortest_paths)

    @staticmethod
    def generate_random_maze(
        grid_size: int, complexity: float = 0.75, density: float = 0.75, seed: int = 0
    ):
        """
        Generate a random maze array.
        Walls are encoded as 1 and free space as 0.

        Adapted from https://github.com/zuoxingdong/mazelab/blob/master/mazelab/generators/random_maze.py
        which is based on https://en.wikipedia.org/wiki/Maze_generation_algorithm
        """
        assert grid_size % 2 == 1, "Maze size must be odd"
        grid_size_pad = grid_size + 2
        np_rng = np.random.default_rng(seed)

        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (grid_size_pad + grid_size_pad)))
        density = int(density * ((grid_size_pad // 2) * (grid_size_pad // 2)))

        # Fill borders
        grid = np.zeros((grid_size_pad, grid_size_pad), dtype=bool)
        grid[0, :] = grid[-1, :] = 1
        grid[:, 0] = grid[:, -1] = 1

        # Make aisles
        for _ in range(density):
            x, y = (
                np_rng.integers(0, grid_size_pad // 2 + 1) * 2,
                np_rng.integers(0, grid_size_pad // 2 + 1) * 2,
            )
            grid[y, x] = 1
            for j in range(complexity):
                neighbours = []
                if x > 1:
                    neighbours.append((y, x - 2))
                if x < grid_size_pad - 2:
                    neighbours.append((y, x + 2))
                if y > 1:
                    neighbours.append((y - 2, x))
                if y < grid_size_pad - 2:
                    neighbours.append((y + 2, x))
                if len(neighbours):
                    y_, x_ = neighbours[np_rng.integers(0, len(neighbours))]
                    if grid[y_, x_] == 0:
                        grid[y_, x_] = 1
                        grid[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_

        return grid.astype(int)[1:-1, 1:-1]
