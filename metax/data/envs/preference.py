"""
Copyright (c) Seijin Kobayashi, Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import enum
import itertools
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from chex import Array, PRNGKey
from flax import struct

from .base import Environment, EnvironmentInteraction


class ACTIONS(enum.Enum):
    NOTHING = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


@struct.dataclass
class PreferenceState:
    done: bool
    timestep: int
    positions: Array  # shape: (1(goal) + num_objects + 1(agent), 2)
    features: Array  # shape: (1(goal) + num_objects, num_features)
    available_distractors: Array
    preference: Array


class CompositionalPreference(Environment):
    #     _layout = """\
    # wwwwwwwwwwwww
    # w     w     w
    # w     w     w
    # w           w
    # w     w     w
    # w     w     w
    # ww wwww     w
    # w     www www
    # w     w     w
    # w     w     w
    # w           w
    # w     w     w
    # wwwwwwwwwwwww
    # """
    _layout = """\
wwwwwww
w  w  w
w  w  w
ww   ww
w  w  w
w  w  w
wwwwwww
"""
    _delta_position = jnp.array(
        [
            [0, 0],  # NOTHING
            [-1, 0],  # UP
            [0, 1],  # RIGHT
            [1, 0],  # DOWN
            [0, -1],  # LEFT
        ]
    )

    def __init__(
        self,
        num_preferences: int,  # ~=num_experts
        num_features: int,  # ~=dim layer weight
        num_objects: int,
        num_hot: int,  # ~= num_hot
        continuous_combinations: bool,
        discount: float,
        frac_ood: float,
        timelimit: int,
        task_support: str,
        seed: int,
    ) -> None:
        super().__init__()
        self.num_preferences = num_preferences
        self.num_features = num_features
        self.num_objects = num_objects
        self.num_hot = num_hot
        self.continuous_combinations = continuous_combinations
        self.discount = discount
        self.frac_ood = frac_ood
        self.timelimit = timelimit
        self.task_support = task_support
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)

        # We assume a fixed grid.
        self.grid = jnp.array(
            [list(map(lambda c: 0 if c == " " else 1, line)) for line in self._layout.splitlines()]
        )
        self.free_coord = jnp.array([(x, y) for (x, y) in zip(*np.where(self.grid == 0))])
        grid_idx_to_coord_matrix = jax.nn.one_hot(
            self.free_coord[:, 0] * self.grid.shape[1] + self.free_coord[:, 1],
            self.grid.shape[0] * self.grid.shape[1],
        )
        self.coord_matrix_to_grid_idx = jnp.argmax(grid_idx_to_coord_matrix.T, axis=-1)
        self.grid_idx_to_coord_matrix = jnp.argmax(grid_idx_to_coord_matrix, axis=-1)
        self.num_free_coord = self.free_coord.shape[0]
        self.num_available_distractors_config = 2**self.num_objects
        self.num_states = self.num_free_coord * self.num_available_distractors_config

        self.preference_basis = jax.random.normal(
            self.rng, (self.num_preferences, self.num_features)
        )

        # Generate all possible combinations of 1:num_hot experts (num_experts choose num_hot)
        preference_combin_all = []
        for h in range(1, self.num_hot + 1):
            perms = itertools.combinations(range(self.num_preferences), h)
            preference_idx = np.array(list(perms)).reshape(-1, h)
            preference_combin_all_k_hot = self.k_hot(preference_idx)
            preference_combin_all.append(preference_combin_all_k_hot)

        preference_combin_all = jnp.concatenate(preference_combin_all)

        if self.task_support == "connected" or self.task_support == "disconnected":
            assert self.num_hot == 2
            assert self.num_preferences > 4 and self.num_preferences % 2 == 0
            # connected: 0 1 2 3 4 5 6 7 01 12 23 34 45 56 67 70 02 13 24 35 46 57 60 71
            preference_combin = [self.k_hot(np.arange(self.num_preferences)[:, None])]  # one-hots
            preference_combin.append(self.k_hot(np.stack((  # two-hots 01 12 23 34 45 56 67 70
                np.arange(self.num_preferences),
                (np.arange(self.num_preferences) + 1) % self.num_preferences)).T
            ))
            preference_combin.append(self.k_hot(np.stack((  # two-hots 02 13 24 35 46 57 60 71
                np.arange(self.num_preferences),
                (np.arange(self.num_preferences) + 2) % self.num_preferences)).T
            ))
            preference_combin_connected = np.concatenate(preference_combin)

            @partial(np.vectorize, signature="(n),(m,n)->()")
            def elem_in_array(elem, array):
                return np.any(np.all(elem == array, axis=1))

            mask_connected = elem_in_array(preference_combin_all, preference_combin_connected)

            # disconnected: 1 and 2 hots out of (0,1,2,3) U 1 and 2 hots out of (4,5,6,7)
            mask_1_hot = jnp.sum(preference_combin_all, axis=-1) == 1
            mask_2_hot = jnp.sum(preference_combin_all, axis=-1) == 2
            mask_preference_combin_1 = jnp.all(preference_combin_all[:, :self.num_preferences // 2] == 0, axis=1)
            mask_preference_combin_2 = jnp.all(preference_combin_all[:, self.num_preferences // 2:] == 0, axis=1)

            mask_disconnected = (
                (mask_1_hot & mask_preference_combin_1) | (mask_1_hot & mask_preference_combin_2) | (
                    mask_2_hot & mask_preference_combin_1) | (mask_2_hot & mask_preference_combin_2)
            )

            if self.task_support == "connected":
                mask_in_dist = mask_connected
            elif self.task_support == "disconnected":
                mask_in_dist = mask_disconnected

            mask_out_dist = ~(mask_connected | mask_disconnected)

            self.preference_in_dist = jnp.array(preference_combin_all[mask_in_dist])
            self.preference_out_dist = jnp.array(preference_combin_all[mask_out_dist])

        elif self.task_support == "non_compositional":
            # Non-compositional task support holds-out the last expert in the last layer
            mask_last_expert = preference_combin_all[:, -1] == 1
            self.preference_in_dist = jnp.array(preference_combin_all[~mask_last_expert])
            self.preference_out_dist = jnp.array(preference_combin_all[mask_last_expert])

        elif self.task_support == "random":
            # Randomly split task experts into in and out distribution tasks
            preference_combin_all = jax.random.permutation(self.rng, preference_combin_all)
            self.num_ood = int(len(preference_combin_all) * self.frac_ood)
            self.preference_in_dist = jnp.array(preference_combin_all[: -self.num_ood])
            self.preference_out_dist = jnp.array(preference_combin_all[-self.num_ood:])

        assert len(self.preference_in_dist) > 0
        assert len(self.preference_out_dist) > 0

        self.objects_all = jax.random.permutation(self.rng, np.arange(self.num_features))

    @partial(jnp.vectorize, excluded=(0,), signature="(n)->(m)")
    def k_hot(self, ind):
        """
        Convert a vector of indeces to a k-hot vector.
        Repeating an index does not change the result.
        """
        return (jnp.sum(jax.nn.one_hot(ind, self.num_preferences), axis=0) > 0) * 1.0

    @property
    def num_actions(self) -> int:
        return len(ACTIONS)

    @property
    def observation_shape(self) -> Tuple[int]:
        return (*self.grid.shape, self.num_features + 2)

    def reset_goal(self, rng: PRNGKey, mode: str) -> Array:
        # Copied from hyperteacher
        rng_tasks, rng_weights = jax.random.split(rng)
        if mode in ["test", "train", "ood"]:
            task_experts = self.preference_out_dist if mode == "ood" else self.preference_in_dist
            task_ids = jax.random.choice(rng_tasks, len(task_experts), shape=())
            embeddings = task_experts[task_ids]

            if mode == "ood":
                task_ids += len(self.preference_in_dist)
        elif "ood_" in mode:
            hotness = int(mode.split("_")[1])
            if hotness <= self.num_hot:
                # Filter the existing task_experts_out_dist for the given hotness
                task_ids = jax.random.choice(
                    key=rng_tasks,
                    a=len(self.preference_out_dist),
                    p=1.0 * jnp.all(
                        jnp.sum(self.preference_out_dist, axis=-1) == hotness, axis=-1
                    ),
                    shape=(),
                )
                embeddings = self.preference_out_dist[task_ids]
            elif hotness <= self.num_preferences:
                # Randomly sample task_experts - everything is ood here
                expert_indeces = jax.random.choice(rng_tasks, self.num_preferences, replace=False, shape=(hotness, ))
                embeddings = self.k_hot(expert_indeces)
                task_ids = -1 * jnp.ones(())  # No unique task IDs available here
            else:
                raise ValueError(f"Invalid hotness {hotness}")

        if self.continuous_combinations:
            # Sample weights uniformly from simplex (see Willms, 2021)
            weights = jax.random.exponential(rng_weights, shape=embeddings.shape)
            weights = weights * embeddings
            weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1)

            # Shift nonzero embeddings to the range [0.5, 1.0] to prevent adding further sparsity
            embeddings = (0.5 * weights + 0.5) * embeddings

        return embeddings, {"task_id": task_ids, "embedding": embeddings[None, :]}

    @partial(jax.jit, static_argnums=(0))
    def reset(
        self, rng: PRNGKey, goal: Array = None
    ) -> Tuple[PreferenceState, EnvironmentInteraction]:
        """Resets the environment to a random, initial state"""
        rng_preference, rng_distractor, rng_pos = jax.random.split(rng, 3)

        if goal is None:
            # Sample a preference from train distribution if None specified
            goal, _ = self.reset_goal(rng_preference, mode="train")

        preference = goal

        # Sample distractors
        distractors = jax.random.choice(
            key=rng_distractor,
            a=self.objects_all,
            shape=(self.num_objects,),
            replace=True,
        )

        positions = jax.random.choice(
            rng_pos, self.free_coord, shape=(self.num_objects + 1,), replace=False
        )

        env_state = PreferenceState(
            done=False,
            timestep=0,
            positions=positions,
            features=distractors,
            available_distractors=jnp.ones((self.num_objects,)),
            preference=preference,
        )

        emission = EnvironmentInteraction(
            observation=self.observe(env_state), reward=0.0, done=False, timestep=0
        )
        return env_state, emission

    @partial(jax.jit, static_argnums=(0))
    def _step(
        self, rng: PRNGKey, env_state, action: Array
    ) -> Tuple[PreferenceState, EnvironmentInteraction]:
        pos_agent = env_state.positions[-1][0], env_state.positions[-1][1]
        distractors_pos = env_state.positions[:-1]
        features = env_state.features
        available_distractors = env_state.available_distractors

        preference = env_state.preference

        next_pos_agent, next_available_distractors, reward = self._move(
            pos_agent, features, available_distractors, distractors_pos, preference, action
        )
        next_timestep = env_state.timestep + 1
        # Update state
        env_state = PreferenceState(
            # If NOTHING is performed, the environment immediately terminates.
            done=jnp.logical_or(next_timestep > self.timelimit, action == ACTIONS.NOTHING.value),
            timestep=next_timestep,
            positions=env_state.positions.at[-1].set(next_pos_agent),
            features=env_state.features,
            available_distractors=next_available_distractors,
            preference=env_state.preference,
        )

        emission = EnvironmentInteraction(
            observation=self.observe(env_state),
            reward=reward,
            done=env_state.done,
            timestep=env_state.timestep,
        )

        return env_state, emission

    def observe(self, env_state: PreferenceState) -> Array:
        distractor_idx = env_state.features
        pos_objects, pos_agent = env_state.positions[0:-1, :], env_state.positions[-1, :]

        # Build the grid
        grid = jnp.zeros((*self.grid.shape, self.num_features + 2))

        grid = grid.at[
            (pos_objects[:, 0]),
            (pos_objects[:, 1]),
            distractor_idx,
        ].set(env_state.available_distractors)
        grid = grid.at[pos_agent[0], pos_agent[1], -2].set(
            1.0
        )  # agent encoded in penultimate channel
        grid = grid.at[:, :, -1].set(self.grid)  # walls encoded in last channel

        return grid

    def _idx_to_state(self, idx):
        grid_idx = idx // self.num_available_distractors_config
        distractor_config_idx = idx % self.num_available_distractors_config
        coord_packed = self.grid_idx_to_coord_matrix[grid_idx]
        coord = coord_packed // self.grid.shape[1], coord_packed % self.grid.shape[1]
        return coord, (((distractor_config_idx & (1 << np.arange(self.num_objects)))) > 0).astype(
            int
        )

    def _state_to_idx(self, coord, available_distractors):
        coord_packed = coord[0] * self.grid.shape[1] + coord[1]
        grid_idx = self.coord_matrix_to_grid_idx[coord_packed]
        distractor_config_idx = available_distractors @ (2 ** jnp.arange(self.num_objects))
        return (grid_idx * self.num_available_distractors_config + distractor_config_idx).astype(
            int
        )

    def _move(
        self, pos_agent, features, available_distractors, distractors_pos, preference, action
    ):
        delta_position = self._delta_position[action]
        next_position = pos_agent[0] + delta_position[0], pos_agent[1] + delta_position[1]
        # TODO(@simon): Remove boundary walls to save some input dim and check if within grid size bounds instead
        next_pos_grid = (
            jax.nn.one_hot(next_position[0], self.grid.shape[0])[..., None]
            * jax.nn.one_hot(next_position[1], self.grid.shape[1])[..., None].T
        )
        hit_wall = (self.grid * next_pos_grid).sum()
        next_position = jax.lax.cond(hit_wall, lambda _: pos_agent, lambda _: next_position, None)
        picked_distractor = (next_position[0] == distractors_pos[:, 0]) * (
            next_position[1] == distractors_pos[:, 1]
        )

        return (
            next_position,
            available_distractors * (1 - picked_distractor),
            (
                (picked_distractor * available_distractors)
                @ jax.nn.one_hot(features, self.num_features)
                @ self.preference_basis.T
                @ preference
            ),
        )

    @partial(jax.jit, static_argnums=(0))
    def demonstrate(self, rng, env_state):
        """Given a state, compute the optimal trajectory to the goal."""
        action_value_init = jnp.zeros((self.num_states, self.num_actions))

        def next_idx_and_reward(idx, action):
            coord, available_distractors = self._idx_to_state(idx)
            next_coord, next_available_feature, reward = self._move(
                coord,
                env_state.features,
                available_distractors,
                env_state.positions[:-1],
                env_state.preference,
                action,
            )
            next_idx = self._state_to_idx(next_coord, next_available_feature)
            # Return the maximum value
            return next_idx, reward

        transition_map, reward_map = jax.vmap(
            jax.vmap(next_idx_and_reward, in_axes=(None, 0)), in_axes=(0, None)
        )(jnp.arange(self.num_states), jnp.arange(self.num_actions))

        def bellman_backup(action_value, t):
            def next_value(idx, action):
                next_idx = transition_map[idx, action]
                reward = reward_map[idx, action]
                # Return the maximum value
                return self.discount * action_value[next_idx].max() + reward

            next_action_value = jax.vmap(
                jax.vmap(next_value, in_axes=(None, 0)), in_axes=(0, None)
            )(jnp.arange(self.num_states), jnp.arange(self.num_actions))
            return next_action_value, None

        action_value, _ = jax.lax.scan(
            bellman_backup, action_value_init, jnp.arange(self.timelimit)
        )

        def env_step(carry, t):
            rng, env_state = carry
            rng, rng_step = jax.random.split(rng)
            pos_agent = env_state.positions[-1]
            idx = self._state_to_idx(pos_agent, env_state.available_distractors)
            action = jnp.argmax(action_value[idx])
            env_state, emission = self.step(rng_step, env_state, action)
            return (rng, env_state), (emission, action_value[idx])

        (_, _), (trajectory, action_values) = jax.lax.scan(
            env_step, (rng, env_state), jnp.arange(self.timelimit)
        )

        # Append initial emission and remove last emission from trajectory
        initial_emission = EnvironmentInteraction(
            observation=self.observe(env_state),
            reward=0.0,
            done=False,
            timestep=0,
        )
        trajectory = jtu.tree_map(
            lambda x, y: jnp.concatenate((jnp.expand_dims(x, axis=0), y)),
            initial_emission,
            trajectory,
        )
        trajectory = jtu.tree_map(lambda x: x[:-1], trajectory)

        return trajectory, action_values
