"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from chex import PRNGKey

from metax.data.envs.base import Environment
from metax.data.envs.grid import CompositionalGrid
from metax.data.envs.preference import CompositionalPreference

from .base import Dataloader, MetaDataset, MultitaskDataset


class ImitationMetaDataloader(Dataloader):
    def __init__(
        self,
        env: Environment,
        num_tasks: int,
        shots_train: int,
        shots_test: int,
        meta_batch_size: int,
        mode: str,
        train_test_split: bool,
        rng: PRNGKey,
    ):
        super().__init__(input_shape=env.observation_shape, output_dim=env.num_actions)
        self.env = env
        self.num_tasks = num_tasks
        self.shots_train = shots_train
        self.shots_test = shots_test
        self.meta_batch_size = meta_batch_size
        self.mode = mode
        self.train_test_split = train_test_split
        self.fixed_rng = rng

        assert num_tasks % meta_batch_size == 0, "num_tasks must be divisible by meta_batch_size"
        self.num_steps = num_tasks // meta_batch_size

    @property
    def sample_input(self):
        return jnp.zeros((1,) + self.env.observation_shape)

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        for rng in jax.random.split(self.fixed_rng, self.num_steps):
            # Sample batch and wrap as MetaDataset
            rngs_batch = jax.random.split(rng, self.meta_batch_size)
            yield self.sample_metatask(rngs_batch)

    @partial(jax.jit, static_argnames="self")
    @partial(jax.vmap, in_axes=(None, 0))
    def sample_metatask(self, rng: PRNGKey) -> MetaDataset:
        rng_goal, rng_task = jax.random.split(rng, 2)
        goal, info = self.env.reset_goal(rng_goal, mode=self.mode)

        @jax.vmap
        def sample_task(rng):
            rng_reset, rng_demo = jax.random.split(rng, 2)
            env_state, _ = self.env.reset(rng_reset, goal=goal)
            trajectory, actions = self.env.demonstrate(rng_demo, env_state)

            return MultitaskDataset(
                x=trajectory.observation,
                y=actions,
                task_id=jnp.full(actions.shape[:1], info["task_id"]),
                info={
                    "mask": ~trajectory.done,
                    "embeddings": jnp.repeat(info["embedding"][None, :], actions.shape[0], axis=0),
                },
            )

        rngs_task = jax.random.split(rng_task, self.shots_train + self.shots_test)
        train_and_test_task = sample_task(rngs_task)

        if self.train_test_split:
            # Split into train and test set
            return MetaDataset(
                train=jtu.tree_map(
                    lambda x: x[:self.shots_train].reshape(-1, *x.shape[2:]), train_and_test_task
                ),
                test=jtu.tree_map(
                    lambda x: x[self.shots_train:].reshape(-1, *x.shape[2:]), train_and_test_task
                ),
            )
        else:
            # No train_test split means, meta.train == meta.test set
            return MetaDataset(
                train=jtu.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), train_and_test_task),
                test=jtu.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), train_and_test_task),
            )


def create_imitation_metaloader(
    name,
    meta_batch_size,
    shots_train,
    shots_test,
    train_test_split,
    num_tasks_train,
    num_tasks_test,
    num_tasks_valid,
    num_tasks_ood: Optional[int] = None,
    seed=None,
    **kwargs,
):
    ood_sets_hot = None
    if name == "compositional_grid":
        env = CompositionalGrid(
            grid_size=kwargs["grid_size"],
            num_interactions=kwargs["num_interactions"],
            num_mazes=kwargs["num_mazes"],
            num_objects=kwargs["num_objects"],
            num_distractors=kwargs["num_distractors"],
            frac_ood=kwargs["frac_ood"],
            task_support=kwargs["task_support"],
            seed=seed,
        )
    elif name == "compositional_preference":
        # Return the various OOD tasks for the compositional preference env.
        ood_sets_hot = jnp.arange(kwargs["num_hot"] + 1, kwargs["num_preferences"] + 1)
        env = CompositionalPreference(
            num_preferences=kwargs["num_preferences"],
            num_features=kwargs["num_features"],
            num_objects=kwargs["num_objects"],
            num_hot=kwargs["num_hot"],
            continuous_combinations=kwargs["continuous_combinations"],
            discount=kwargs["discount"],
            frac_ood=kwargs["frac_ood"],
            timelimit=kwargs["timelimit"],
            task_support=kwargs["task_support"],
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown environment {name}")

    rng_train, rng_test, rng_valid, rng_ood = jax.random.split(jax.random.PRNGKey(seed), 4)

    metatrainloader = ImitationMetaDataloader(
        env=env,
        num_tasks=num_tasks_train,
        shots_train=shots_train,
        shots_test=shots_test if train_test_split else 0,
        meta_batch_size=meta_batch_size,
        mode="train",
        train_test_split=train_test_split,
        rng=rng_train
    )

    metatestloader = ImitationMetaDataloader(
        env=env,
        num_tasks=num_tasks_test,
        shots_train=shots_train,
        shots_test=1,  # HACK: we need shots_support_train, shots_query_train, shots_support_test and shots_query_test
        meta_batch_size=num_tasks_test,
        mode="test",
        train_test_split=True,
        rng=rng_test
    )
    metavalidloader = ImitationMetaDataloader(
        env=env,
        num_tasks=num_tasks_valid,
        shots_train=shots_train,
        shots_test=1,  # HACK: we need shots_support_train, shots_query_train, shots_support_test and shots_query_test
        meta_batch_size=num_tasks_valid,
        mode="test",
        train_test_split=True,
        rng=rng_valid
    )
    metaoodloader = ImitationMetaDataloader(
        env=env,
        num_tasks=num_tasks_ood,
        shots_train=shots_train,
        shots_test=1,  # HACK: we need shots_support_train, shots_query_train, shots_support_test and shots_query_test
        meta_batch_size=num_tasks_ood,
        mode="ood",
        train_test_split=True,
        rng=rng_ood
    )

    if ood_sets_hot is not None:
        metaauxloaders = {
            "ood_{}".format(ood_set): ImitationMetaDataloader(
                env=env,
                num_tasks=num_tasks_ood,
                shots_train=shots_train,
                shots_test=1,  # HACK: we need shots_support_train, shots_query_train, shots_support_test and shots_query_test
                meta_batch_size=num_tasks_ood,
                mode="ood_{}".format(ood_set),
                train_test_split=True,
                rng=r,
            )
            for ood_set, r in zip(ood_sets_hot, jax.random.split(rng_ood, len(ood_sets_hot)))
        }
    else:
        metaauxloaders = None

    return metatrainloader, metatestloader, metavalidloader, metaoodloader, metaauxloaders
