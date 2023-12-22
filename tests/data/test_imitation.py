"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import unittest

import jax
import jax.numpy as jnp

from metax.data.envs.grid import CompositionalGrid
from metax.data.imitation import (ImitationMetaDataloader,
                                  create_imitation_metaloader)
from metax.utils import tree_length


class ImitationTestCase(unittest.TestCase):
    rng = jax.random.PRNGKey(0)

    def test_ImitationMetaDataloader(self):
        env = CompositionalGrid(
            grid_size := 7,
            num_interactions := 3,
            num_mazes := 2,
            num_objects := 5,
            num_distractors := 2,
            frac_ood := 0.2,
            task_support := "random",
            seed := 2022,
        )

        loader = ImitationMetaDataloader(
            env,
            num_tasks := 2048,
            shots_train := 1,
            shots_test := 1,
            meta_batch_size := 128,
            mode="train",
            train_test_split=False,
            rng=self.rng
        )
        assert len(loader) == num_tasks / meta_batch_size

        for batch in loader:
            assert jnp.all(batch.train.task_id == batch.test.task_id)
            # assert jnp.all(batch.train.x != batch.test.x)
            # assert jnp.any(batch.train.y != batch.test.y)
            assert len(batch.train.x) == meta_batch_size

    def test_create_imitation_metaloader(self):
        trainloader, testloader, validloader, oodloader, _ = create_imitation_metaloader(
            name := "compositional_grid",
            meta_batch_size := 128,
            shots_train := 2,
            shots_test := 2,
            train_test_split := False,
            num_tasks_train  := 4096,
            num_tasks_test := 1024,
            num_tasks_valid := 1024,
            num_tasks_ood := 1024,
            seed  := 2022,
            grid_size=7,
            num_interactions=3,
            num_mazes=2,
            num_objects=5,
            num_distractors=2,
            frac_ood=0.2,
            task_support="random",
        )

        assert trainloader.sample_input.shape == (1, 7, 7, 5 + 2)

        goals_ood = []
        for batch in oodloader:
            goals_ood.append(jnp.unique(batch.test.task_id[:, 0], axis=0))
        goals_ood = jnp.concatenate(goals_ood)

        goals_train = []
        for batch in trainloader:
            goals_train.append(jnp.unique(batch.test.task_id[:, 0], axis=0))
        goals_train = jnp.unique(jnp.concatenate(goals_train), axis=0)

        assert len(goals_train) + len(goals_ood) == 3 * 2 * 5 * 4

        # Check that ood tasks are disjoint from train tasks
        for g_ood in goals_ood:
            assert not jnp.any(g_ood == goals_train)

        for batch_test, batch_valid in zip(testloader, validloader):
            assert tree_length(batch_test) == num_tasks_test
            not jnp.all(batch_test.train.x == batch_valid.train.x)
