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
import numpy as np
import pandas as pd
import plotly.express as px

from metax.data.dataset.teacher import HyperTeacher


class HyperTeacherTestCase(unittest.TestCase):
    def test_regression_visual(self):
        data_generator = HyperTeacher(
            input_dim=1,
            output_dim=1,
            hidden_dim=16,
            num_hidden=3,
            num_experts=8,
            num_hot=1,
            frac_ood=0.25,
            scale=3.0,
            classification=False,
            normalize_classifier=False,
            targets_temperature=0.1,
            continuous_combinations=True,
            chunking=True,
            task_support="random",
            seed=0
        )

        df_list = []
        for task in range(25):
            rng = jax.random.PRNGKey(task)
            data = data_generator.sample(rng, num_tasks=5, num_samples=1000, mode="train")
            df_list.append(pd.DataFrame({
                "x": data.x[0, :, 0],
                "y": data.y[0, :, 0],
                "task": np.ones_like(data.y[0, :, 0]) * task,
            }))

        df = pd.concat(df_list).sort_values(by=["task"])
        fig = px.scatter(df, x="x", y="y", facet_col="task", facet_col_wrap=5)
        fig.show()

    def test_classification_visual(self):
        data_generator = HyperTeacher(
            input_dim=2,
            output_dim=8,
            hidden_dim=32,
            num_hidden=3,
            num_experts=8,
            num_hot=3,
            frac_ood=0.25,
            scale=1.0,
            classification=True,
            normalize_classifier=True,
            targets_temperature=0.1,
            continuous_combinations=True,
            chunking=False,
            task_support="random",
            seed=0
        )

        df_list = []
        for task in range(12):
            rng = jax.random.PRNGKey(task)
            data = data_generator.sample(rng, num_tasks=5, num_samples=512, mode="train")
            df_list.append(pd.DataFrame({
                "x": data.x[0, :, 0],
                "y": data.x[0, :, 1],
                "class": data.info["hard_targets"][0, :],
                "task": data.task_id[0, :],
            }))

        df = pd.concat(df_list).sort_values(by=["task", "class"])
        df["class"] = df["class"].astype(str)
        fig = px.scatter(df, x="x", y="y", color="class", facet_col="task", facet_col_wrap=5)
        fig.show()

    def test_summary_stats(self):
        # with jax.disable_jit():
        data_generator = HyperTeacher(
            input_dim=2,
            output_dim=8,
            hidden_dim=32,
            num_hidden=3,
            num_experts=8,
            num_hot=3,
            frac_ood=0.25,
            scale=1.0,
            classification=True,
            normalize_classifier=True,
            targets_temperature=0.1,
            continuous_combinations=True,
            chunking=True,
            task_support="random",
            seed=0
        )
        data = data_generator.sample(jax.random.PRNGKey(0), num_tasks=1000, num_samples=10, mode="train")
        mean, stdv = np.mean(data.y), np.std(data.y)
        min, max = np.min(data.y), np.max(data.y)
        print("mean: {:2f}±{:2f} \t min: {:2f} \t max: {:2f} \t".format(mean, stdv, min, max))

    def test_experts(self):
        task_support = "connected"
        data_generator = HyperTeacher(
            input_dim=2,
            output_dim=8,
            hidden_dim=8,
            num_hidden=2,
            num_experts=8,
            num_hot=2,
            frac_ood=0.25,
            scale=1.0,
            classification=True,
            normalize_classifier=True,
            targets_temperature=0.1,
            continuous_combinations=False,
            chunking=True,
            task_support=task_support,
            seed=0
        )

        # Check that all expected expert combinations are present
        embeddings, _ = data_generator._sample_embeddings(jax.random.PRNGKey(0), 10000, mode="test")
        assert len(jnp.unique(embeddings, axis=0)) == len(data_generator.task_experts_in_dist)
        assert len(embeddings) == 10000

        embeddings, _ = data_generator._sample_embeddings(jax.random.PRNGKey(0), 10000, mode="ood")
        assert len(jnp.unique(embeddings, axis=0)) == len(data_generator.task_experts_out_dist)
        assert len(embeddings) == 10000

        embeddings, _ = data_generator._sample_embeddings(jax.random.PRNGKey(0), 10000, mode="train")
        assert len(jnp.unique(embeddings, axis=0)) == len(data_generator.task_experts_in_dist)
        assert len(embeddings) == 10000

        if task_support in ["dense_vs_rest", "non_compositional", "random"]:
            embeddings, _ = data_generator._sample_embeddings(jax.random.PRNGKey(0), 10000, mode="ood_1")
            assert jnp.all(jnp.sum(embeddings, axis=-1) == 1)
            assert len(embeddings) == 10000

        embeddings, _ = data_generator._sample_embeddings(jax.random.PRNGKey(0), 10000, mode="ood_4")
        assert jnp.all(jnp.sum(embeddings, axis=-1) == 4)
        assert len(embeddings) == 10000

        embeddings, _ = data_generator._sample_embeddings(jax.random.PRNGKey(0), 10000, mode="ood_8")
        assert jnp.all(jnp.sum(embeddings, axis=-1) == 8)
        assert len(embeddings) == 10000

    def test_uniform_simplex(self):
        rng = jax.random.PRNGKey(0)
        embeddings = jnp.ones(shape=(num_tasks := 2048, num_experts := 2))
        weights = jax.random.exponential(rng, shape=(num_tasks, num_experts))
        weights = weights / (1 + jnp.sum(weights, axis=-1, keepdims=True))
        embeddings = weights

        fig = px.scatter_3d(x=embeddings[:, 0], y=embeddings[:, 1], z=embeddings[:, 2], opacity=0.5)
        fig.write_html("simplex.html")
        # fig.show()
