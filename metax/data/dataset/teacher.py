"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import itertools
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from metax.data.base import MultitaskDataset
from metax.data.dataset.base import DatasetGenerator
from metax.models.mlp import MultilayerPerceptron
from metax.utils import PytreeReshaper


class HyperTeacher(DatasetGenerator):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden: int,
        num_experts: int,
        num_hot: int,
        frac_ood: float,
        scale: float,
        classification: bool,
        normalize_classifier: bool,
        targets_temperature: float,
        continuous_combinations: bool,
        chunking: bool,
        task_support: str,
        seed: int,
    ) -> None:
        """
        Args:
            num_hidden: Number of hidden layers of teacher. If num_hidden=1 the input layer will be
                        hnet generated, otherwise the input layer is fixed and all hidden layers
                        afterwards are hnet generated.
        """
        super().__init__(input_shape=(input_dim, ), output_dim=output_dim)
        assert num_experts >= num_hot, "num_experts must be >= num_hot"
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.num_experts = num_experts
        self.num_hot = num_hot
        self.frac_ood = frac_ood
        self.scale = scale
        self.classification = classification
        self.normalize_classifier = normalize_classifier
        self.targets_temperature = targets_temperature
        self.continuous_combinations = continuous_combinations
        self.chunking = chunking
        self.task_support = task_support
        self.fixed_rng = jax.random.PRNGKey(seed)

        # If num_hidden=1, input is hnet generated
        self.num_hidden_hnet = self.num_hidden - (self.num_hidden > 1)

        # Generate all possible combinations of 1:num_hot experts (num_experts choose num_hot)
        experts = []
        for h in range(1, self.num_hot + 1):
            perms = itertools.combinations(range(self.num_experts), h)
            experts_idx = np.array(list(perms)).reshape(-1, h)
            experts_k_hot = self.k_hot(experts_idx)
            experts.append(experts_k_hot)

        # Generate all possible, layer-wise combinations of experts
        experts = np.concatenate(experts)
        combins = itertools.product(experts, repeat=self.num_hidden_hnet)
        task_experts_all = np.stack([np.stack(args, axis=0) for args in combins], axis=0)

        if self.task_support == "connected" or self.task_support == "disconnected":
            assert self.num_hot == 2
            assert self.num_experts > 4 and self.num_experts % 2 == 0
            # connected: 0 1 2 3 4 5 6 7 01 12 23 34 45 56 67 70 02 13 24 35 46 57 60 71
            experts = [self.k_hot(np.arange(self.num_experts)[:, None])]  # one-hots
            experts.append(self.k_hot(np.stack((  # two-hots 01 12 23 34 45 56 67 70
                np.arange(self.num_experts),
                (np.arange(self.num_experts) + 1) % self.num_experts)).T
            ))
            experts.append(self.k_hot(np.stack((  # two-hots 02 13 24 35 46 57 60 71
                np.arange(self.num_experts),
                (np.arange(self.num_experts) + 2) % self.num_experts)).T
            ))
            combins = itertools.product(np.concatenate(experts), repeat=self.num_hidden_hnet)
            experts_connected = jnp.stack([np.stack(args, axis=0) for args in combins], axis=0)

            @partial(np.vectorize, signature="(n,k),(m,n,k)->()")
            def elem_in_array(elem, array):
                return np.any(np.all(elem == array, axis=(1, 2)))

            mask_connected = elem_in_array(task_experts_all, experts_connected)

            # disconnected: 1 and 2 hots out of (0,1,2,3) U 1 and 2 hots out of (4,5,6,7)
            mask_1_hot = jnp.all(jnp.sum(task_experts_all, axis=-1) == 1, axis=-1)
            mask_2_hot = jnp.all(jnp.sum(task_experts_all, axis=-1) == 2, axis=-1)
            mask_experts_1 = jnp.all(task_experts_all[:, :, :self.num_experts // 2] == 0, axis=(1, 2))
            mask_experts_2 = jnp.all(task_experts_all[:, :, self.num_experts // 2:] == 0, axis=(1, 2))

            mask_disconnected = (
                (mask_1_hot & mask_experts_1) | (mask_1_hot & mask_experts_2) | (
                    mask_2_hot & mask_experts_1) | (mask_2_hot & mask_experts_2)
            )

            if self.task_support == "connected":
                mask_in_dist = mask_connected
            elif self.task_support == "disconnected":
                mask_in_dist = mask_disconnected

            mask_out_dist = ~(mask_connected | mask_disconnected)

            self.task_experts_in_dist = jnp.array(task_experts_all[mask_in_dist])
            self.task_experts_out_dist = jnp.array(task_experts_all[mask_out_dist])

        elif self.task_support == "dense":
            # Only use dense embeddings for in distribution tasks
            assert num_hot == num_experts
            mask = jnp.sum(task_experts_all, axis=(-2, -1)) == self.num_hot * self.num_hidden_hnet
            self.task_experts_in_dist = jnp.array(task_experts_all[mask])
            self.task_experts_out_dist = jnp.array(task_experts_all[~mask])

        elif self.task_support == "diagonal":
            # Curriculum with each layer of the form (0,1,2,...), (1,2,3,...), (2,3,4,...), ...
            # Sliding-window of expert combinations
            experts_seq = self.k_hot((
                np.arange(self.num_hot)[None, :] + np.arange(self.num_experts)[:, None]
            ) % self.num_experts)

            # One-hot combinations
            experts_single = self.k_hot(np.arange(self.num_experts)[:, None])

            # Concatenate and repeat over layers
            experts = np.concatenate((experts_seq, experts_single))
            combins = itertools.product(experts, repeat=self.num_hidden_hnet)
            self.task_experts_in_dist = jnp.stack([np.stack(args, axis=0) for args in combins], axis=0)

            # Out of distribution tasks are all combinations not in the curriculum
            @partial(np.vectorize, signature="(n,k),(m,n,k)->()")
            def elem_in_array(elem, array):
                return np.any(np.all(elem == array, axis=(1, 2)))

            self.task_experts_out_dist = jnp.array(
                task_experts_all[~elem_in_array(task_experts_all, self.task_experts_in_dist)]
            )
        elif self.task_support == "non_compositional":
            # Non-compositional task support holds-out the last expert in the last layer
            mask_last_expert = task_experts_all[:, -1, -1] == 1
            self.task_experts_in_dist = jnp.array(task_experts_all[~mask_last_expert])
            self.task_experts_out_dist = jnp.array(task_experts_all[mask_last_expert])

        elif self.task_support == "random":
            # Randomly split task experts into in and out distribution tasks
            task_experts_all = jax.random.permutation(self.fixed_rng, task_experts_all)
            self.num_ood = int(len(task_experts_all) * self.frac_ood)
            self.task_experts_in_dist = jnp.array(task_experts_all[:-self.num_ood])
            self.task_experts_out_dist = jnp.array(task_experts_all[-self.num_ood:])

        else:
            raise ValueError(f"Invalid task support {self.task_support}")

        assert len(self.task_experts_in_dist) > 0, "Make sure there are indistribution tasks left"
        assert len(self.task_experts_out_dist) > 0, "Make sure there are ood tasks left"

        if self.num_hidden == 1:
            # In the case of a single hidden layer, the input layer will be hnet generated
            names_layers = (
                *["hidden_{}".format(i) for i in range(self.num_hidden)],
                "output",
            )
        else:
            # In the case of more than one hidden layer, use a fixed input layer
            # and hnet generate the hidden layers
            names_layers = (
                "input",
                *["hidden_{}".format(i) for i in range(self.num_hidden - 1)],
                "output",
            )

        # Define the target network as a multilayer perceptron
        @hk.without_apply_rng
        @hk.transform
        def target_network(inputs, is_training):
            return MultilayerPerceptron(
                output_sizes=self.num_hidden * (self.hidden_dim,) + (output_dim,),
                activation=jax.nn.relu,
                reparametrized_linear=True,
                with_bias=not self.classification,
                b_init=None
                if self.classification
                else hk.initializers.RandomUniform(minval=-1, maxval=1),
                w_init=hk.initializers.TruncatedNormal(stddev=self.scale),
                names_layers=names_layers,
            )(inputs, is_training=is_training)

        self.target_network = target_network

        # Infer shapes of target network to define the weight generator
        # Input/output params are fixed, hidden params generated by the weight generator
        target_params_shape = jax.eval_shape(
            partial(target_network.init, is_training=True),
            jax.random.PRNGKey(0),
            jnp.empty((1, input_dim)),
        )
        target_params_shape = jtu.tree_map(jnp.shape, target_params_shape)
        hidden_params_shape = {k: v for k, v in target_params_shape.items() if "hidden" in k}
        num_elements_per_layer = PytreeReshaper(
            {k: v for k, v in target_params_shape.items() if "hidden_0" in k}
        ).num_elements

        # Define linear weight generator to generate hidden params
        @hk.without_apply_rng
        @hk.transform
        def weight_generator(embedding):
            if self.chunking:
                # Chunking reuses experts across layers
                linear_bank = hk.Linear(
                    num_elements_per_layer,
                    with_bias=False,
                    w_init=hk.initializers.TruncatedNormal(stddev=self.scale),
                )
                params_hidden_flat = hk.vmap(linear_bank, split_rng=False)(embedding)
            else:
                # No chunking uses different experts for each layer
                linear_bank = hk.Linear(
                    num_elements_per_layer * self.num_hidden_hnet,
                    with_bias=False,
                    w_init=hk.initializers.TruncatedNormal(stddev=self.scale),
                )
                params_hidden_flat = linear_bank(
                    embedding.reshape(self.num_hidden_hnet * self.num_experts)
                )

            return PytreeReshaper(hidden_params_shape)(params_hidden_flat.reshape(-1))

        self.weight_generator = weight_generator

        # Use fixed seed to generate the teacher hnet and fixed in/out params
        fixed_rng_in_out, fixed_rng_teacher = jax.random.split(self.fixed_rng)
        params_in_out = self.target_network.init(
            fixed_rng_in_out, jnp.empty((1, self.input_dim)), is_training=True
        )
        self.params_in_out = {k: v for k, v in params_in_out.items() if "hidden" not in k}
        self.params_teacher = self.weight_generator.init(
            fixed_rng_teacher,
            jnp.empty((self.num_hidden_hnet, self.num_experts))
        )

    @partial(jnp.vectorize, excluded=(0,), signature="(n)->(m)")
    def k_hot(self, ind):
        """
        Convert a vector of indeces to a k-hot vector.
        Repeating an index does not change the result.
        """
        return (jnp.sum(jax.nn.one_hot(ind, self.num_experts), axis=0) > 0) * 1.0

    def teacher(self, embedding, inputs):
        params_hidden = self.weight_generator.apply(self.params_teacher, embedding)
        params_target = {**self.params_in_out, **params_hidden}

        return self.target_network.apply(params_target, inputs, is_training=True)

    @partial(jax.jit, static_argnames=("self", "num_tasks", "num_samples", "mode"))
    def sample(self, rng, num_tasks, num_samples, mode):
        assert mode in ["test", "train"] or "ood" in mode
        info = dict()

        rng_embed, rng_data, rng_targets, rng_teacher = jax.random.split(rng, 4)

        # Sample task embeddings
        embeddings, task_ids = self._sample_embeddings(rng_embed, num_tasks, mode)
        task_ids = jnp.repeat(task_ids[:, None], num_samples, axis=1)  # Consistent leading dims
        info["embeddings"] = jnp.repeat(embeddings[:, None], num_samples, axis=1)

        # Sample inputs and compute targets for each task
        inputs = jax.random.uniform(
            rng_data, minval=-1.0, maxval=1.0, shape=(num_tasks, num_samples, self.input_dim)
        )
        targets = jax.vmap(self.teacher)(embeddings, inputs)

        # Create classification targets by sampling from a softmax
        if self.classification:
            if self.normalize_classifier:
                # Normalise each output unit to zero-mean unit-variance to reduce class imbalance
                input_distribution = jax.random.uniform(
                    self.fixed_rng, minval=-1.0, maxval=1.0, shape=(1000, self.input_dim)
                )
                target_distribution = jax.vmap(self.teacher, in_axes=(0, None))(
                    embeddings, input_distribution
                )
                targets_mean = jnp.mean(target_distribution, axis=1, keepdims=True)
                targets_std = jnp.std(target_distribution, axis=1, keepdims=True)
                targets = (targets - targets_mean) / targets_std

            # logits = jax.nn.log_softmax(targets / self.targets_temperature)
            # targets = jax.random.categorical(rng_targets, logits)  # sample from softmax
            targets = jax.nn.softmax(targets / self.targets_temperature)  # soft targets
            info["hard_targets"] = jnp.argmax(targets, axis=-1)

        return MultitaskDataset(x=inputs, y=targets, task_id=task_ids, info=info)

    def _sample_embeddings(self, rng, num_tasks, mode):
        """
        Sample expert panels defining each task.
        """
        # Redefine ID as unique index computed from experts
        rng_tasks, rng_weights = jax.random.split(rng)
        if mode in ["test", "train", "ood"]:
            task_experts = self.task_experts_out_dist if mode == "ood" else self.task_experts_in_dist
            task_ids = jax.random.choice(rng_tasks, len(task_experts), shape=(num_tasks,))
            embeddings = task_experts[task_ids]

            if mode == "ood":
                task_ids += len(self.task_experts_in_dist)
        elif "ood_" in mode:
            hotness = int(mode.split("_")[1])
            if hotness <= self.num_hot:
                # NOTE: In case no task of hotness in ood_set, this is undefined (but will probably return a sample from all ood tasks)
                # Filter the existing task_experts_out_dist for the given hotness
                task_ids = jax.random.choice(
                    key=rng_tasks,
                    a=len(self.task_experts_out_dist),
                    p=1.0 * jnp.all(
                        jnp.sum(self.task_experts_out_dist, axis=-1) == hotness, axis=-1
                    ),
                    shape=(num_tasks,),
                )
                embeddings = self.task_experts_out_dist[task_ids]
            elif hotness <= self.num_experts:
                # Randomly sample task_experts - everything is ood here
                @partial(jnp.vectorize, signature="(n)->(m)")
                def sample_single_comb(rng):
                    return jax.random.choice(rng, self.num_experts, replace=False, shape=(hotness, ))

                rngs_tasks = jax.random.split(rng_tasks, num_tasks * self.num_hidden_hnet).reshape(
                    (num_tasks, self.num_hidden_hnet, -1)
                )
                expert_indeces = sample_single_comb(rngs_tasks)
                embeddings = self.k_hot(expert_indeces)
                task_ids = -1 * jnp.ones((num_tasks,))  # No unique task IDs available here
            else:
                raise ValueError(f"Invalid hotness {hotness}")

        if self.continuous_combinations:
            # Sample weights uniformly from simplex (see Willms, 2021)
            weights = jax.random.exponential(rng_weights, shape=embeddings.shape)
            weights = weights * embeddings
            weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1)

            # Shift nonzero embeddings to the range [0.5, 1.0] to prevent adding further sparsity
            embeddings = (0.5 * weights + 0.5) * embeddings

        return embeddings, task_ids
