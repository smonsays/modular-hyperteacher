"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import math

import ml_collections as mlc


def dataset_config(dataset: str, **kwargs):
    config = mlc.ConfigDict()

    if dataset == "family":
        config.name = dataset
        config.num_tasks_test = 1000
        # config.num_tasks_train = 100000
        config.num_tasks_valid = 1000
        config.shots_test = 10
        config.shots_train = 10

    elif dataset in [
        "harmonic",
        "linear",
        "polynomial",
        "sawtooth",
        "sinusoid",
        "sinusoid_family",
    ]:
        config.name = dataset
        config.num_tasks_test = 100
        # config.num_tasks_train = 25000
        config.num_tasks_valid = 100
        config.shots_test = 10
        config.shots_train = 10

    elif dataset == "compositional_grid":
        config.name = dataset
        config.num_tasks_ood = 1024
        config.num_tasks_test = 1024
        config.num_tasks_valid = 1024
        config.shots_test = kwargs["shots_test"]
        config.shots_train = kwargs["shots_train"]
        config.grid_size = kwargs["grid_size"]
        config.num_interactions = kwargs["num_interactions"]
        config.num_objects = kwargs["num_objects"]
        config.num_mazes = kwargs["num_mazes"]
        config.num_distractors = kwargs["num_distractors"]
        config.frac_ood = kwargs["frac_ood"]
        config.task_support = kwargs["task_support"]
        config.train_test_split = kwargs["train_test_split"]

    elif dataset == "compositional_preference":
        config.name = dataset
        config.num_tasks_ood = 1024
        config.num_tasks_test = 1024
        config.num_tasks_valid = 1024
        config.shots_test = kwargs["shots_test"]
        config.shots_train = kwargs["shots_train"]
        config.num_preferences = kwargs["num_preferences"]
        config.num_features = kwargs["num_features"]
        config.num_objects = kwargs["num_objects"]
        config.num_hot = kwargs["num_hot"]
        config.continuous_combinations = kwargs["continuous_combinations"]
        config.discount = kwargs["discount"]
        config.frac_ood = kwargs["frac_ood"]
        config.timelimit = kwargs["timelimit"]
        config.task_support = kwargs["task_support"]
        config.train_test_split = kwargs["train_test_split"]

    elif dataset == "hyperteacher":
        config.name = dataset
        config.num_tasks_ood = 1024
        config.num_tasks_test = 1024
        # config.num_tasks_train = kwargs["num_tasks_train"]
        config.num_tasks_valid = 1024
        config.shots_test = kwargs["shots_test"]
        config.shots_train = kwargs["shots_train"]
        config.input_dim = 16
        config.output_dim = 8
        config.hidden_dim = kwargs["hidden_dim"]
        config.num_hidden = kwargs["num_hidden"]
        config.num_experts = kwargs["num_experts"]
        config.frac_ood = kwargs["frac_ood"]
        config.num_hot = kwargs["num_hot"]
        config.ood_sets_hot = [2**i for i in range(int(math.log2(kwargs["num_experts"])) + 1)]
        config.scale = 1.0
        config.classification = True
        config.normalize_classifier = kwargs.get("normalize_classifier", True)
        config.targets_temperature = kwargs["targets_temperature"]
        config.continuous_combinations = kwargs["continuous_combinations"]
        config.chunking = kwargs["chunking"]
        config.task_support = kwargs["task_support"]
        config.train_test_split = kwargs["train_test_split"]

    else:
        raise NotImplementedError

    return config


def model_config(model: str, **kwargs):
    config = mlc.ConfigDict()

    if model == "mlp":
        config.type = "mlp"
        config.batch_norm = False
        config.hidden_dim = kwargs["hidden_dim"]
        config.num_hidden = kwargs["num_hidden"]
    else:
        raise NotImplementedError

    return config
