"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import ml_collections as mlc

from configs.common import model_config


def get_config():

    config = mlc.ConfigDict(type_safe=False)

    config = mlc.ConfigDict(type_safe=False)
    config.name = "hnet_linear"

    config.base_model = model_config("mlp", hidden_dims=(16, 16, 16))
    config.dataset = "hyperteacher"
    # config.dataset_num_encounters = 3  # corresponds now to num_ood = 4
    config.dataset_targets_temperature = 0.1
    config.dataset_frac_ood = 0.25
    config.dataset_num_experts = 4
    config.dataset_num_hot = 2
    config.dataset_num_hidden = 3
    config.dataset_continuous_combinations = True
    config.dataset_hidden_dim = 8
    config.dataset_chunking = True
    config.steps_outer = 1000
    config.dataset_num_tasks_train = 1000

    config.first_order = False
    config.meta_batch_size = 32
    config.method = "maml"
    config.random_params_init = True
    config.optim_outer = "adamw"
    config.seed = 2023

    config.batch_size = 32
    config.chunking = True
    config.embedding_dropout = 0.0
    config.embedding_nonlinearity = "linear"
    config.hnet_init = "default"
    config.l1_reg = 0.0
    config.l2_reg = 0.0
    config.lr_inner = 0.3
    config.lr_outer = 0.001
    config.meta_model = "linear_hnet"
    config.num_templates = 6
    config.optim_inner = "adamw"
    config.schedule_outer = True
    config.steps_inner = 10

    return config
