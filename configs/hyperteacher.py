"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import ml_collections as mlc


def get_config(model):

    config = mlc.ConfigDict(type_safe=False)
    config.name = model

    config.dataset = "hyperteacher"
    config.dataset_targets_temperature = 0.1
    config.dataset_frac_ood = 0.25
    config.dataset_num_hot = 2
    config.dataset_num_experts = 4
    config.dataset_num_hidden = 3
    config.dataset_hidden_dim = 32
    config.dataset_normalize_classifier = True
    config.dataset_shots_train = 256
    config.dataset_shots_test = 256
    config.dataset_train_test_split = True
    config.steps_outer = 200_000
    # config.dataset_num_tasks_train = 100_000
    config.dataset_continuous_combinations = False
    config.dataset_chunking = False
    config.dataset_task_support = "random"
    config.ratio_templates_experts = 4

    config.first_order = False
    config.meta_batch_size = 128
    config.method = "maml"
    config.optim_outer = "adamw"
    config.weight_decay_outer = 1e-4
    config.schedule_outer = None
    config.seed = 2023
    config.ignore_nan_grads_outer = True

    if model == "hnet_deepmlp":
        # hparams: 648192, params: 256
        config.base_model = "mlp"
        config.base_model_hidden_dim = 128
        config.base_model_num_hidden = 3

        config.batch_size = None
        config.chunking = True
        config.embedding_dropout = 0.0
        config.embedding_nonlinearity = "linear"
        config.hnet_init = "default"
        config.l1_reg = 0.0
        config.l2_reg = 0.0
        config.lr_inner = 0.3
        config.lr_outer = 0.003
        config.max_meta_grad_norm = 1.0
        config.meta_model = "hnet_deepmlp"
        # config.num_templates = 64
        config.optim_inner = "adamw"
        config.steps_inner = 10
        config.weight_decay_outer = 1e-2
        # config.steps_inner_test = 100

    elif model == "hnet_linear":
        # hparams: 1262336, params: 264
        config.base_model = "mlp"
        config.base_model_hidden_dim = 128
        config.base_model_num_hidden = 3

        config.batch_size = None
        config.chunking = True
        config.fast_bias = True
        config.embedding_dropout = 0.0
        config.embedding_nonlinearity = "linear"
        config.hnet_init = "default"
        config.l1_reg = 0.0
        config.l2_reg = 0.0
        config.lr_inner = 0.3
        config.lr_outer = 0.001
        config.max_meta_grad_norm = 1.0
        config.meta_model = "linear_hnet"
        # config.num_templates = 64
        config.optim_inner = "adamw"
        config.steps_inner = 10
        config.weight_decay_outer = 1e-2
        # config.steps_inner_test = 100

    elif model == "anil512":
        # hparams: 538120, params: 4104
        config.base_model = "mlp"
        config.base_model_hidden_dim = 512
        config.base_model_num_hidden = 3

        config.batch_size = 64
        config.lr_inner = 0.03
        config.lr_outer = 0.001
        config.max_meta_grad_norm = 2.0
        config.meta_model = "anil"
        config.optim_inner = "adamw"
        config.steps_inner = 100

    elif model == "learned_init384":
        # hparams: 305288, params: 305288
        config.base_model = "mlp"
        config.base_model_hidden_dim = 384  # NOTE 512 is too large for the 3090
        config.base_model_num_hidden = 3

        config.batch_size = None
        config.l2_reg = None
        config.lr_inner = 0.01
        config.lr_outer = 0.001
        config.max_meta_grad_norm = 2.0
        config.meta_model = "learned_init"
        config.optim_inner = "adamw"
        config.steps_inner = 10

    elif model == "no_meta":
        # hparams: 137992, params: 137992
        config.base_model = "mlp"
        config.base_model_hidden_dim = 256
        config.base_model_num_hidden = 3

        config.batch_size = None
        config.l2_reg = None
        config.lr_inner = 1.0
        config.lr_outer = 0.0003
        config.max_meta_grad_norm = 2.0
        config.method = "reptile"
        config.meta_model = "learned_init"
        config.optim_inner = "sgd"
        config.steps_inner = 1
        config.steps_outer = 100_000

    else:
        raise ValueError

    return config
