"""
Copyright (c) Seijin Kobayashi
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

    config.dataset = "compositional_preference"

    config.dataset_shots_train = 16
    config.dataset_shots_test = 16
    config.dataset_num_preferences = 8
    config.dataset_num_features = 8
    config.dataset_num_objects = 4
    config.dataset_num_hot = 3
    config.dataset_continuous_combinations = True
    config.dataset_discount = 0.9
    config.dataset_frac_ood = 0.25
    config.dataset_timelimit = 8
    config.dataset_train_test_split = True
    config.dataset_task_support = "random"
    config.steps_outer = 100_000

    config.first_order = False
    config.meta_batch_size = 128
    config.method = "maml"
    config.optim_outer = "adamw"
    config.weight_decay_outer = 1e-2
    config.schedule_outer = None
    config.seed = 2023
    config.ignore_nan_grads_outer = True

    if model == "hnet_deepmlp":
        # hparams: 1'162'816, params: 96
        config.base_model = "mlp"
        config.base_model_hidden_dim = 64
        config.base_model_num_hidden = 2  # 2 works as well as 3

        config.batch_size = None
        config.chunking = True
        config.embedding_dropout = 0.0
        config.embedding_nonlinearity = "linear"
        config.hnet_init = "default"
        config.l1_reg = 0.0
        config.l2_reg = 0.0
        config.lr_inner = 0.1
        config.lr_outer = 0.0003
        config.max_meta_grad_norm = 2.0
        config.meta_model = "hnet_deepmlp"
        config.num_templates = 32
        config.optim_inner = "adamw"
        config.steps_inner = 10

    elif model == "hnet_linear":
        # hparams: 1'149'216, params: 133
        config.base_model = "mlp"
        config.base_model_hidden_dim = 64
        config.base_model_num_hidden = 3

        config.batch_size = None
        config.chunking = True
        config.fast_bias = True
        config.embedding_dropout = 0.0
        config.embedding_nonlinearity = "linear"
        config.hnet_init = "default"
        config.l1_reg = 0.0
        config.l2_reg = 0.0
        config.lr_inner = 0.1
        config.lr_outer = 0.0003
        config.max_meta_grad_norm = 1.0
        config.meta_model = "linear_hnet"
        config.num_templates = 32
        config.optim_inner = "adamw"
        config.steps_inner = 10

    elif model == "anil512":
        # hparams: 1041925, params: 2565
        config.base_model = "mlp"
        config.base_model_hidden_dim = 512
        config.base_model_num_hidden = 4

        config.batch_size = 64
        config.lr_inner = 0.01
        config.lr_outer = 0.0003
        config.max_meta_grad_norm = 2.0
        config.meta_model = "anil"
        config.optim_inner = "adamw"
        config.steps_inner = 100

    elif model == "learned_init368":
        # hparams: 454117, params: 454117
        config.base_model = "mlp"
        config.base_model_hidden_dim = 368  # NOTE 512 is too large for the 3090
        config.base_model_num_hidden = 3

        config.batch_size = None
        config.l2_reg = None
        config.lr_inner = 0.01
        config.lr_outer = 0.001
        config.max_meta_grad_norm = 2.0
        config.meta_model = "learned_init"
        config.optim_inner = "adamw"
        config.steps_inner = 10
    else:
        raise ValueError

    return config
