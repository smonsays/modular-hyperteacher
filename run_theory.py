"""
Copyright (c) Seijin Kobayashi
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import itertools
import math
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

import wandb
from metax.models.regression import RidgeRegression


class MultitaskNet(hk.Module):
    def __init__(
        self,
        dim_input,
        dim_hidden,
        dim_output,
        dim_task,
        use_bias=False,
        num_hidden_layers=1,
        name="multitasknet",
    ):
        super().__init__(name=name)

        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.num_hidden_layers = num_hidden_layers
        self.dim_output = dim_output
        self.dim_task = dim_task
        self.use_bias = use_bias

        self.weight_shape = []
        dim_input = self.dim_input
        for _ in range(self.num_hidden_layers):
            self.weight_shape.append((dim_input, self.dim_hidden))
            dim_input = self.dim_hidden

    def __call__(self, x, e):
        h = x
        for l, s in enumerate(self.weight_shape):
            hnet = hk.Linear(np.prod(s), name="hnet_{}".format(l), with_bias=False)
            W = jnp.reshape(hnet(e), s)
            u = (h @ W) / math.sqrt(s[0])
            h = jax.nn.relu(u)
        head = hk.Linear(self.dim_output, name="head", with_bias=self.use_bias)
        y = head(h)

        return y


class UniformData:
    def __init__(self, dim_data, seed, frac_ood):
        self.dim_data = dim_data
        self.seed = seed
        self.frac_ood = frac_ood
        self.eval_keys = ["train", "eval"]

    def __call__(self, rng, key):
        if key == "train":
            return jax.random.uniform(
                rng, shape=(self.dim_data,), minval=-math.sqrt(3), maxval=math.sqrt(3)
            )
        if key == "eval":
            scale = self.frac_ood / (1 - self.frac_ood)
            return jax.random.uniform(
                rng, shape=(self.dim_data,), minval=math.sqrt(3), maxval=2 * math.sqrt(3) * scale
            )


class SphereData:
    def __init__(self, dim_data, seed):
        self.dim_data = dim_data
        self.seed = seed
        self.eval_keys = ["train", "eval"]

    def __call__(self, rng, key):
        data = jax.random.normal(rng, shape=(self.dim_data,))
        norm = math.sqrt(self.dim_data)
        randvect = jax.random.normal(jax.random.PRNGKey(self.seed), shape=(self.dim_data,))
        if key == "train":
            flip = 2 * (randvect @ data < 0) - 1
        elif key == "eval":
            flip = 2 * (randvect @ data > 0) - 1
        data = data * flip
        return data / jnp.linalg.norm(data) * norm


class NormalData:
    def __init__(self, dim_data, seed):
        self.dim_data = dim_data
        self.seed = seed
        self.eval_keys = ["train", "eval"]

    def __call__(self, rng, key):
        data = jax.random.normal(rng, shape=(self.dim_data,))
        randvect = jax.random.normal(jax.random.PRNGKey(self.seed), shape=(self.dim_data,))
        if key == "train":
            flip = 2 * (randvect @ data < 0) - 1
        elif key == "eval":
            flip = 2 * (randvect @ data > 0) - 1
        data = data * flip
        return data


class SparseData:
    def __init__(self, dim_data, seed, frac_ood, num_hot, continuous, manual_train=None):
        self.dim_data = dim_data
        self.seed = seed
        self.frac_ood = frac_ood
        self.num_hot = num_hot
        self.continuous = continuous
        self.experts = {}
        self.eval_keys = ["train"]

        if manual_train is not None and len(manual_train) > 0:
            print("Training tasks:", manual_train)
            self.experts["train"] = jax.vmap(self.k_hot)(np.array(manual_train))
            for h in range(1, dim_data + 1):
                self.experts[f"eval_{h/self.dim_data:.3f}"] = self.get_k_hot_experts(h)
                self.eval_keys.append(f"eval_{h/self.dim_data:.3f}")
        else:
            less_hot_experts = []
            for h in range(1, num_hot):
                less_hot_experts.append(self.get_k_hot_experts(h))

            exact_hot_experts = self.get_k_hot_experts(num_hot)
            exact_hot_experts = jax.random.permutation(
                jax.random.PRNGKey(self.seed), exact_hot_experts
            )
            num_ood = int(len(exact_hot_experts) * self.frac_ood)
            if num_ood > 0:
                exact_hot_experts_in_dist = jnp.array(exact_hot_experts[:-num_ood])
                self.experts["train"] = jnp.concatenate(
                    less_hot_experts + [exact_hot_experts_in_dist]
                )

                exact_hot_experts_out_dist = jnp.array(exact_hot_experts[-num_ood:])
                self.experts["eval"] = exact_hot_experts_out_dist
                self.eval_keys.append("eval")
            else:
                self.experts["train"] = jnp.concatenate(
                    less_hot_experts + [jnp.array(exact_hot_experts)]
                )

            for h in range(1, dim_data + 1):
                self.experts[f"eval_{h/self.dim_data:.3f}"] = self.get_k_hot_experts(h)
                self.eval_keys.append(f"eval_{h/self.dim_data:.3f}")

    def get_k_hot_experts(self, k):
        perms = itertools.combinations(range(self.dim_data), k)
        experts_idx = np.array(list(perms)).reshape(-1, k)
        experts_k_hot = jax.vmap(self.k_hot)(experts_idx)
        return experts_k_hot

    def k_hot(self, idx):
        return (jnp.sum(jax.nn.one_hot(idx, self.dim_data), axis=0) > 0) * 1.0

    def __call__(self, rng, key):
        """
        Sample expert panels defining each task.
        """
        rng_tasks, rng_weights = jax.random.split(rng)
        task_experts = self.experts[key]
        task_ids = jax.random.choice(rng_tasks, len(task_experts), shape=())
        embeddings = task_experts[task_ids]

        if self.continuous or key != "train":
            # Sample weights uniformly from simplex (see Willms, 2021)
            weights = jax.random.exponential(rng_weights, shape=embeddings.shape)
            weights = weights * embeddings
            weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1)

            # Shift nonzero embeddings to the range [0.5, 1.0] to prevent adding further sparsity
            embeddings = (0.5 * weights + 0.5) * embeddings
        else:
            embeddings = embeddings / jnp.linalg.norm(embeddings)
        return embeddings


def run(cfg):
    # Some heuristic
    n_steps = cfg["train_step"] * cfg["n_task"]

    # Define data
    x_data = UniformData(cfg["n_input"], cfg["seed"], 0)

    if cfg["datatype"] == "uniform":
        print("Using Uniform tasks")
        e_data = UniformData(cfg["n_task"], cfg["seed"])
    if cfg["datatype"] == "sphere":
        print("Using Uniform tasks")
        e_data = SphereData(cfg["n_task"], cfg["seed"])
    if cfg["datatype"] == "normal":
        print("Using Uniform tasks")
        e_data = NormalData(cfg["n_task"], cfg["seed"])
    if cfg["datatype"] == "sparse":
        print("Using Sparse tasks")
        e_data = SparseData(
            cfg["n_task"],
            cfg["seed"],
            cfg["frac_ood"],
            math.floor(cfg["n_hot_fraction"] * cfg["n_task"]),
            cfg["continuous_combinations"],
            [[int(ss) for ss in s.split(",")] for s in cfg["manual_task"].split(";")]
            if len(cfg["manual_task"]) > 0
            else None,
        )

    def get_x(rng, batch_size):
        return jax.vmap(lambda r: x_data(r, "train"))(jax.random.split(rng, batch_size))

    def get_e(r):
        return e_data(r, "train")

    # Define Modules
    teacher = hk.without_apply_rng(
        hk.transform(
            lambda x, e: MultitaskNet(
                cfg["n_input"],
                cfg["n_hidden"],
                cfg["n_output"],
                cfg["n_task"],
                num_hidden_layers=cfg["n_hidden_layers"],
                use_bias=cfg["use_bias"],
            )(x, e)
        )
    )

    student = hk.without_apply_rng(
        hk.transform(
            lambda x, e: MultitaskNet(
                cfg["n_input"],
                int(cfg["n_hidden"] * cfg["n_hidden_student_factor"]),
                cfg["n_output"],
                int(cfg["n_task_student_factor"] * cfg["n_task"]),
                num_hidden_layers=cfg["n_hidden_layers_student"],
                use_bias=cfg["use_bias"],
            )(x, e)
        )
    )

    ### Define optimizers
    def loss_fn(pred, target):
        return 0.5 * ((target - pred) ** 2).mean()
    
    def relative_wrt_sample_loss_fn(pred, target):
        return 0.5 * (((target - pred) ** 2).mean(axis=-1) / (target**2).mean(axis=-1)).mean()

    def relative_wrt_population_loss_fn(pred, target):
        return 0.5 * (((target - pred) ** 2).mean() / (target**2).mean())

    if cfg["scheduler"] == "multistep":
        scheduler = optax.piecewise_constant_schedule(
            cfg["learning_rate"],
            boundaries_and_scales={
                int(n_steps * cfg["lr_step_every"] * i): cfg["lr_step_factor"]
                for i in range(1, math.ceil(1 / cfg["lr_step_every"]))
            },
        )
    elif cfg["scheduler"] == "cosine":
        scheduler = optax.cosine_decay_schedule(
            cfg["learning_rate"], n_steps, alpha=cfg["cosine_alpha"]
        )
    else:
        raise ValueError("unknown scheduler")
    optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=scheduler, weight_decay=cfg["weight_decay"])
    inner_optimizer = optax.adam(learning_rate=cfg["inner_learning_rate"])

    # Initialize
    rng = jax.random.PRNGKey(cfg["seed"])
    rng_train, rng_eval, rng_student, rng_teacher = jax.random.split(rng, 4)
    dummy_x = jnp.zeros(cfg["n_input"])
    dummy_e = jnp.zeros(cfg["n_task"])
    dummy_e_student = jnp.zeros(int(cfg["n_task_student_factor"] * cfg["n_task"]))
    student_params_init = student.init(rng_student, dummy_x, dummy_e_student)
    teacher_params = teacher.init(rng_teacher, dummy_x, dummy_e)
    optim_state_init = optimizer.init(student_params_init)

    @jax.jit
    def infer_task(rng, student_params, e):
        if cfg["embedding_init"] == "normal":
            e_inferred = jax.random.normal(
                rng, (int(cfg["n_task_student_factor"] * cfg["n_task"]),)
            )
        elif cfg["embedding_init"] == "constant":
            e_inferred = jax.random.normal(
                rng_student, (int(cfg["n_task_student_factor"] * cfg["n_task"]),)
            )
        inner_optim_state = inner_optimizer.init(e_inferred)

        def inner_step(carry, rng_x):
            e_inferred, inner_optim_state = carry
            x = get_x(rng_x, cfg["batch_size"])
            y = teacher.apply(teacher_params, x, e)

            def inner_loss(e):
                pred = student.apply(student_params, x, e)
                return loss_fn(pred, y)

            grads = jax.grad(inner_loss)(e_inferred)
            updates, inner_optim_state = inner_optimizer.update(grads, inner_optim_state)
            e_inferred = optax.apply_updates(e_inferred, updates)
            return (e_inferred, inner_optim_state), {}

        (e_inferred, _), _ = jax.lax.scan(
            inner_step, (e_inferred, inner_optim_state), jax.random.split(rng, cfg["inner_step"])
        )

        return e_inferred

    @jax.jit
    def meta_train(rng, student_params, optim_state):
        def outer_step(carry, rng):
            student_params, optim_state = carry

            rng_infer, rng_x, rng_e = jax.random.split(rng, 3)
            e = jax.vmap(get_e)(jax.random.split(rng_e, cfg["meta_batch_size"]))

            def outer_loss(student_params):
                e_inferred = jax.vmap(infer_task, in_axes=(0, None, 0))(
                    jax.random.split(rng_infer, cfg["meta_batch_size"]), student_params, e
                )
                if not cfg["second_order"]:
                    e_inferred = jax.lax.stop_gradient(e_inferred)

                x = jax.vmap(get_x, in_axes=(0, None))(
                    jax.random.split(rng_x, cfg["meta_batch_size"]), cfg["batch_size"]
                )
                y_teacher = jax.vmap(teacher.apply, in_axes=(None, 0, 0))(teacher_params, x, e)
                y_student = jax.vmap(student.apply, in_axes=(None, 0, 0))(
                    student_params, x, e_inferred
                )
                return loss_fn(y_student, y_teacher)

            loss, grads = jax.value_and_grad(outer_loss)(student_params)
            updates, optim_state = optimizer.update(grads, optim_state, student_params)
            student_params = optax.apply_updates(student_params, updates)
            return (student_params, optim_state), {
                "outer_loss": loss,
                "learning_rate": optim_state.hyperparams["learning_rate"],
            }

        (student_params, optim_state), metric = jax.lax.scan(
            outer_step, (student_params, optim_state), jax.random.split(rng, cfg["log_every"])
        )
        return student_params, optim_state, metric

    def get_task_inferred_e(student_params, key):
        rng_infer, rng_e = jax.random.split(rng_eval, 2)
        e = jax.vmap(lambda r: e_data(r, key))(jax.random.split(rng_e, cfg["eval_meta_batch_size"]))
        e_inferred = jax.vmap(infer_task, in_axes=(0, None, 0))(
            jax.random.split(rng_infer, cfg["eval_meta_batch_size"]), student_params, e
        )

        # F, res, a, b = np.linalg.lstsq(e_inferred, e)
        return e, e_inferred

    def get_student_hnet(student_params):
        return student_params["multitasknet/hnet_0"]["w"].reshape(
            (
                int(cfg["n_task_student_factor"] * cfg["n_task"]),
                cfg["n_input"],
                int(cfg["n_hidden"] * cfg["n_hidden_student_factor"]),
            )
        )

    def get_teacher_hnet():
        return teacher_params["multitasknet/hnet_0"]["w"].reshape(
            (cfg["n_task"], cfg["n_input"], cfg["n_hidden"])
        )

    # def relative_mse(a, b):
    #     return (jnp.linalg.norm(a - b) ** 2 / jnp.linalg.norm(a) / jnp.linalg.norm(b)).mean()
    @jax.jit
    def eval_templates(F, student_params):
        student_weights_lin1 = get_student_hnet(student_params)
        w = jnp.linalg.pinv(F)

        student_weights_lin1 = jnp.einsum("Tt,TiO->tiO", w, student_weights_lin1)
        student_weights_lin1 = student_weights_lin1 / jnp.linalg.norm(
            student_weights_lin1, axis=1, keepdims=True
        )

        teacher_weights_lin1 = get_teacher_hnet()
        teacher_weights_lin1 = teacher_weights_lin1 / jnp.linalg.norm(
            teacher_weights_lin1, axis=1, keepdims=True
        )

        template_correlation = jnp.einsum(
            "tio,tiO->toO", teacher_weights_lin1, student_weights_lin1
        )
        template_alignment = jnp.max(template_correlation, axis=2)

        student_weights_lin1_all = student_weights_lin1 / jnp.linalg.norm(
            student_weights_lin1, axis=(0, 1), keepdims=True
        )
        teacher_weights_lin1_all = teacher_weights_lin1 / jnp.linalg.norm(
            teacher_weights_lin1, axis=(0, 1), keepdims=True
        )

        template_correlation_all = jnp.einsum(
            "tio,tiO->oO", teacher_weights_lin1_all, student_weights_lin1_all
        )
        template_alignment_all = jnp.max(template_correlation_all, axis=-1)

        return {
            "template_alignment_mean": template_alignment.mean(),
            "template_alignment_max": template_alignment.max(),
            "template_alignment_min": template_alignment.min(),
            "template_alignment_all_mean": template_alignment_all.mean(),
            "template_alignment_all_max": template_alignment_all.max(),
            "template_alignment_all_min": template_alignment_all.min(),
        }

    @jax.jit
    @partial(jax.vmap, in_axes=(None, None, None, -1, -1))  # vmap over experts
    def ridge_r2_score(rng, x_train, x_test, y_train, y_test):
        reg_jax = RidgeRegression(
            feature_dim=cfg["n_task"] * cfg["n_task_student_factor"], l2_reg=0.01, intercept=False
        )
        params = reg_jax.init(rng, x_train)
        params = reg_jax.fit(params, x_train, y_train)
        return reg_jax.score(params, x_test, y_test), params.weight

    @jax.jit
    def eval_inference(student_params, e, e_inferred):
        metric = {}
        x = jax.vmap(get_x, in_axes=(0, None))(
            jax.random.split(rng_eval, cfg["eval_meta_batch_size"]), cfg["eval_batch_size"]
        )

        y_teacher = jax.vmap(teacher.apply, in_axes=(None, 0, 0))(teacher_params, x, e)
        # y_student_projected = jax.vmap(teacher.apply, in_axes=(None, 0, 0))(teacher_params, x, e_inferred @ F)
        y_student = jax.vmap(student.apply, in_axes=(None, 0, 0))(student_params, x, e_inferred)
        metric["outer_loss"] = loss_fn(y_student, y_teacher)
        metric["outer_relative_wrt_sample_loss"] = relative_wrt_sample_loss_fn(y_student, y_teacher)
        metric["outer_relative_wrt_population_loss"] = relative_wrt_population_loss_fn(
            y_student, y_teacher
        )

        # metric["outer_loss_reconstructed"] =  loss_fn(y_student_projected, y_teacher)

        # metric["relative_residual_e"] = relative_mse(e, e_inferred @ F)

        student_weights_lin1 = get_student_hnet(student_params)
        student_weights_lin1 = jnp.einsum("bT,TiO->biO", e_inferred, student_weights_lin1)
        student_weights_lin1 = student_weights_lin1 / jnp.linalg.norm(
            student_weights_lin1, axis=(1), keepdims=True
        )

        teacher_weights_lin1 = get_teacher_hnet()
        teacher_weights_lin1 = jnp.einsum("bt,tio->bio", e, teacher_weights_lin1)
        teacher_weights_lin1 = teacher_weights_lin1 / jnp.linalg.norm(
            teacher_weights_lin1, axis=(1), keepdims=True
        )

        weight_correlation = jnp.einsum("bio,biO->boO", teacher_weights_lin1, student_weights_lin1)
        # Alignment when the mapping teacher node -> student node is task independent
        generated_weight_avr_alignment = jnp.max(weight_correlation.mean(0), axis=-1)
        # Alignment when the mapping teacher node -> student node is task dependent
        generated_weight_max_alignment = jnp.max(weight_correlation, axis=-1).mean(0)
        metric.update(
            {
                "generated_weight_avr_alignment_mean": generated_weight_avr_alignment.mean(),
                "generated_weight_avr_alignment_max": generated_weight_avr_alignment.max(),
                "generated_weight_avr_alignment_min": generated_weight_avr_alignment.min(),
                "generated_weight_max_alignment_mean": generated_weight_max_alignment.mean(),
                "generated_weight_max_alignment_max": generated_weight_max_alignment.max(),
                "generated_weight_max_alignment_min": generated_weight_max_alignment.min(),
            }
        )
        return metric

    params = student_params_init
    state = optim_state_init
    rng = rng_train
    for step in range(n_steps // cfg["log_every"]):
        print(f"Training iteraton [{(step+1)*cfg['log_every']:>5d}/{n_steps:>5d}]")
        rng, _ = jax.random.split(rng)
        params, state, metric = meta_train(rng, params, state)

        loss = metric["outer_loss"].mean()
        learning_rate = metric["learning_rate"].mean()
        log_dict = {"step": (step + 1) * cfg["log_every"], "loss": loss, "lr": learning_rate}

        print("Evaluating")
        e_train, e_inferred_train = get_task_inferred_e(params, "train")
        for ood_key in e_data.eval_keys:
            e_ood, e_inferred_ood = get_task_inferred_e(params, ood_key)
            coeffs, F = ridge_r2_score(rng_eval, e_inferred_train, e_inferred_ood, e_train, e_ood)
            for expert in range(coeffs.shape[0]):
                log_dict["_" + ood_key + f"_train_vs_ood_r2_expert_{expert}"] = coeffs[expert]
            log_dict.update(
                {
                    "_" + ood_key + "_train_vs_ood" + k: v
                    for (k, v) in eval_templates(F, params).items()
                }
            )

            coeffs, F = ridge_r2_score(rng_eval, e_inferred_ood, e_inferred_ood, e_ood, e_ood)
            for expert in range(coeffs.shape[0]):
                log_dict["_" + ood_key + f"_ood_vs_ood_r2_expert_{expert}"] = coeffs[expert]
            log_dict.update(
                {
                    "_" + ood_key + "_ood_vs_ood" + k: v
                    for (k, v) in eval_templates(F, params).items()
                }
            )

            log_dict.update(
                {
                    "_" + ood_key + "_" + k: v
                    for (k, v) in eval_inference(params, e_ood, e_inferred_ood).items()
                }
            )
        wandb.log(log_dict)


if __name__ == "__main__":
    cfg = {
        "seed": 42,
        # Architecture
        "use_bias": False,
        "n_hidden": 16,
        "n_hidden_student_factor": 2,
        "n_hidden_layers": 1,
        "n_hidden_layers_student": 1,
        "n_task_student_factor": 2,
        # Optimization
        "train_step": 10000,
        "learning_rate": 1e-3,
        "weight_decay": 0,
        "scheduler": "cosine",
        "lr_step_every": 0.3,
        "lr_step_factor": 0.3,
        "cosine_alpha": 0.001,
        "inner_learning_rate": 0.003,
        "inner_step": 300,
        "meta_batch_size": 64,
        "batch_size": 256,
        "embedding_init": "constant",
        # Data
        "problem": "regression",
        "data_seed": 42,
        "n_input": 16,
        "n_output": 4,
        "datatype": "sparse",
        "n_task": 6,
        # "manual_task": "0,1,2;2,3,4;4,5,0", #"0,1;1,2;2,0;3,4;4,5;5,3",
        "manual_task": "0,1;1,2;2,0;3,4;4,5;5,3",
        "n_hot_fraction": 0.5,
        "frac_ood": 0.0,
        "continuous_combinations": True,
        "log_every": 1000,
        "second_order": False,
        "eval_batch_size": 1024,
        "eval_meta_batch_size": 512,
    }

    wandb.init(
        config=cfg,
        entity="REPLACE",
        project="REPLACE",
        mode="online",
    )
    print(f"Running agent {wandb.run.id}")
    run(wandb.config)
