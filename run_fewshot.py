"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import logging
import os
import random
from datetime import datetime

import jax
import jax.tree_util as jtu
import optax
from absl import app, flags
from ml_collections import ConfigDict, config_flags

import metax
import wandb
from configs.common import dataset_config, model_config
from metax import data, energy, models, utils
from metax.callbacks import (callback_hparams, get_callback_analyze_embedding,
                             get_callback_decode_modules, get_callback_params)
from metax.experiment import CallbackEvent, FewshotExperiment

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", default="logs", help="Working directory.")
flags.DEFINE_bool("wandb", True, "Log to wandb.")
flags.DEFINE_string("wandb_entity", "", "Wandb entity.")
flags.DEFINE_string("wandb_project", "", "Wandb project.")
flags.DEFINE_bool("callback", False, "Toggling whether callbacks are used.")
jax.config.parse_flags_with_absl()  # Expose jax flags, (e.g. --jax_disable_jit )

config_flags.DEFINE_config_file(
    name="config",
    default="configs/hyperteacher.py:hnet_linear",
    help_string='Training configuration `configs/[experiment].py:[method]`.',
)


def run_fewshot(cfg, logbook=None):
    cfg = ConfigDict(cfg)

    # Prepare randomness
    if cfg["seed"] is None:
        cfg["seed"] = random.randint(0, 99999)
    rng = jax.random.PRNGKey(cfg["seed"])
    rng_init, rng_runner, rng_data = jax.random.split(rng, 3)

    # Allow specifying dataset and base_model configuration through
    # flat args for better compatibility with wandb sweeps
    if isinstance(cfg["dataset"], str):
        # All dataset argument names should start with `dataset_`
        cfg["dataset"] = dataset_config(
            cfg["dataset"],
            **{"_".join(k.split("_")[1:]): v for k, v in cfg.items() if "dataset_" in k},
        ).to_dict()

    if isinstance(cfg["base_model"], str):
        cfg["base_model"] = model_config(
            cfg["base_model"],
            **{"_".join(k.split("_")[2:]): v for k, v in cfg.items() if "base_model_" in k},
        ).to_dict()

    # Create metadatasets
    if cfg["dataset"]["name"] in [
        "family",
        "harmonic",
        "hyperteacher",
        "linear",
        "polynomial",
        "sawtooth",
        "sinusoid",
        "sinusoid_family",
    ]:
        metatrainset, metatestset, metavalidset, metaoodset, metauxsets = data.create_synthetic_metadataset(
            meta_batch_size=cfg["meta_batch_size"],
            num_tasks_train=cfg["steps_outer"] * cfg["meta_batch_size"],
            seed=cfg["seed"],
            **cfg["dataset"],
        )

        input_shape = metatrainset.input_shape
        output_dim = metatrainset.output_dim
        sample_input = metatrainset.sample_input
    elif cfg["dataset"]["name"] in ["compositional_grid", "compositional_preference"]:
        metatrainset, metatestset, metavalidset, metaoodset, metauxsets = data.create_imitation_metaloader(
            meta_batch_size=cfg["meta_batch_size"],
            num_tasks_train=cfg["steps_outer"] * cfg["meta_batch_size"],
            seed=cfg["seed"],
            **cfg["dataset"],
        )
        input_shape = metatrainset.input_shape
        output_dim = metatrainset.output_dim
        sample_input = metatrainset.sample_input
    else:
        raise ValueError

    # Create data-dependent loss functions and meta-model
    if cfg["dataset"]["name"] == "sinusoid":
        loss_fn_inner = energy.SquaredError(reduction="sum")
        loss_fn_outer = energy.SquaredError(reduction="sum")

    elif cfg["dataset"]["name"] in [
        "family",
        "harmonic",
        "linear",
        "polynomial",
        "sawtooth",
        "sinusoid_family",
    ]:
        loss_fn_inner = energy.SquaredError(reduction="mean")
        loss_fn_outer = energy.SquaredError(reduction="mean")

    elif "hyperteacher" in cfg["dataset"]["name"]:
        if cfg["dataset"]["classification"]:
            loss_fn_inner = energy.KLDivergence(reduction="mean")
            loss_fn_outer = energy.KLDivergence(reduction="mean")
        else:
            loss_fn_inner = energy.SquaredError(reduction="mean")
            loss_fn_outer = energy.SquaredError(reduction="mean")

    elif "compositional_grid" in cfg["dataset"]["name"]:
        loss_fn_inner = energy.CrossEntropyMasked(reduction="mean")
        loss_fn_outer = energy.CrossEntropyMasked(reduction="mean")

    elif "compositional_preference" in cfg["dataset"]["name"]:
        loss_fn_inner = energy.SquaredErrorMasked(reduction="mean")
        loss_fn_outer = energy.SquaredErrorMasked(reduction="mean")

    else:
        raise ValueError

    if cfg["meta_model"] == "anil":
        if cfg["base_model"]["type"] == "mlp":
            base_model = models.MultilayerPerceptron(
                output_sizes=[cfg["base_model"]["hidden_dim"]] * cfg["base_model"]["num_hidden"],
                activate_final=True,
                batch_norm=cfg["base_model"]["batch_norm"],
            )
        else:
            raise ValueError

        meta_model = metax.module.AlmostNoInnerLoop(
            loss_fn_inner=loss_fn_inner,
            loss_fn_outer=loss_fn_outer,
            body=base_model,
            output_dim=output_dim,
        )
    elif "hnet" in cfg["meta_model"]:
        if cfg["base_model"]["type"] == "mlp":
            assert cfg["base_model"]["num_hidden"] > 0, "Need at least 1 hidden layer."
            base_model_input = models.MultilayerPerceptron(
                output_sizes=[cfg["base_model"]["hidden_dim"]],
                activate_final=True,
                batch_norm=cfg["base_model"]["batch_norm"],
                reparametrized_linear=True,
            )
            base_model_hidden = models.MultilayerPerceptron(
                output_sizes=[cfg["base_model"]["hidden_dim"]] * (cfg["base_model"]["num_hidden"] - 1),
                activate_final=True,
                batch_norm=cfg["base_model"]["batch_norm"],
                reparametrized_linear=True,
            )
            base_model_output = models.MultilayerPerceptron(
                output_sizes=[output_dim],
                activate_final=False,
                batch_norm=cfg["base_model"]["batch_norm"],
                reparametrized_linear=True,
            )

        else:
            raise ValueError

        # If ratio_templates_experts is set, use it to determine the number of templates
        assert not cfg.get("ratio_templates_experts", 0) or not cfg.get("num_templates", 0)
        if cfg.get("ratio_templates_experts", 0):
            num_templates = cfg["ratio_templates_experts"] * cfg["dataset"]["num_experts"]
            logging.info("Using `ratio_templates_experts` to set num_templates to {}".format(num_templates))
        else:
            num_templates = cfg["num_templates"]
            logging.info("Using `num_templates` to set num_templates to {}".format(num_templates))

        meta_model = metax.module.MetaHypernetwork(
            loss_fn_inner=loss_fn_inner,
            loss_fn_outer=loss_fn_outer,
            target_network_input=base_model_input,
            target_network_hidden=base_model_hidden,
            target_network_output=base_model_output,
            input_shape=input_shape,
            output_dim=output_dim,
            hidden_dim=cfg["base_model"]["hidden_dim"],
            num_templates=num_templates,
            chunking=cfg["chunking"],
            weight_generator=cfg["meta_model"].split("_")[0],
            embedding_nonlinearity=cfg["embedding_nonlinearity"],
            embedding_dropout=cfg.get("embedding_dropout", None),
            embedding_norm_stop_grad=cfg.get("embedding_norm_stop_grad", True),
            embedding_normalization=cfg.get("embedding_normalization", True),
            embedding_constant_init=cfg.get("embedding_constant_init", False),
            hnet_init=cfg["hnet_init"],
            l1_reg=cfg.get("l1_reg", None),
            l2_reg=cfg.get("l2_reg", None),
            zero_threshold=cfg.get("zero_threshold", 0),
            fast_bias=cfg.get("fast_bias", False)
        )
    elif cfg["meta_model"] == "learned_init":
        if cfg["base_model"]["type"] == "mlp":
            base_model = models.MultilayerPerceptron(
                output_sizes=[cfg["base_model"]["hidden_dim"]] * cfg["base_model"]["num_hidden"] + [output_dim],
                batch_norm=cfg["base_model"]["batch_norm"],
            )
        else:
            raise ValueError

        meta_model = metax.module.LearnedInit(
            loss_fn_inner=loss_fn_inner,
            loss_fn_outer=loss_fn_outer,
            base_learner=base_model,
            reg_strength=cfg["l2_reg"],
        )
    else:
        raise ValueError('Model "{}" not defined.'.format(cfg["meta_model"]))

    # Create optimisers
    optim_fn_inner = utils.create_optimizer(cfg["optim_inner"], {"learning_rate": cfg["lr_inner"]})

    if cfg.get("schedule_outer", None) == "warmup":
        schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=cfg["lr_outer"],
            transition_steps=1000,
        )
    elif cfg.get("schedule_outer", None) == "warmup_cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg["lr_outer"],
            warmup_steps=100,
            decay_steps=cfg["steps_outer"],
            end_value=1e-7,
        )
    else:
        schedule = cfg["lr_outer"]

    optim_fn_outer = getattr(optax, cfg["optim_outer"])(
        learning_rate=schedule, weight_decay=cfg["weight_decay_outer"]
    )

    if "max_meta_grad_norm" in cfg:
        optim_fn_outer = optax.chain(
            optax.clip_by_global_norm(cfg["max_meta_grad_norm"]),
            optim_fn_outer,
        )

    if cfg.get("ignore_nan_grads_outer", False):
        optim_fn_outer = optax.apply_if_finite(optim_fn_outer, 1)

    # Setup the meta-learning algorithm
    if cfg["method"] == "maml":
        meta_learner = metax.learner.ModelAgnosticMetaLearning(
            meta_model=meta_model,
            batch_size=cfg["batch_size"],
            steps_inner=cfg["steps_inner"],
            optim_fn_inner=optim_fn_inner,
            optim_fn_outer=optim_fn_outer,
            first_order=cfg["first_order"],
        )
    elif cfg["method"] == "reptile":
        meta_learner = metax.learner.Reptile(
            meta_model=meta_model,
            batch_size=cfg["batch_size"],
            steps_inner=cfg["steps_inner"],
            optim_fn_inner=optim_fn_inner,
            optim_fn_outer=optim_fn_outer,
        )
    else:
        raise ValueError('Method "{}" not defined.'.format(cfg["method"]))

    # Log number of params and hparams
    rng_unused = jax.random.PRNGKey(0)
    hparams, hstate = meta_model.reset_hparams(rng_unused, sample_input)
    params, _ = meta_model.reset_params(rng_unused, hparams, hstate, sample_input)

    logging.info(
        "hparams: {}, params: {}".format(
            sum(x.size for x in jtu.tree_leaves(hparams)),
            sum(x.size for x in jtu.tree_leaves(params)),
        )
    )

    # Setup runner
    runner = FewshotExperiment(
        meta_learner=meta_learner,
        meta_batch_size=cfg["meta_batch_size"],
        steps_outer=cfg["steps_outer"],
        steps_inner_test=cfg.get("steps_inner_test", default=cfg["steps_inner"]),
        metatrainset=metatrainset,
        metavalidset=metavalidset,
        metatestset=metatestset,
        metaoodset=metaoodset,
        metauxsets=metauxsets,
        logbook=logbook,
    )

    if logbook is not None and FLAGS.callback:
        # Add callbacks
        runner.add_callback(CallbackEvent.START, callback_hparams)
        runner.add_callback(CallbackEvent.END, callback_hparams)

        runner.add_callback(CallbackEvent.START, get_callback_params(metavalidset))
        runner.add_callback(CallbackEvent.END, get_callback_params(metavalidset))

        if "hnet" in cfg["meta_model"]:
            callback_fn = get_callback_analyze_embedding(metavalidset)
            runner.add_callback(CallbackEvent.STEP, callback_fn)
            runner.add_callback(CallbackEvent.END, callback_fn)

        if (
            cfg["dataset"]["name"] == "hyperteacher"
            or cfg["dataset"]["name"] == "compositional_grid"
            or cfg["dataset"]["name"] == "compositional_preference"
        ):
            callback_fn = get_callback_decode_modules(metatestset, metaoodset)
            runner.add_callback(CallbackEvent.STEP, callback_fn)

    # Run
    meta_state_init = runner.reset(rng_init, sample_input)
    return runner.run(rng_runner, meta_state_init)


def main(argv):
    # Setup config and logger
    cfg = FLAGS.config
    dataset_name = cfg["dataset"] if isinstance(cfg["dataset"], str) else cfg["dataset"]["name"]

    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')[:-3] + "_fewshot_{}_{}_{}".format(
        cfg["method"], cfg["meta_model"], dataset_name
    )
    log_dir = utils.setup_logging(run_id, FLAGS.workdir)

    if FLAGS.wandb:
        wandb.init(
            config=cfg,
            entity=FLAGS.wandb_entity,
            project=FLAGS.wandb_project,
            dir=log_dir,
            mode="online",
        )
        logbook = wandb
    else:
        logbook = None

    # Start the actual run
    logging.info("Running on {}".format(jax.default_backend()))
    logging.info("Start training with parametrization:\n{}".format(cfg))
    data.save_dict_as_json(cfg.to_dict(), run_id + "_config", log_dir)

    # with jax.disable_jit():
    meta_state = run_fewshot(cfg, logbook=logbook)

    # Save model state
    data.save_pytree(os.path.join(log_dir, run_id + "_model"), meta_state)

    if logbook is not None:
        wandb.finish()
        utils.zip_and_remove(os.path.join(log_dir, "wandb"))


if __name__ == "__main__":
    app.run(main)
