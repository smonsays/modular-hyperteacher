# Discovering modular solutions that generalize compositionally

Official code to reproduce experiments in [Discovering modular solutions that generalize compositionally](https://www.arxiv.com/MISSING). Code is based on [`metax`](https://github.com/smonsays/metax), a meta-learning research library in jax.


## Installation

Install jax according to the [instructions for your platform](https://jax.readthedocs.io/en/latest/installation.html) after which you can install the remaining dependencies with:
```
pip install -r requirements.txt
```

## Structure

All experiments have a corresponding sweep file in `sweeps/` and can be run using
```bash
`wandb sweep /sweeps/[folder]/[name].yaml`
```

where `[folder]` and `[name]` need to be replaced accordingly. 

Hyperparameters for all methods and experiments can be found in `configs/`. If you'd like to directly run a specific experiment for a single seed you can use:

```bash
python run_fewshot.py --config 'configs/[experiment].py:[method]'
```

where `experiment` can be
- `compositional_grid`
- `hyperteacher`
- `preference_grid`

and `method` can be
- `hnet_linear`
- `hnet_deepmlp`
- `anil512`
- `learned_init384`

For the empirical validation of the theory consider `run_theory.py`.

## Citation

If you use this code in your research, please cite the paper:

```
MISSING
```

## Acknowledgements
Research supported with Cloud TPUs from Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/).
