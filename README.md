This repository contains the code and experiments for the TMLR paper *ViViT:
Curvature access through the generalized Gauss-Newton's low-rank structure*:

```bibtex

@article{dangel2022vivit,
  title =        {Vi{V}i{T}: Curvature Access Through The Generalized
                  Gauss-Newton{\textquoteright}s Low-Rank Structure},
  author =       {Felix Dangel and Lukas Tatzel and Philipp Hennig},
  journal =      {Transactions on Machine Learning Research (TMLR)},
  year =         2022,
}

```

# Reproducing the experiments

**Note:** Experiments were generated and verified to run on Ubuntu 20.04.3 LTS
with `python=3.8.5` and `pip==21.2.4`.

First, clone the repository and change into the repository's root:
```bash
git clone https://github.com/f-dangel/vivit-experiments.git
cd vivit-experiments
```

## Installation

### `conda` (recommended)

We recommend using the `conda` environment specified in
[.conda_eny.yml](`conda_eny.yml`). If you have `conda` installed, you can build
the environment using the command
```bash
conda env create --file .conda_env.yml
```

and load it with the command

```bash
conda activate vivit-experiments
```

(To disable the environment, run `conda deactivate`, to remove the environment,
run `conda env remove -n vivit-experiments`).

### Manual (alternative)

In your environment of choice, run the following commands

```bash
# main library requirements
pip install -r requirements.txt

# for development/experiments
pip install -r requirements-dev.txt
pip install -r exp/requirements-exp.txt

# main library
pip install -e .
```

## Reproducing our experiments (overview)

The experiments and instructions are contained in subdirectories of `exp/`.
Follow the instructions in their `README.md` files to run them:

  - [Spectral
    densities](exp/exp10_performance_ggn_evals/README.md#spectral-densities)
    (Figures 1.a, S.4, S.5)

  - [Critical batch sizes for the GGN
    eigenvalues](exp/exp10_performance_ggn_evals/README.md#ggn-eigenvalues)
    (Figures 2.a (top), S.8a (left), S.8b (left), S.9a (left), S.9b (left),
    S.10a (left), S.10b (left), S.11a (left), S.11b (left), S.12a (left), S.12b
    (left), S.14a (left), S.14b (left), S.16a (left), S.16b (left), S.17a
    (left), S.17b (left))

  - [Critical batch size for the GGN's leading
    eigenpair](exp/exp10_performance_ggn_evals/README.md#ggn-top-eigenpair)
    (Figures S.2a (bottom), S.8a (right), S.8b (right), S.9a (right), S.9b
    (right), S.10a (right), S.10b (right), S.11a (right), S.11b (right), S.12a
    (right), S.12b (right), S.14a (right), S.14b (right), S.16a (right), S.16b
    (right), S.17a (right), S.17b (right))

  - [Run time comparison ViViT vs. power iteration for computing the `k` leading
    eigenvalues](exp/exp10_performance_ggn_evals/README.md#runtime-performance)
    (Figures 2.b, S.8d, S.8d, S.9c, S.9d, S.10c, S.10d, S.11c, S.11d, S.12c,
    S.12d, S.13c, S.13d, S.14c, S.14d, S.15c, S.15d, S.16c, S.16d, S.17c, S.17d)

  - [Overlap between different curvature matrices during
    training](exp/exp13_full_batch_monitoring/README.md)
    (Figures 3, S.18, S.19, S.20, S.21, S.22, S.23, S.24, S.25, S.26)

  - [Convergence hyperparameter analysis of the power
    iteration](exp/exp10_performance_ggn_evals/README.md#tolerance-experiment)
    (Figure S.6)

  - [Comparison between power iteration on the Gram matrix vs. on the
    GGN](exp/exp10_performance_ggn_evals/README.md#power-iteration-on-the-gram-matrix-vs-on-the-ggn)
    (Figure S.7)

  - [Critical batch sizes for damped Newton
    step](exp/exp10_performance_ggn_evals/README.md#damped-newton-step)
    (Table S.2)

  - [Learning rate grid search for SGD and Adam on the ResNet32 CIFAR-10 test
    problem](exp/exp15_resnet_gridsearch/README.md)
    (Table S.3)

  - [Run time comparison between naive and optimized approach for Gram matrix
    computation for linear layers](exp/exp17_optimized_gram/README.md)
    (Run time comparison in Appendix C.1)
