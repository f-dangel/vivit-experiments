In this experiment, we explore the computational limits of our approach under
different approximation schemes to reduce cost.

**Hardware information:**

- CPU: Intel® Core™ i7-8700K CPU @ 3.70GHz × 12 (31.3 GiB)
- GPU: GeForce RTX 2080 Ti (11264 MB)

# Memory performance #

**Setting & Critical batch size:** We consider different tasks that are
computed during a regular gradient computation with PyTorch. Under different GGN
approximations (exact, MC=1-sampled), allocated samples (full mini-batch, fraction of
mini-batch), and parameter groupings (full net, layerwise), we find the critical batch
size (maximum batch size that does not lead to out-of-memory errors) using
bisection.

## GGN eigenvalues ##

We find the critical batch size for computing the GGN's nontrivial eigenvalues
from the Gram matrix. To reproduce the results:

1. (Optional) Extract `results.zip` to use the original data with `unzip
   results.zip`
2. Run `python call_run_N_crit_evals.py`. Find the critical batch sizes under
   `results/N_crit/evals`.
3. Create the LaTeX tables with `python tab_N_crit_evals.py`.
4. Clean up with `bash clean.sh` to remove the results.

## GGN top eigenpair ##

We find the critical batch size for computing the GGN's leading eigenpair. To
reproduce the results:

1. (Optional) Extract `results.zip` to use the original data with `unzip
   results.zip`
2. Run `python call_run_N_crit_evecs.py`. Find the critical batch sizes under
   `results/N_crit/evecs`.
3. Create the LaTeX tables with `python tab_N_crit_evecs.py`.
4. Clean up with `bash clean.sh` to remove the results.

## Damped Newton step ##

We find the critical batch size for computing a damped Newton step. To reproduce
the results:

1. (Optional) Extract `results.zip` to use the original data with `unzip
   results.zip`
2. Run `python call_run_N_crit_newton_step.py`. Find the critical batch sizes
   under `results/N_crit/newton_step`.
3. Create the LaTeX tables with `python tab_N_crit_newton_step.py`.
4. Clean up with `bash clean.sh` to remove the results.

# Runtime performance

We repeatedly measure the runtime of computing the `k` leading GGN eigenpairs
for the full matrix and a per-layer block-diagonal approximation (`k` eigenpairs
computed per block). Within our approach, we use different approximations of the
GGN. The baseline is a power iteration based on matrix-free multiplication with
the GGN. For the visualization, we report the smallest run times among repeated runs.
To reproduce the results:

1. (Optional) Extract `results.zip` to use the original data with `unzip
   results.zip`
2. Run `python call_run_time_evecs.py` to execute the runtime measurements of
   our approach. Find the run times in `results/time/evecs`.
3. Run `python call_run_time_evecs_power.py` to execute the runtime measurements
   of the power iteration baseline. Find the run times in
   `results/time/evecs/*power*.json`.
4. Create the figures with `python plot_time_evecs.py`.
5. Clean up with `bash clean.sh` to remove the results.

## Tolerance experiment

We run the power iteration using different convergence hyperparameters, as well
as ViViT to compute top eigenpairs of the GGN on a mini-batch. We plot the mean
relative accuracy of the resulting eigenvalues over time.

1. Run `python call_run_time_evecs_tol.py` to generate the ViViT runtime results
2. Run `python call_run_time_evecs_power_tol.py` to generate the power iteration
   runtime results.
3. Run the plotting script: `python plot_time_evecs_tol.py`

## Power iteration on the Gram matrix vs. on the GGN

We compare computing top eigenpairs via a power iteration on the GGN versus a
power iteration on the Gram matrix, using the same convergence hyperparameters.

1. Run `python call_run_time_evecs_gram_power_tol.py` to generate the ViViT
   runtime results
2. Run `python call_run_time_evecs_power.py` to generate the power iteration
   runtime results.
3. Run the plotting script: `python plot_time_evecs_gram_power_tol.py`

# Spectral densities

We compute and visualize the full eigenvalue spectrum as histogram for different
approximations of the GGN (exact, MC=1-sampled) and allocated samples (full
mini-batch, fraction of mini-batch). To reproduce the results:

1. (Optional) Extract `results.zip` to use the original data with `unzip
   results.zip`
2. Run `python run_evals.py`. The results are stored under `results/evals`.
3. Plot `python plot_evals.py`. The results are stored under `fig/evals`.
3. Clean up with `bash clean.sh` to remove the results.