In this experiment, we monitor the GGN during neural network training. In
particular, we are interested in the effect of different approximations on the
GGN's structural properties. 


### Procedure

To reproduce the results, the following steps need to be executed:

0. **(Optional) use original data:** If you want to skip the computation steps 1
and 2, extract the original data via `unzip results.zip`.

1. **Neural network training:** First, we train different test problems with
both SGD and the Adam optimizer by calling `python train.py`. At certain
checkpoints during the optimization procedure, we store a copy of the network's
parameters in the subfolder `results/checkpoints`. The test problems, the
checkpoint-grid, and the hyperparameters for the optimization procedure are
specified in `config.py`.  

2. **Compute full-batch quantities:** Next, we compute and store the top-C
eigenspaces of the full-batch GGN and full-batch Hessian for each test problem
and checkpoint by calling `run_eval.py` (which internally calls `eval.py`). The
results are stored in the subfolder `results/eval`. 

3. **Plotting:** There are multiple plotting scripts implementing the
computation and visualization of different quantities. Each of those scripts
creates a subfolder in `results/plots` that contains the plots as `PDF`-files
and (in most cases) `.json`-files that contain the computed plotting data. By
default, the script uses the `.json`-file and only recomputes it if the
respective file is not found (or if the user explicitly requests a
re-computation). If the file size allows, we provide the `.json`-files in the
respective subfolders. 

    3.1 `plot_loss_accuracy.py`: Plot the training metrics training/test
    loss/accuracy for all  training runs executed in step 1 by calling `python
    plot_loss_accuracy.py`. The results are stored in the subfolder
    `results/plots/loss_accuracy`. 

    3.2 `plot_eigspace_ggn_vs_hessian.py`: Plot the overlap of the top-C
    eigenspaces of the full-batch Hessian and full-batch GGN (these full-batch
    eigenspaces are computed in step 2). The results are stored in the subfolder
    `results/plots/eigspace_ggn_vs_hessian`. 

    3.3 `plot_eigspace_vivit_vs_fb.py`: Compute and plot the overlap between the
    GGN's full-batch eigenspace (as ground truth) and different approximations
    (e.g. a mini-batch eigenspace). The results are stored in the subfolder
    `results/plots/eigspace_vivit_vs_fb`. 
    
    3.4 `plot_eigspace_vivit_vs_mb.py`: Compute and plot the overlap between the
    GGN's mini-batch eigenspace (as ground truth) and different further
    approximations (e.g. an MC-approximation). The results are stored in the
    subfolder `results/plots/eigspace_vivit_vs_mb`. 

    3.5 `plot_eigvals_vivit_vs_fb.py`: Compute the directional curvature of the
    quadratic model based on a GGN approximation and compare it to the
    directional curvature of the full-batch model along the same directions.
    Plot the relative errors. The results are stored in the subfolder
    `results/plots/eigvals_vivit_vs_fb`.

    3.6 `plot_gammas_lambdas.py`: Compute and plot SNRs of the per-sample
    directional derivatives. The results are stored in the subfolder
    `results/plots/gammas_lambdas`.

    **Note:** Most of these scripts above use command line arguments to specify
    the test problem for which the visualization is created - running them
    without this specification will cause an error. The scripts with the prefix
    `run_` automatically call the above plotting scripts for all configurations. 


### Notes

We also implemented two test scripts `test_eigspace_pi_vivit.py` and
`test_gammas_lambdas_pi.py` which test functionality specifically implemented
for this experiment. 
