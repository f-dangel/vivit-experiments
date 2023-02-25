This experiment performs a grid search for optimal learning rates (SGD with momentum and Adam) on the resnet32-architecture trained on CIFAR-10.


### Procedure

To reproduce the results, the following steps need to be executed:
1. Perform the gridsearch by calling `python gridsearch_Adam.py` and `gridsearch_SGD.py` (or run `python run_gridsearch.py` to execute both scripts). 

    We use the same search grid as in the [DeepOBS paper](https://arxiv.org/pdf/1903.05499.pdf): A log-equidistant grid from $10^{-5}$ to $10^2$ with $36$ gridpoints. The following hyperparameters were taken from the [original ResNet paper](https://arxiv.org/pdf/1512.03385.pdf), Section 4.2: 
    - mini-batch size $N=128$
    - 64k iterations (on 45k training examples), i.e. approximately 180 training epochs
    - SGD with momentum of 0.9

    Adam uses the default parameters ($\beta_1=0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$).
2. Perform the analysis on the metrics generated in step 1 by calling `python analyse_results.py`. To find the "best" learning rates, we use test accuracy as the performance metric (as in the [DeepOBS paper](https://arxiv.org/pdf/1903.05499.pdf), Section 2.2).


### Results

- With performance metric ``test_accuracies`` and mode ``best`` (the best test accuracy, evaluated once every epoch, over all training epochs), we get the following results:
    ```
    SGD    : lr =     0.06309573444801936, performance = 0.893    
    Adam   : lr =    0.002511886431509582, performance = 0.898
    ```
- If we take a look at the visualization, where the ``best`` test accuaracy is shown over the learning rate, we observe that, at these learning rates, the test accuracy is flat. This indicates that the learning rates are "stable" choices. 
