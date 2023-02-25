"""Train neural network from the DeepOBS library and save model at certain checkpoints
during training."""

from config import CONFIGURATIONS, run_config

if __name__ == "__main__":
    for config in CONFIGURATIONS:
        run_config(config)
