import argparse


def run_optimizer(lr, damping):
    """
    A dummy function that runs an optimizer. ``lr`` and ``damping`` represent potential
    hyperparameters.
    """

    print(f"Running dummy optimizer with parameters lr = {lr}, damping = {damping}")

    # Run actual optimizer


if __name__ == "__main__":
    """This function calls the (dummy) function ``run_optimizer``. The two
    hyperparameters ``lr`` and ``damping`` are collected from the command line.

    An example call looks like this:
    ``python ./run_optimizer.py --lr=0.1 --damping=0.01``
    """

    # Create parser, specify two command line arguments
    parser_description = "Collect hyperparameters for running the optimizer."
    parser = argparse.ArgumentParser(description=parser_description)

    parser.add_argument(
        "--lr",
        dest="lr",
        action="store",
        type=float,
        required=True,
        help="The learning rate (float)",
    )

    parser.add_argument(
        "--damping",
        dest="damping",
        action="store",
        type=float,
        required=True,
        help="The damping parameter (float)",
    )

    # Collect arguments and call ``run_optimizer``
    args = parser.parse_args()
    run_optimizer(lr=args.lr, damping=args.damping)
