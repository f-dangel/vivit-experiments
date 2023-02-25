"""Contains DeepOBS runners that integrate BackPACK."""

import os
import warnings
from typing import Any, Callable, Dict, List, Optional

import dill
import numpy as np
import torch
from backobs import extend
from backpack import disable
from deepobs.pytorch.runners import PTRunner
from deepobs.pytorch.testproblems import TestProblem
from torch import Tensor, isfinite
from torch.optim import Optimizer

from exp.utils.path import write_to_json
from vivit.optim.base import BackpackOptimizer


class BackpackRunner(PTRunner):
    """DeepOBS runner with BackPACK-compatible problem and support for closures.

    Note:
        - No support for regularization: Problems can only be extended if their ℓ₂
          regularization is turned off.
        - Two different closures can be created. One is incompatible with the
          PyTorch interface described in (https://pytorch.org/docs/stable/optim.html)
          but allows more freedom when implementing an optimizer that uses BackPACK
          extensions.

          For optimizers that do not inherit from ``BackpackOptimizer``, the closure
          will automatically be converted to a compatible one. But if you want to
          build your own optimizer based on BackPACK+closures, read the documentation
          of ``create_closure``.
    """

    @staticmethod
    def create_closure(
        tproblem: TestProblem,
        internals: bool = True,
        savefield: str = "_internals",
    ) -> Callable[[], Tensor]:
        """Create a closure that evaluates the loss on the same mini-batch.

        Note:
            The created closure does not perform the steps described in
            (https://pytorch.org/docs/stable/optim.html). See below for details.

        Args:
            tproblem: DeepOBS problem.
            internals: Whether to attach internal tensors from the forward pass
                as an attribute of the return value. Default: ``True``.
            savefield: Attribute name under which internal tensors will
                be saved in the return value. Default: ``'_internals'``.

        Returns:
            Function that evaluates the loss on a fixed mini-batch.
        """
        forward_func = tproblem.get_batch_loss_and_accuracy_func(
            add_regularization_if_available=False, internals=internals
        )

        def closure() -> Tensor:
            """Evaluates loss and model output on a mini-batch.

            Note:
                This closure is not compatible with the interface described in the
                PyTorch documentation (https://pytorch.org/docs/stable/optim.html):
                - It does not zero the gradients
                - Its return values have a different signature

            Returns:
                Mini-batch loss and a dictionary with internal tensors (model output,
                inputs, labels, accuracy).
            """
            if internals:
                loss, accuracy, info = forward_func()
            else:
                (loss, accuracy), info = forward_func(), {}

            info["accuracy"] = accuracy
            setattr(loss, savefield, info)

            return loss

        return closure

    @staticmethod
    def make_torch_compatible(
        closure: Callable[[], Tensor], opt: Optimizer
    ) -> Callable[[], Tensor]:
        """Take closure created with ``create_closure`` and make it PyTorch-compatible.

        Args:
            closure: Closure created with ``create_closure``.
            opt: Optimizer.

        Returns:
            PyTorch-compatible closure. Attaches internal tensors from the forward pass
            to the return value's ``._internals`` attribute.
        """

        def compatible_closure() -> Tensor:
            """PyTorch-compatible closure for the optimizer interface.

            Note:
                From the docs (https://pytorch.org/docs/stable/optim.html), closures
                - clear the gradients
                - compute the loss
                - compute the gradients
                - return the loss

            Returns:
                Mini-batch loss. Has internal tensors from the forward pass attached to
                its ``._internals`` attribute.
            """
            opt.zero_grad()
            loss = closure()
            loss.backward()

            return loss

        return compatible_closure

    def training(
        self,
        tproblem: TestProblem,
        hyperparams: Dict[str, Any],
        num_epochs: int,
        print_train_iter: int,
        train_log_interval: int,
        tb_log: Optional[bool] = None,
        tb_log_dir: Optional[str] = None,
        debug: Optional[bool] = False,
        **train_params,
    ) -> Dict[str, List[Any]]:
        """Make problem BackPACK-compatible and train using a closure.

        Args:
            tproblem: DeepOBS problem.
            hyperparams: Dictionary containing the optimizer hyperparameters.
            num_epochs: Number of training epochs.
            print_train_iter: Interval for printing the training progress.
            train_log_interval: Mini-batch interval for logging.
            tb_log: Whether to use tensorboard logging or not (currently not supported).
            tb_log_dir: The path where to save tensorboard events.
            **training_params: Additional training parameters that can be implemented by
                subclasses.

        Returns:
            Dictionary with lists of training metrics as items.
        """
        tproblem = extend(tproblem)
        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []

        minibatch_train_losses = []

        assert not tb_log, "tb_log is not supported."

        global_step = 0

        for epoch_count in range(num_epochs + 1):
            # Evaluation at beginning of epoch
            print("********************************")
            print(f"Evaluating after {epoch_count} of {num_epochs} epochs...")

            with disable():
                loss_, acc_ = self.evaluate(tproblem, phase="TRAIN")
                train_losses.append(loss_)
                train_accuracies.append(acc_)

                loss_, acc_ = self.evaluate(tproblem, phase="VALID")
                valid_losses.append(loss_)
                valid_accuracies.append(acc_)

                loss_, acc_ = self.evaluate(tproblem, phase="TEST")
                test_losses.append(loss_)
                test_accuracies.append(acc_)

            print("********************************")

            # Exit train loop after last evaluation round
            if epoch_count == num_epochs:
                break

            # Training
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    closure = self.create_closure(tproblem)

                    # required to run with built-in PyTorch optimizers like SGD
                    if not isinstance(opt, BackpackOptimizer):
                        closure = self.make_torch_compatible(closure, opt)

                    batch_loss = opt.step(closure)

                    if batch_count % train_log_interval == 0:
                        minibatch_train_losses.append(batch_loss.item())
                        if print_train_iter:
                            print(
                                f"Epoch {epoch_count}, step {batch_count},"
                                + f" loss {batch_loss}"
                            )

                    batch_count += 1
                    global_step += 1

                except StopIteration:
                    break

            if not isfinite(batch_loss):
                (
                    train_losses,
                    valid_losses,
                    test_losses,
                    train_accuracies,
                    valid_accuracies,
                    test_accuracies,
                    minibatch_train_losses,
                ) = self._abort_routine(
                    epoch_count,
                    num_epochs,
                    train_losses,
                    valid_losses,
                    test_losses,
                    train_accuracies,
                    valid_accuracies,
                    test_accuracies,
                    minibatch_train_losses,
                )
                break
            else:
                continue

        output = {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            "valid_accuracies": valid_accuracies,
            "test_accuracies": test_accuracies,
        }

        return output


class CheckpointRunner(PTRunner):
    """Save model and loss function at checkpoints during training."""

    def __init__(self, optimizer_class, hyperparameter_names, checkpoint_dir):
        super().__init__(optimizer_class, hyperparameter_names)
        self.checkpoint_dir = checkpoint_dir

    @staticmethod
    def get_checkpoints_savedir(problem_cls, optimizer_cls, checkpoints_output):
        """Return sub-directory where checkpoint data is saved."""
        savedir = os.path.join(
            checkpoints_output, f"{problem_cls.__name__}", f"{optimizer_cls.__name__}"
        )
        os.makedirs(savedir, exist_ok=True)

        return savedir

    @staticmethod
    def get_checkpoint_savepath(
        checkpoint, optimizer_cls, problem_cls, checkpoints_output, extension=".pt"
    ):
        """Return the savepath for a checkpoint."""
        epoch_count, batch_count = checkpoint
        savedir = CheckpointRunner.get_checkpoints_savedir(
            problem_cls, optimizer_cls, checkpoints_output
        )
        return os.path.join(
            savedir, f"epoch_{epoch_count:05d}_batch_{batch_count:05d}{extension}"
        )

    def save_checkpoint_data(self, data, checkpoint, optimizer_cls, problem_cls):
        """Save data at checkpoint."""
        savepath = self.get_checkpoint_savepath(
            checkpoint, optimizer_cls, problem_cls, self.checkpoint_dir
        )

        print(f"Saving to {savepath}")
        # DeepOBS models have pre-forward hooks that don't serialize with pickle.
        # See https://github.com/pytorch/pytorch/issues/1148
        torch.save(data, savepath, pickle_module=dill)

    @staticmethod
    def get_summary_savepath(
        problem_cls, optimizer_cls, checkpoint_dir, extension=".json"
    ):
        """Return save path for training summary."""
        savedir = CheckpointRunner.get_checkpoints_savedir(
            problem_cls, optimizer_cls, checkpoint_dir
        )
        return os.path.join(savedir, f"metrics{extension}")

    def training(  # noqa: C901
        self,
        tproblem,
        hyperparams,
        num_epochs,
        print_train_iter,
        train_log_interval,
        tb_log,
        tb_log_dir,
    ):
        """Template training function from ``StandardRunner`` + checkpointing.

        Modified parts are marked by ``CUSTOM``.
        """

        # CUSTOM: Verify all checkpoints will be hit
        self._all_checkpoints_hit(num_epochs)

        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []

        minibatch_train_losses = []

        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter

                summary_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError as e:
                warnings.warn(
                    "Not possible to use tensorboard for pytorch. Reason: " + e.msg,
                    RuntimeWarning,
                )
                tb_log = False
        global_step = 0

        for epoch_count in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            self.evaluate_all(
                epoch_count,
                num_epochs,
                tproblem,
                train_losses,
                valid_losses,
                test_losses,
                train_accuracies,
                valid_accuracies,
                test_accuracies,
            )

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                break

            ### Training ### # noqa: E266

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    opt.zero_grad()

                    # CUSTOM: Save models at checkpoint
                    if self.is_checkpoint(epoch_count, batch_count):
                        self.checkpoint(epoch_count, batch_count, opt, tproblem)

                    batch_loss, _ = tproblem.get_batch_loss_and_accuracy()
                    batch_loss.backward()
                    opt.step()

                    if batch_count % train_log_interval == 0:
                        minibatch_train_losses.append(batch_loss.item())
                        if print_train_iter:
                            print(
                                "Epoch {0:d}, step {1:d}: loss {2:g}".format(
                                    epoch_count, batch_count, batch_loss
                                )
                            )
                        if tb_log:
                            summary_writer.add_scalar(
                                "loss", batch_loss.item(), global_step
                            )

                    batch_count += 1
                    global_step += 1

                except StopIteration:
                    break

            if not np.isfinite(batch_loss.item()):
                self._abort_routine(
                    epoch_count,
                    num_epochs,
                    train_losses,
                    valid_losses,
                    test_losses,
                    train_accuracies,
                    valid_accuracies,
                    test_accuracies,
                    minibatch_train_losses,
                )
                break
            else:
                continue

        if tb_log:
            summary_writer.close()
        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            "valid_accuracies": valid_accuracies,
            "test_accuracies": test_accuracies,
        }

        # CUSTOM: Write metrics to a custom file
        self.summary(output, tproblem, opt)

        return output

    # IO helpers

    def set_checkpoints(self, points):
        """Set points where to save the model during training."""
        assert all(len(point) == 2 for point in points), "Checkpoints must be 2-tuples"
        self._checkpoints = [tuple(point) for point in points]

    def get_checkpoints(self):
        """Return checkpoints."""
        try:
            return self._checkpoints
        except AttributeError as e:
            raise Exception("Did you use 'set_checkpoints'?") from e

    def is_checkpoint(self, epoch_count, batch_count):
        """Return whether an iteration is a check point."""
        return (epoch_count, batch_count) in self.get_checkpoints()

    def _all_checkpoints_hit(self, num_epochs):
        """Raise exception if checkpoints won't be hit."""
        checkpoint_epochs = [point[0] for point in self._checkpoints]
        assert all(
            epoch < num_epochs for epoch in checkpoint_epochs
        ), "Some checkpoints won't be reached"

    def checkpoint(self, epoch_count, batch_count, opt, tproblem):
        """Save model and loss function at a checkpoint."""
        data = {
            "model": tproblem.net,
            "loss_func": tproblem.loss_function(reduction="mean"),
        }
        self.save_checkpoint_data(
            data, (epoch_count, batch_count), opt.__class__, tproblem.__class__
        )

    def summary(self, output, tproblem, opt):
        """Save summary."""
        savepath = CheckpointRunner.get_summary_savepath(
            tproblem.__class__, opt.__class__, self.checkpoint_dir
        )
        write_to_json(savepath, output)
