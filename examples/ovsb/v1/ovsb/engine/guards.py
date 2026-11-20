"""Skipping of non-finite training steps, identically on every process.

bfloat16 autocast runs without a gradient scaler, so nothing skips a step
whose loss or gradient is not finite. The training step checks both; the
decision is all-reduced so that every process of a distributed run takes
the same branch (a process that skipped alone would miss the gradient
all-reduce of the others and hang).
"""
import logging
from typing import Optional

import torch
import torch.distributed as dist

# consecutive non-finite steps tolerated before the run aborts
NONFINITE_ABORT = 20


def all_ranks_finite(value: torch.Tensor) -> bool:
    """True when the scalar ``value`` is finite on every process.

    One all-reduce (MIN) in a distributed run and one host sync; every
    process must call it at the same point of the step.
    """
    finite = torch.isfinite(value.detach()).to(torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


class NonFiniteGuard:
    """Bookkeeping of skipped steps within one epoch.

    ``ok()`` after a step that ran resets the count of consecutive skips.
    ``skip(i, epoch_iter)`` after a skipped step says how the epoch loop
    continues and raises once ``limit`` steps in a row were skipped: the
    weights are then most likely not finite and the run would only spin to
    its time limit. Fed with all-reduced decisions, every process raises
    at the same step.
    """

    def __init__(self, limit: int = NONFINITE_ABORT) -> None:
        self.limit = int(limit)
        self.run = 0    # consecutive skipped steps
        self.total = 0  # skipped steps since construction

    def ok(self) -> None:
        """A step ran: reset the consecutive count."""
        self.run = 0

    def skip(self, i: int, epoch_iter: int, what: str = 'loss',
             epoch: Optional[int] = None,
             logger: Optional[logging.Logger] = None) -> str:
        """Count a skipped step.

        Args:
            i: zero-based index of the step in the epoch.
            epoch_iter: steps per epoch.
            what: what was not finite ('loss' or 'grad norm').
            epoch: epoch number for the messages.
            logger: logger for the warning (None: silent).

        Returns:
            'break' when the step was the last of the epoch (the loop must
            still end the epoch), else 'continue'.

        Raises:
            RuntimeError: ``limit`` consecutive steps were skipped.
        """
        self.run += 1
        self.total += 1
        if logger is not None:
            logger.warning('non-finite %s at epoch %s batch %d, step skipped '
                           'on all ranks (%d consecutive, %d in this epoch)'
                           % (what, epoch, i + 1, self.run, self.total))
        if self.run >= self.limit:
            raise RuntimeError(
                '%d consecutive non-finite %s steps (epoch %s, batch %d): '
                'the weights are most likely NaN; aborting instead of '
                'spinning to the time limit' % (self.run, what, epoch, i + 1))
        return 'break' if (i + 1) >= epoch_iter else 'continue'
