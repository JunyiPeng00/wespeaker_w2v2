"""Stop request of a training run: SIGUSR1 and SIGTERM.

Both signals only set a flag in the process that receives them. The epoch
loop calls ``StopFlag.poll`` once per step on every process; the flags are
all-reduced (MAX) there, so a signal that reached one process, or reached
the processes at different times, stops all of them after the same step.
The loop then writes a checkpoint and the processes exit normally.

Who sends what: a launcher should send SIGUSR1 to every training process
directly. ``torchrun`` forwards a SIGTERM it receives to the processes and
kills them 30 seconds later, which is shorter than a checkpoint of a large
model takes; SIGTERM is handled for that case too (the checkpoint files are
written atomically, a kill during the save leaves the previous state).

The handlers act only in the process that installed them. A child forked
later (a DataLoader worker) inherits them; there a signal restores the
default action and is delivered again, so workers terminate as usual.
"""
import os
import signal
from types import FrameType
from typing import Dict, Optional, Sequence

import torch
import torch.distributed as dist

STOP_SIGNALS = (signal.SIGUSR1, signal.SIGTERM)


class StopFlag:
    """Stop request shared by the processes of a run.

    Args:
        signals: signals that request a stop.

    Attributes:
        received: name of the first signal this process got (or the reason
            given to ``request``), None if none.
        agreed: True once ``poll`` has seen a request on any process.
    """

    def __init__(self, signals: Sequence[int] = STOP_SIGNALS) -> None:
        self.signals = tuple(signals)
        self.received: Optional[str] = None
        self.agreed = False
        self._owner: Optional[int] = None
        self._previous: Dict[int, object] = {}

    def install(self) -> 'StopFlag':
        """Install the handlers (main thread of the process only)."""
        self._owner = os.getpid()
        for signum in self.signals:
            self._previous[signum] = signal.signal(signum, self._handle)
        return self

    def restore(self) -> None:
        """Put back the handlers that ``install`` replaced."""
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)
        self._previous = {}
        self._owner = None

    def _handle(self, signum: int, frame: Optional[FrameType]) -> None:
        if os.getpid() != self._owner:  # a forked child: default action
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        if self.received is None:
            self.received = signal.Signals(signum).name

    def request(self, reason: str = 'request') -> None:
        """Ask for a stop from code, as a signal would."""
        if self.received is None:
            self.received = reason

    def poll(self, device: torch.device) -> bool:
        """True when any process has a stop request.

        One all-reduce (MAX) in a distributed run; every process must call
        it at the same point of the same step. Once True it stays True.

        Args:
            device: training device of this process (the tensor of the
                all-reduce lives there).
        """
        if self.agreed:
            return True
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor([int(self.received is not None)],
                                dtype=torch.int32, device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            self.agreed = bool(flag.item())
        else:
            self.agreed = self.received is not None
        return self.agreed
