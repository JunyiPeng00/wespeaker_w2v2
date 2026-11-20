"""Training log: the progress table and the loss lines of an interval.

Every process accumulates sums since the start of the epoch in an
``EpochMeter``. At a log interval all processes call ``reduce()`` at the
same step (one all-reduce); the process with rank 0 formats the lines.
Loss, accuracy and the loss terms are therefore means over the batches of
all processes, from the start of the epoch to the logged step.

Table columns: ``Lr`` is the learning rate of the base param group
(pooling backend and classifier head), ``LrFE`` that of the encoder.
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import tableprint as tp
import torch
import torch.distributed as dist

from ovsb.losses.bundle import Components, Stats

HEADER = ('Epoch', 'Batch', 'Lr', 'LrFE', 'Margin', 'Loss', 'Acc')
_WIDTH, _STYLE = 10, 'grid'
_FIELDS = ('steps', 'loss', 'rows', 'correct', 'cons', 'cons_n', 'pauc',
           'pauc_n')


def all_reduce_sum(values: torch.Tensor,
                   device: torch.device) -> torch.Tensor:
    """Sum of ``values`` over all processes, as a new CPU tensor.

    The tensor travels through ``device`` (the process's training device)
    for the all-reduce. Not distributed: a copy of ``values``.
    """
    total = values.detach().clone()  # the all-reduce is in place
    if dist.is_available() and dist.is_initialized():
        total = total.to(device)
        dist.all_reduce(total)
    return total.cpu()


def header_lines() -> List[str]:
    """The lines of the table header."""
    return tp.header(list(HEADER), width=_WIDTH, style=_STYLE).split('\n')


def bottom_line() -> str:
    """The closing line of the table."""
    return tp.bottom(len(HEADER), width=_WIDTH, style=_STYLE)


def row_line(epoch: int, batch: int, lr: float, lr_frontend: float,
             margin: float, loss: float, acc: float) -> str:
    """One table row; ``batch`` is the one-based step within the epoch."""
    return tp.row((epoch, batch, lr, lr_frontend, margin, loss, acc),
                  width=_WIDTH, style=_STYLE)


@dataclass
class Totals:
    """Means over all processes since the start of the epoch.

    ``loss`` is the mean objective per step, ``acc`` the top-1 accuracy of
    the margin logits in percent, ``cons`` / ``pauc`` the mean unweighted
    consistency / pair-AUC term over the ``cons_n`` / ``pauc_n`` batches
    that had it (NaN when there was none).
    """
    steps: int
    loss: float
    acc: float
    cons: float
    cons_n: int
    pauc: float
    pauc_n: int


class EpochMeter:
    """Per-process sums of one epoch; ``reduce`` gives the global means.

    Args:
        device: training device of this process, used for the all-reduce.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.reset()

    def reset(self) -> None:
        """Start a new epoch."""
        self.sums = torch.zeros(len(_FIELDS), dtype=torch.float64)
        self.last_stats: Stats = {}

    def add(self, loss: torch.Tensor, logits: torch.Tensor,
            labels: torch.Tensor, parts: Components, stats: Stats) -> None:
        """Add one step whose loss was finite.

        Args:
            loss: scalar objective of the step.
            logits: (B, C) margin logits.
            labels: (B,) class labels on the device of ``logits``.
            parts: detached unweighted terms; None where a term was absent.
            stats: pair statistics of the batch (kept for the log line).
        """
        correct = (logits.detach().argmax(1) == labels).sum()
        cons, pauc = parts.get('cons'), parts.get('pauc')
        self.sums += torch.tensor([
            1.0, float(loss), float(labels.shape[0]), float(correct),
            0.0 if cons is None else float(cons), float(cons is not None),
            0.0 if pauc is None else float(pauc), float(pauc is not None)],
            dtype=torch.float64)
        if pauc is not None:
            self.last_stats = dict(stats)

    def reduce(self) -> Totals:
        """Global means; every process must call it at the same step."""
        total = dict(zip(_FIELDS,
                         all_reduce_sum(self.sums, self.device).tolist()))
        nan = float('nan')

        def mean(name: str, count: str) -> float:
            return total[name] / total[count] if total[count] else nan

        return Totals(steps=int(total['steps']), loss=mean('loss', 'steps'),
                      acc=100.0 * mean('correct', 'rows'),
                      cons=mean('cons', 'cons_n'),
                      cons_n=int(total['cons_n']),
                      pauc=mean('pauc', 'pauc_n'),
                      pauc_n=int(total['pauc_n']))


def detail_lines(totals: Totals, epoch: int, batch: int,
                 last_stats: Optional[Stats] = None,
                 center: Optional[Any] = None,
                 time_mask: Optional[Sequence[int]] = None) -> List[str]:
    """The loss lines printed above a table row.

    Args:
        totals: reduced means of the epoch so far.
        epoch, batch: position of the log entry (``batch`` one-based).
        last_stats: pair statistics of this process's last batch with a
            pair-AUC term (``EpochMeter.last_stats``).
        center: the running centre of this process (``describe()``).
        time_mask: [masked, valid] frames of this process's encoder.
    """
    lines = []
    if totals.cons_n > 0:
        lines.append('consistency loss %.4f (%d pair batches) at epoch %d '
                     'batch %d' % (totals.cons, totals.cons_n, epoch, batch))
    if totals.pauc_n > 0:
        last = last_stats or {}
        lines.append(
            'pair_auc loss %.4f (%d batches; last batch %d pos %d neg, cos '
            'pos %.3f neg %.3f) at epoch %d batch %d' % (
                totals.pauc, totals.pauc_n, last.get('pauc_pos', 0),
                last.get('pauc_neg', 0), last.get('pauc_cos_pos', 0.0),
                last.get('pauc_cos_neg', 0.0), epoch, batch))
        if center is not None:
            lines.append(center.describe())
        if time_mask and time_mask[1]:
            lines.append('time_mask: %.4f of valid frames masked (%d / %d)' %
                         (time_mask[0] / time_mask[1], time_mask[0],
                          time_mask[1]))
    return lines
