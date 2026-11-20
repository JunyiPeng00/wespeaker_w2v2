"""Epoch loop and run loop.

``run_epoch`` drives ``ovsb.engine.step.train_step`` over the batches of
one epoch; ``fit`` runs the epochs of a process, saves and stops.

Data seam. The loop gets its batches from a ``TrainData`` object, the only
thing it knows about the data layer:

    data.loader(epoch, stream, generator) -> iterable of collated batches

called once per epoch on every process, after the generators were set for
the epoch (``ovsb.engine.seeding.begin_epoch``). ``stream`` is 0 except for
the pass that continues an interrupted epoch; ``generator`` is what the
DataLoader must draw its worker seeds from (None: the torch CPU generator).
The iterable must give at least the remaining steps of the epoch. An
optional ``data.stat_lines()`` returns lines for the log at the end of an
epoch (sampler or codec-helper statistics of this process).

One step (every process, same step index ``i``, zero-based in the epoch):

1. ``train_step`` with the global index ``(epoch - 1) * epoch_iter + i``;
2. a non-finite loss is counted only; any other step enters the epoch
   meter and the per-arm log, and a non-finite gradient norm is counted;
3. every ``run.log_batch_interval`` steps (unless the step was skipped):
   the means since the start of the epoch are all-reduced and the process
   of rank 0 prints the loss lines, the per-arm lines, the failed codec
   calls of the interval if there were any (``batch['chain_fail']``, summed
   over all processes) and a table row;
4. the stop flag is all-reduced; a stop request ends the epoch after this
   step on every process.

After the last step of the epoch, or of a stopped epoch, one more table
row is printed with the means so far.

Stopping and saving (``fit``). At the end of a complete epoch the state is
saved when ``run.save_epoch_interval`` or ``run.num_avg`` ask for it and
always when the process stops there. The process stops at an epoch
boundary when a stop was requested during the last step or during the
save, or when it has run ``epochs_this_run`` epochs. A stop request inside
an epoch saves the partial state (steps done, optimiser, running centres,
stream id) and ends the process; the next process continues that epoch at
the same step with ``stream + 1`` (``ovsb.engine.seeding``).

After a resume inside an epoch the means of the log cover the steps since
the resume, and the per-arm lines of the interrupted interval are lost.
"""
import logging
import os
import time
from dataclasses import dataclass
from typing import (Any, Dict, Iterable, List, Mapping, Optional,
                    Protocol)

import torch

from ovsb.config import schema
from ovsb.config.schema import CODECS, Config
from ovsb.engine import checkpoint, seeding
from ovsb.engine.arm_log import ArmLogger
from ovsb.engine.checkpoint import ResumePoint
from ovsb.engine.console import (EpochMeter, Totals, all_reduce_sum,
                                 bottom_line, detail_lines, row_line)
from ovsb.engine.guards import NonFiniteGuard
from ovsb.engine.optim import learning_rates
from ovsb.engine.signals import StopFlag
from ovsb.engine.step import SKIP_GRAD, SKIP_LOSS, train_step, unwrap
from ovsb.losses.bundle import LossBundle

Batch = Mapping[str, Any]
FINISHED, SIGNAL, SEGMENT = 'finished', 'signal', 'segment'


class TrainData(Protocol):
    """The data seam of the loop (module docstring)."""

    def loader(self, epoch: int, stream: int,
               generator: Optional[torch.Generator]) -> Iterable[Batch]:
        """Batches of ``(epoch, stream)`` for this process."""


class ChainFailMeter:
    """Failed codec calls of the batches, per codec, over all processes.

    The data layer marks a row whose codec call failed (the row is then
    trained without the codec) with the codec's index in
    ``batch['chain_fail']``, -1 otherwise. Batches without the field count
    nothing.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.counts = torch.zeros(len(CODECS), dtype=torch.float64)
        self.epoch_total = 0

    def add(self, batch: Batch) -> None:
        """Count the failed rows of one batch of this process."""
        if 'chain_fail' not in batch:
            return
        code = batch['chain_fail'].long().cpu()
        self.counts += torch.bincount(
            code[code >= 0], minlength=len(CODECS)).to(torch.float64)

    def flush(self) -> Dict[str, int]:
        """Counts since the last flush summed over all processes, by codec
        name (codecs without a failure left out); every process must call
        it at the same step."""
        total = all_reduce_sum(self.counts, self.device)
        self.counts = torch.zeros_like(self.counts)
        found = {name: int(n) for name, n in zip(CODECS, total.tolist()) if n}
        self.epoch_total += sum(found.values())
        return found


@dataclass
class EpochResult:
    """Outcome of ``run_epoch``."""
    steps_done: int   # steps of the epoch done so far, start_step included
    stopped: bool     # a stop request ended the loop
    totals: Totals    # means over the steps this call ran, all processes
    skipped: int      # non-finite steps this call skipped
    chain_fails: int  # failed codec calls in those steps, all processes


@dataclass
class FitResult:
    """Outcome of ``fit``.

    ``reason`` is 'finished' (all epochs done), 'signal' (stop request) or
    'segment' (``epochs_this_run`` reached). The run continues at
    ``epoch`` after ``step`` steps; for 'finished' these are the last
    epoch and its step count.
    """
    reason: str
    epoch: int
    step: int

    @property
    def finished(self) -> bool:
        return self.reason == FINISHED


def _time_mask(model: torch.nn.Module) -> Optional[List[int]]:
    return getattr(unwrap(model).frontend, 'time_mask_stats', None)


def run_epoch(loader: Iterable[Batch], model: torch.nn.Module,
              bundle: LossBundle, cfg: Config, device: torch.device,
              optimizer: torch.optim.Optimizer, scheduler: Any,
              margin_scheduler: Any, epoch: int,
              logger: Optional[logging.Logger] = None,
              stop: Optional[StopFlag] = None, start_step: int = 0,
              arm_log: Optional[ArmLogger] = None,
              amp: Optional[bool] = None) -> EpochResult:
    """Train the steps ``start_step .. epoch_iter - 1`` of one epoch.

    Args:
        loader: batches of the epoch for this process.
        model: the model, optionally in a DistributedDataParallel wrapper.
        bundle: the objective (holds this process's running centre).
        cfg: the recipe config; steps per epoch are
            ``ovsb.config.schema.epoch_iter(cfg)``.
        device: training device of this process.
        optimizer, scheduler, margin_scheduler: as for ``train_step``.
        epoch: one-based epoch number.
        logger: the log of the run on the process that prints (rank 0),
            None on the others.
        stop: stop flag polled after every step (None: never stops).
        start_step: steps of the epoch already done by an earlier process.
        arm_log: per-arm log, or None.
        amp: bfloat16 autocast, as for ``train_step``.

    Raises:
        RuntimeError: the loader ended before the epoch did, or too many
            consecutive steps were not finite.
    """
    epoch_iter = schema.epoch_iter(cfg)
    if not 0 <= start_step < epoch_iter:
        raise ValueError('start_step %d outside the epoch of %d steps' %
                         (start_step, epoch_iter))
    interval = cfg.run.log_batch_interval
    model.train()
    meter, guard = EpochMeter(device), NonFiniteGuard()
    fails = ChainFailMeter(device)
    done, stopped = start_step, False

    def log_fails(batch_no: int) -> None:
        found = fails.flush()
        if found and logger is not None:
            logger.info('codec chain: %d failed call(s) since the last '
                        'report, rows kept without the codec %s at epoch %d '
                        'batch %d' % (sum(found.values()), found, epoch,
                                      batch_no))

    for i, batch in enumerate(loader, start_step):
        cur_iter = (epoch - 1) * epoch_iter + i
        fails.add(batch)
        res = train_step(model, bundle, batch, cfg, device, optimizer,
                         scheduler, margin_scheduler, cur_iter, amp)
        if res.skipped == SKIP_LOSS:
            guard.skip(i, epoch_iter, SKIP_LOSS, epoch, logger)
        else:
            meter.add(res.loss, res.logits, res.labels, res.parts, res.stats)
            if arm_log is not None:
                arm_log.add(batch, res.logits, res.labels)
            if res.skipped == SKIP_GRAD:
                guard.skip(i, epoch_iter, SKIP_GRAD, epoch, logger)
            else:
                guard.ok()
        if res.skipped is None and (i + 1) % interval == 0:
            totals = meter.reduce()
            if logger is not None:
                for line in detail_lines(totals, epoch, i + 1,
                                         meter.last_stats, bundle.center,
                                         _time_mask(model)):
                    logger.info(line)
            if arm_log is not None:
                arm_log.flush(logger, epoch, i + 1)
            log_fails(i + 1)
            if logger is not None:
                logger.info(row_line(
                    epoch, i + 1, *learning_rates(optimizer),
                    margin_scheduler.get_margin(), totals.loss, totals.acc))
        done = i + 1
        stopped = stop is not None and stop.poll(device)
        if stopped or done == epoch_iter:
            break
    if done < epoch_iter and not stopped:
        raise RuntimeError(
            'epoch %d: the data loader ended after step %d of %d' %
            (epoch, done, epoch_iter))
    totals = meter.reduce()
    log_fails(done)
    if logger is not None:
        logger.info(row_line(epoch, done, *learning_rates(optimizer),
                             margin_scheduler.get_margin(), totals.loss,
                             totals.acc))
    return EpochResult(done, stopped, totals, guard.total, fails.epoch_total)


def _epoch_summary(logger: logging.Logger, epoch: int, res: EpochResult,
                   seconds: float, bundle: LossBundle, model: torch.nn.Module,
                   data: TrainData) -> None:
    logger.info('epoch %d: %d steps done in %.0f s, %d non-finite steps '
                'skipped, %d failed codec calls' % (
                    epoch, res.steps_done, seconds, res.skipped,
                    res.chain_fails))
    if bundle.center is not None:
        logger.info(bundle.center.describe())
    mask = _time_mask(model)
    if mask and mask[1]:
        logger.info('time_mask: %.4f of valid frames masked (%d / %d)' %
                    (mask[0] / mask[1], mask[0], mask[1]))
    for line in getattr(data, 'stat_lines', list)():
        logger.info(line)


def link_final_model(model_dir: str, num_epochs: int) -> str:
    """Point ``final_model.pt`` at the model of the last epoch."""
    link = os.path.join(model_dir, 'final_model.pt')
    if os.path.lexists(link):
        os.remove(link)
    os.symlink('model_%d.pt' % num_epochs, link)
    return link


def fit(model: torch.nn.Module, bundle: LossBundle, cfg: Config,
        device: torch.device, optimizer: torch.optim.Optimizer,
        scheduler: Any, margin_scheduler: Any, data: TrainData,
        start: ResumePoint, rank: int = 0, world: int = 1,
        logger: Optional[logging.Logger] = None,
        stop: Optional[StopFlag] = None,
        epochs_this_run: Optional[int] = None,
        arm_log: Optional[ArmLogger] = None,
        amp: Optional[bool] = None) -> FitResult:
    """Run the epochs from ``start`` on this process (module docstring).

    Every process of a distributed run calls it with the same arguments
    apart from ``rank``, ``logger`` and its own objects; the process of
    rank 0 writes the files.

    Args:
        model, bundle, cfg, device, optimizer, scheduler, margin_scheduler,
            arm_log, amp: as for ``run_epoch``; the training state of
            ``start`` is already restored.
        data: the data seam.
        start: where the run continues (``find_resume_point``).
        rank, world: index of this process and number of processes.
        logger: the log of the run on rank 0, None on the others.
        stop: stop flag (signals); None: the run cannot be interrupted.
        epochs_this_run: stop after this many epochs were completed by
            this process (None: run to the last epoch). The epoch that
            continues an interrupted one counts.
    """
    if epochs_this_run is not None and epochs_this_run < 1:
        raise ValueError('epochs_this_run must be >= 1, got %r' %
                         (epochs_this_run, ))
    run, model_dir = cfg.run, os.path.join(cfg.run.exp_dir, 'models')
    epoch_iter = schema.epoch_iter(cfg)
    dim = cfg.model.backend.embed_dim
    writer = rank == 0
    completed = 0
    for epoch in range(start.start_epoch, run.num_epochs + 1):
        first = epoch == start.start_epoch
        stream = start.start_stream if first else 0
        generator = seeding.begin_epoch(run.seed, rank, epoch, stream)
        began = time.time()
        res = run_epoch(data.loader(epoch, stream, generator), model, bundle,
                        cfg, device, optimizer, scheduler, margin_scheduler,
                        epoch, logger, stop, start.start_step if first else 0,
                        arm_log, amp)
        if logger is not None:
            _epoch_summary(logger, epoch, res, time.time() - began, bundle,
                           model, data)
        if res.steps_done < epoch_iter:  # stopped inside the epoch
            states = checkpoint.gather_center_states(bundle.center, dim,
                                                     device)
            if writer:
                checkpoint.save_partial(
                    model_dir, epoch, res.steps_done, stream, model,
                    optimizer, scheduler, margin_scheduler, states, world,
                    epoch_iter)
            if logger is not None:
                logger.info('stop request: epoch %d interrupted after step '
                            '%d of %d, partial state saved' %
                            (epoch, res.steps_done, epoch_iter))
            return FitResult(SIGNAL, epoch, res.steps_done)

        completed += 1
        last = epoch == run.num_epochs
        saved = False

        def save() -> None:
            nonlocal saved
            if saved:
                return
            states = checkpoint.gather_center_states(bundle.center, dim,
                                                     device)
            if writer:
                checkpoint.save_epoch(
                    model_dir, epoch, model, optimizer, scheduler,
                    margin_scheduler, states, world, stream,
                    cfg.engine.save_optimizer_state)
            saved = True

        reason = None
        if res.stopped:
            reason = SIGNAL
        elif epochs_this_run is not None and completed >= epochs_this_run:
            reason = SEGMENT
        if epoch % run.save_epoch_interval == 0 or \
                epoch > run.num_epochs - run.num_avg or reason is not None:
            save()
        if last:
            break
        if reason is None and stop is not None and stop.poll(device):
            reason = SIGNAL  # requested while the epoch was being saved
            save()
        if reason is not None:
            if logger is not None:
                logger.info('%s: stopping after epoch %d of %d' % (
                    'stop request' if reason == SIGNAL else
                    '%d epoch(s) of this process done' % completed, epoch,
                    run.num_epochs))
            return FitResult(reason, epoch + 1, 0)
    if writer:
        link_final_model(model_dir, run.num_epochs)
    if logger is not None:
        logger.info(bottom_line())
    return FitResult(FINISHED, run.num_epochs, epoch_iter)
