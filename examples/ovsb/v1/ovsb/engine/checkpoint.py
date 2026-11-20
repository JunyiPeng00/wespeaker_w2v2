"""Checkpoints of a training run: initial weights, epoch end, interruption.

Files in ``<exp_dir>/models``:

    model_N.pt          weights after epoch N (wespeaker format)
    train_state_N.pt    optimiser and scheduler position after epoch N; only
                        the newest is kept (wespeaker format)
    aux_state_N.pt      running centre of every process after epoch N
    partial_model.pt    weights of an interrupted epoch
    partial_state.pt    epoch, steps done, stream id, optimiser, scheduler
                        positions and the running centre of every process
                        at the interruption
    head_spk.json       class list the classifier head was verified against

``model_N.pt`` and ``train_state_N.pt`` are written by the wespeaker core
(``wespeaker.utils.train_state``); this module adds the other files around
them and decides where a run resumes.

Completion markers. Every file is written under another name and renamed.
An epoch is complete when its marker exists: ``train_state_N.pt`` (with
``engine.save_optimizer_state``; it is written last, after
``aux_state_N.pt`` and ``model_N.pt``), else ``model_N.pt``. An interrupted
epoch is resumable when ``partial_state.pt`` exists: both partial files are
written under staging names first, then the old ``partial_state.pt`` is
removed, then the two files are renamed, the state last. A process killed
at any point of a save therefore leaves either the new state or an older
complete one, never a mixture.

Resume order (``find_resume_point``): the partial state when it belongs to
an epoch after the newest complete one, else the newest complete epoch,
else a fresh start from the configured initial checkpoint.

The partial files are not named ``model_<N>.pt``, so tools that collect
epoch checkpoints (averaging, the drift gate) never see them.
"""
import glob
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.distributed as dist

from ovsb.losses.pair_auc import RunningCenter
from ovsb.model.build import HEAD_MODULE
from ovsb.model.head_record import HeadMismatchError, check_run_head
from wespeaker.utils.checkpoint import load_checkpoint, save_checkpoint
from wespeaker.utils.train_state import (atomic_save, restore_for_resume,
                                         save_epoch_artifacts,
                                         train_state_path)

PARTIAL_MODEL = 'partial_model.pt'
PARTIAL_STATE = 'partial_state.pt'
STAGING = '.next'
HEAD_WEIGHT = HEAD_MODULE + '.weight'
FRESH, EPOCH, PARTIAL = 'fresh', 'epoch', 'partial'
# scalar fields of a centre state, in the order of the gathered rows
_CENTER_SCALARS = ('n_upd', 'n_rows', 'n_skipped', 'row_norm')

CenterStates = Optional[List[Dict[str, Any]]]


class ResumeError(RuntimeError):
    """The saved state cannot continue under the current setup."""


def model_path(model_dir: str, epoch: int) -> str:
    """``model_<epoch>.pt`` in ``model_dir``."""
    return os.path.join(model_dir, 'model_%d.pt' % epoch)


def aux_state_path(model_dir: str, epoch: int) -> str:
    """``aux_state_<epoch>.pt`` in ``model_dir``."""
    return os.path.join(model_dir, 'aux_state_%d.pt' % epoch)


def saved_epochs(model_dir: str) -> List[int]:
    """Epochs N >= 1 that have a ``model_N.pt``, ascending."""
    epochs = []
    for path in glob.glob(os.path.join(model_dir, 'model_*.pt')):
        match = re.fullmatch(r'model_(\d+)\.pt', os.path.basename(path))
        if match and int(match.group(1)) >= 1:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def complete_epochs(model_dir: str, save_optimizer_state: bool) -> List[int]:
    """Saved epochs whose completion marker exists, ascending."""
    return [n for n in saved_epochs(model_dir) if not save_optimizer_state
            or os.path.exists(train_state_path(model_dir, n))]


@dataclass
class ResumePoint:
    """Where a run continues.

    ``kind`` is 'fresh' (nothing saved: epoch 1 from the initial
    checkpoint), 'epoch' (``epoch`` is complete) or 'partial' (``epoch``
    was interrupted after ``step`` steps of its pass number ``stream``).
    """
    kind: str
    epoch: int = 0
    step: int = 0
    stream: int = 0
    weights: Optional[str] = None  # checkpoint with the model weights

    @property
    def start_epoch(self) -> int:
        """Epoch the run continues in."""
        return self.epoch if self.kind == PARTIAL else self.epoch + 1

    @property
    def start_step(self) -> int:
        """Steps of ``start_epoch`` that are already done."""
        return self.step if self.kind == PARTIAL else 0

    @property
    def start_stream(self) -> int:
        """Stream id of the pass that continues ``start_epoch``."""
        return self.stream + 1 if self.kind == PARTIAL else 0

    def describe(self) -> str:
        """One line for the log."""
        if self.kind == FRESH:
            return 'fresh start at epoch 1'
        if self.kind == EPOCH:
            return 'resume after epoch %d (%s)' % (self.epoch, self.weights)
        return ('resume inside epoch %d at step %d, stream %d (%s)' %
                (self.epoch, self.step, self.start_stream, self.weights))


def _peek(path: str) -> Any:
    """Load a file for its structure: tensors stay on disk where the
    format allows it (memory map), else the file is read."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False,
                          mmap=True)
    except RuntimeError:  # not a zip archive: the legacy format
        return torch.load(path, map_location='cpu', weights_only=False)


def find_resume_point(model_dir: str,
                      save_optimizer_state: bool = True) -> ResumePoint:
    """The newest usable state of ``model_dir`` (module docstring).

    Reads file names and, when there is one, the header of
    ``partial_state.pt`` (memory-mapped: no tensor data is read).
    """
    complete = complete_epochs(model_dir, save_optimizer_state)
    newest = complete[-1] if complete else 0
    weights = os.path.join(model_dir, PARTIAL_MODEL)
    state_file = os.path.join(model_dir, PARTIAL_STATE)
    if os.path.exists(state_file) and os.path.exists(weights):
        state = _peek(state_file)
        if int(state['epoch']) > newest:
            return ResumePoint(PARTIAL, int(state['epoch']),
                               int(state['step']), int(state['stream']),
                               weights)
    if newest:
        return ResumePoint(EPOCH, newest, weights=model_path(model_dir,
                                                             newest))
    return ResumePoint(FRESH)


# ------------------------------------------------------------------ weights
def check_head_shape(model: torch.nn.Module, path: str) -> None:
    """Refuse a checkpoint whose classifier head does not fit the model.

    The wespeaker loader cuts a larger head to the first rows of the model
    and ignores a missing one; both would train another class assignment
    than the head record describes.

    Raises:
        HeadMismatchError: ``projection.weight`` is absent from the
            checkpoint or has another shape than the model's.
    """
    saved = _peek(path)
    if isinstance(saved, dict) and 'state_dict' in saved:
        saved = saved['state_dict']
    want = tuple(model.state_dict()[HEAD_WEIGHT].shape)
    if HEAD_WEIGHT not in saved:
        raise HeadMismatchError('%s: no %s; the model needs a head of shape '
                                '%s' % (path, HEAD_WEIGHT, want))
    have = tuple(saved[HEAD_WEIGHT].shape)
    if have != want:
        raise HeadMismatchError(
            '%s: %s has shape %s, the model needs %s (classes x sub-centres'
            ', embedding size); build the initial checkpoint for this '
            'training list' % (path, HEAD_WEIGHT, have, want))


def load_init(model: torch.nn.Module, path: str, spks: Iterable[str],
              model_dir: str, write: bool = True,
              logger: Optional[logging.Logger] = None) -> str:
    """Load the weights a process starts from, after two checks.

    1. Head record (``ovsb.model.head_record.check_run_head``): the class
       list the head of ``path`` was built for is the current one. The
       sidecar of ``path`` is used when it exists (an initial checkpoint),
       else the record of the run in ``model_dir`` (a checkpoint the run
       wrote itself).
    2. Head shape (``check_head_shape``).

    Then the wespeaker loader copies the tensors into ``model``.

    Args:
        model: the unwrapped model, on any device.
        path: initial checkpoint, ``model_N.pt`` or ``partial_model.pt``.
        spks: speaker ids of the training list.
        model_dir: the run's model directory.
        write: write the run's head record (True on one process only).
        logger: logger for one line about the check.

    Returns:
        'sidecar' or 'run': the record the head was verified against.
    """
    rows = int(model.state_dict()[HEAD_WEIGHT].shape[0])
    spks = list(spks)
    how = check_run_head(path, model_dir, spks, rows, write=write)
    check_head_shape(model, path)
    load_checkpoint(model, path)
    if logger is not None:
        logger.info('head class-list check: %s (%d classes, %d rows); '
                    'loaded %s' % (how, len(set(spks)), rows, path))
    return how


# ------------------------------------------------------------ centre states
def gather_center_states(center: Optional[RunningCenter], dim: int,
                         device: torch.device) -> CenterStates:
    """The centre state of every process, in rank order.

    One all-reduce in a distributed run; every process must call it at the
    same point and gets the same list. Values travel as float64 (exact for
    the fp32 sums and the counters).

    Args:
        center: the running centre of this process, None when the
            objective has none.
        dim: embedding size.
        device: training device of this process.

    Returns:
        ``RunningCenter.state_dict()`` of rank 0, 1, ...; None without a
        centre.
    """
    if center is None:
        return None
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    own = center.state_dict()
    table = torch.zeros(world, 1 + len(_CENTER_SCALARS) + dim,
                        dtype=torch.float64)
    head = len(_CENTER_SCALARS) + 1
    table[rank, 1:head] = torch.tensor(
        [float(own[name]) for name in _CENTER_SCALARS], dtype=torch.float64)
    if own['acc'] is not None:
        if own['acc'].numel() != dim:
            raise ValueError('centre of size %d, expected %d' %
                             (own['acc'].numel(), dim))
        table[rank, 0] = 1.0
        table[rank, head:] = own['acc'].double()
    if distributed:
        table = table.to(device)
        dist.all_reduce(table)
        table = table.cpu()
    states = []
    for row in table:
        state = dict(own)
        state.update(n_upd=int(row[1]), n_rows=int(row[2]),
                     n_skipped=int(row[3]), row_norm=float(row[4]),
                     acc=row[head:].float().clone() if row[0] else None)
        states.append(state)
    return states


def _check_world(state: Dict[str, Any], world: int, what: str) -> None:
    if int(state['world']) != world:
        raise ResumeError(
            '%s was written by a run of %d process(es), this run has %d; '
            'resume with the same number of processes' %
            (what, state['world'], world))


def _load_center(center: Optional[RunningCenter], states: CenterStates,
                 rank: int, what: str) -> None:
    if center is None:
        return
    if states is None:
        raise ResumeError('%s holds no running-centre state, but the '
                          'objective has a centre' % what)
    center.load_state_dict(states[rank])


# -------------------------------------------------------------------- saves
def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _commit(staged: str, final: str) -> None:
    """Rename a fully written staging file to its final name."""
    os.replace(staged, final)


def remove_partial(model_dir: str) -> None:
    """Delete the partial files, the state (the marker) first."""
    _remove(os.path.join(model_dir, PARTIAL_STATE))
    _remove(os.path.join(model_dir, PARTIAL_MODEL))


def save_epoch(model_dir: str, epoch: int, model: torch.nn.Module,
               optimizer: torch.optim.Optimizer, scheduler: Any,
               margin_scheduler: Any, center_states: CenterStates,
               world: int, stream: int = 0,
               save_optimizer_state: bool = True) -> None:
    """End-of-epoch save; call on one process (rank 0).

    Order: ``aux_state_N.pt``, then the wespeaker artifacts (``model_N.pt``,
    ``train_state_N.pt``), then the files the new state supersedes are
    removed (older aux states, the partial files of this epoch).

    Args:
        model: the model, wrapped or not.
        center_states: ``gather_center_states`` of this point.
        world: number of training processes.
        stream: stream id of the pass that finished the epoch (0 unless
            the epoch was resumed in the middle); kept as a record.
    """
    current = aux_state_path(model_dir, epoch)
    atomic_save({'epoch': int(epoch), 'stream': int(stream),
                 'world': int(world), 'center': center_states}, current)
    save_epoch_artifacts(model_dir, epoch, model, optimizer, None, scheduler,
                         margin_scheduler, None, save_optimizer_state, False)
    for old in glob.glob(os.path.join(model_dir, 'aux_state_*.pt')):
        if os.path.abspath(old) != os.path.abspath(current):
            _remove(old)
    remove_partial(model_dir)


def save_partial(model_dir: str, epoch: int, step: int, stream: int,
                 model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 scheduler: Any, margin_scheduler: Any,
                 center_states: CenterStates, world: int,
                 epoch_iter: int) -> None:
    """Save an interrupted epoch; call on one process (rank 0).

    Args:
        epoch: the interrupted epoch.
        step: optimiser steps of the epoch that are done (the resumed run
            continues with the step of this zero-based index).
        stream: stream id of the interrupted pass.
        model: the model, wrapped or not.
        scheduler, margin_scheduler: their ``current_iter`` is saved.
        center_states: ``gather_center_states`` of this point.
        world: number of training processes (checked on resume).
        epoch_iter: steps per epoch of this run (checked on resume).
    """
    weights = os.path.join(model_dir, PARTIAL_MODEL)
    state_file = os.path.join(model_dir, PARTIAL_STATE)
    save_checkpoint(model, weights + STAGING)
    atomic_save({'epoch': int(epoch), 'step': int(step),
                 'stream': int(stream), 'world': int(world),
                 'epoch_iter': int(epoch_iter),
                 'optimizer': optimizer.state_dict(),
                 'scheduler_iter': int(scheduler.current_iter),
                 'margin_scheduler_iter': int(margin_scheduler.current_iter),
                 'center': center_states}, state_file + STAGING)
    _remove(state_file)  # from here on the old pair is not a state
    _commit(weights + STAGING, weights)
    _commit(state_file + STAGING, state_file)


# ------------------------------------------------------------------- resume
def resume(point: ResumePoint, model_dir: str,
           optimizer: torch.optim.Optimizer, scheduler: Any,
           margin_scheduler: Any, center: Optional[RunningCenter], rank: int,
           world: int, epoch_iter: int, save_optimizer_state: bool = True,
           logger: Optional[logging.Logger] = None) -> None:
    """Restore the training state of ``point`` on this process.

    Call on every process after the schedulers were built (they split the
    optimiser's param groups) and after the weights of ``point.weights``
    were loaded. Restores the optimiser, the position of both schedulers
    and this process's running centre. A fresh point restores nothing.

    Raises:
        ResumeError: the state was written with another number of
            processes or steps per epoch, or a file of the state is absent.
    """
    if point.kind == FRESH:
        return
    if point.kind == EPOCH:
        restore_for_resume(point.weights, optimizer, None, scheduler,
                           margin_scheduler, None, save_optimizer_state,
                           logger)
        aux_file = aux_state_path(model_dir, point.epoch)
        if not os.path.exists(aux_file):
            raise ResumeError(
                '%s is missing: epoch %d was not saved by this trainer, its '
                'running-centre state is unknown' % (aux_file, point.epoch))
        aux = torch.load(aux_file, map_location='cpu', weights_only=False)
        _check_world(aux, world, aux_file)
        _load_center(center, aux['center'], rank, aux_file)
        return
    what = os.path.join(model_dir, PARTIAL_STATE)
    state = torch.load(what, map_location='cpu', weights_only=False)
    _check_world(state, world, what)
    if int(state['epoch_iter']) != epoch_iter:
        raise ResumeError(
            '%s was written with %d steps per epoch, this run has %d; '
            'resume with the same data.sampler.sample_num_per_epoch and '
            'batch size' % (what, state['epoch_iter'], epoch_iter))
    optimizer.load_state_dict(state['optimizer'])
    scheduler.current_iter = state['scheduler_iter']
    margin_scheduler.current_iter = state['margin_scheduler_iter']
    _load_center(center, state['center'], rank, what)
    if logger is not None:
        logger.info('Restored optimizer and centre state from %s' % what)
