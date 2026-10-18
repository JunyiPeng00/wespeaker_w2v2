# Copyright (c) 2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optimizer / scaler / scheduler / EMA state next to model_N.pt.

Files in <exp_dir>/models/ for epoch N:
    model_N.pt        model state dict (unchanged format)
    ema_model_N.pt    EMA weights, same format (ema_decay set)
    train_state_N.pt  {'epoch', 'optimizer', 'scaler', 'scheduler',
                       'margin_scheduler', 'ema_num_updates'}
                      (save_optimizer_state: true)
"""

import os

import torch


def atomic_save(obj, path):
    """write-then-rename: a job killed mid-save never leaves a truncated
    file that a resume would pick up"""
    tmp_path = path + '.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _plain_attrs(obj):
    # schedulers here are driven by step(cur_iter); their scalar attributes
    # (current_iter, fix_already, ...) are all the state there is
    return {k: v for k, v in vars(obj).items()
            if isinstance(v, (bool, int, float, str))}


def ema_model_path(model_dir, epoch):
    return os.path.join(model_dir, 'ema_model_{}.pt'.format(epoch))


def train_state_path(model_dir, epoch):
    return os.path.join(model_dir, 'train_state_{}.pt'.format(epoch))


def save_train_state(path, epoch, optimizer, scaler=None, scheduler=None,
                     margin_scheduler=None, ema=None):
    state = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'scheduler': _plain_attrs(scheduler) if scheduler else None,
        'margin_scheduler':
        _plain_attrs(margin_scheduler) if margin_scheduler else None,
        'ema_num_updates': ema.num_updates if ema is not None else None,
    }
    atomic_save(state, path)


# Schedulers are pure functions of (config, cur_iter): train.py rebuilds them
# from the config and run_epoch() calls step(cur_iter) on every batch. Only
# the iteration counter is restored. Config-derived attributes (max_iter,
# initial_lr, scale_ratio, increase/fix_start_iter, ...) keep the CURRENT
# config (a mismatch with the saved run is logged). MarginScheduler
# .fix_already is NOT restored: the margin lives on the
# projection as a plain attribute that model_N.pt does not hold, so a
# restored fix_already=True made step() return early and left the margin
# at initial_margin for the rest of a run resumed after fix_start_epoch.
RESTORED_SCHEDULER_KEYS = ('current_iter', )


def load_train_state(path, optimizer, scaler=None, scheduler=None,
                     margin_scheduler=None, logger=None):
    """Restore in place (call AFTER the scheduler restructured the param
    groups). Returns the raw state dict (for 'epoch', 'ema_num_updates')."""
    warn = logger.warning if logger is not None else (lambda *a: None)
    state = torch.load(path, map_location='cpu', weights_only=False)
    optimizer.load_state_dict(state['optimizer'])
    if scaler is not None and state.get('scaler'):
        scaler.load_state_dict(state['scaler'])
    for obj, key in ((scheduler, 'scheduler'),
                     (margin_scheduler, 'margin_scheduler')):
        if obj is not None and state.get(key):
            for k, v in state[key].items():
                if k in RESTORED_SCHEDULER_KEYS:
                    setattr(obj, k, v)
                elif k != 'fix_already' and hasattr(obj, k) and \
                        getattr(obj, k) != v:
                    warn('{}.{}: saved {} != config {}; using the config '
                         'value'.format(key, k, v, getattr(obj, k)))
    return state


def save_epoch_artifacts(model_dir, epoch, model, optimizer, scaler=None,
                         scheduler=None, margin_scheduler=None, ema=None,
                         save_optimizer_state=False, keep_all_states=False):
    """Rank-0 end-of-epoch save: model_N.pt (+ ema_model_N.pt, +
    train_state_N.pt). Only the newest train_state is kept unless
    keep_all_states (it is ~2x the trainable weights)."""
    import glob
    from wespeaker.utils.checkpoint import save_checkpoint
    save_checkpoint(model, os.path.join(model_dir,
                                        'model_{}.pt'.format(epoch)))
    if ema is not None:
        atomic_save(ema.model_state_dict(model),
                    ema_model_path(model_dir, epoch))
    if save_optimizer_state:
        current = train_state_path(model_dir, epoch)
        save_train_state(current, epoch, optimizer, scaler, scheduler,
                         margin_scheduler, ema)
        if not keep_all_states:
            for old in glob.glob(os.path.join(model_dir, 'train_state_*.pt')):
                if os.path.abspath(old) != os.path.abspath(current):
                    os.remove(old)


def restore_for_resume(checkpoint, optimizer, scaler=None, scheduler=None,
                       margin_scheduler=None, ema=None,
                       save_optimizer_state=False, logger=None):
    """Resume from <dir>/model_N.pt: restore train_state_N.pt (optimizer,
    scaler, schedulers) when save_optimizer_state, and ema_model_N.pt into
    `ema`. Missing files (fresh start, stage hand-off via model_0.pt) leave a
    fresh optimizer / an EMA initialised from the loaded model."""
    import re
    log = logger.info if logger is not None else (lambda *a: None)
    warn = logger.warning if logger is not None else (lambda *a: None)
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint))
    epoch = int(re.findall(r"(?<=model_)\d+(?=\.pt)",
                           os.path.basename(checkpoint))[0])
    state = None
    if save_optimizer_state:
        state_file = train_state_path(ckpt_dir, epoch)
        if os.path.exists(state_file):
            state = load_train_state(state_file, optimizer, scaler, scheduler,
                                     margin_scheduler, logger)
            log('Restored optimizer/scaler/scheduler state from {}'.format(
                state_file))
        else:
            warn('save_optimizer_state: {} not found, fresh optimizer'.format(
                state_file))
    if ema is not None:
        ema_file = ema_model_path(ckpt_dir, epoch)
        if os.path.exists(ema_file):
            ema.load_model_state_dict(
                torch.load(ema_file, map_location='cpu', weights_only=False),
                num_updates=(state or {}).get('ema_num_updates') or 0)
            log('Restored EMA from {}'.format(ema_file))
        else:
            log('EMA initialised from the loaded model ({} not found)'.format(
                ema_file))
    return state
