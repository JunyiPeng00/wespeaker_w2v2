"""Optimiser, learning-rate schedule and margin schedule of the recipe.

AdamW over all model parameters in two param groups, in this order:

    group 0  encoder   parameters named ``<frontend module>.*``
    group 1  base      pooling backend and classifier head

The wespeaker ``ExponentialDecrease`` scheduler makes the split and sets
the learning rates at every step ``t`` (zero-based, over the whole run):

    base(t)    = c(t) * initial_lr * (final_lr / initial_lr) ** (t / T)
    encoder(t) = frontend_lr_ratio * base(t)
    c(t)       = scale_ratio * min(1, t / (warm_up_epoch * epoch_iter))

with ``T = num_epochs * epoch_iter`` and ``scale_ratio = processes x
batch_size / 64`` (linear warm-up from zero, then exponential decay). The
margin of the classifier head is constant.
"""
from typing import Any, Dict, Tuple

import torch

from ovsb.config import schema
from ovsb.config.schema import Config
from wespeaker.utils.schedulers import ExponentialDecrease, MarginScheduler


def build_optimizer(model: torch.nn.Module,
                    cfg: Config) -> torch.optim.Optimizer:
    """AdamW over ``model.parameters()`` in one group.

    ``build_schedulers`` splits the group in two; call it before the first
    step and before restoring an optimiser state. Build the optimiser after
    the model is on its device (``fused`` checks the parameter devices).

    Args:
        model: the unwrapped model (not a DistributedDataParallel wrapper;
            both give the same parameters in the same order).
        cfg: the recipe config.
    """
    optim = cfg.optim
    return getattr(torch.optim, optim.optimizer)(
        model.parameters(), lr=optim.initial_lr,
        weight_decay=optim.weight_decay, fused=optim.fused)


def build_schedulers(
        optimizer: torch.optim.Optimizer, model: torch.nn.Module,
        cfg: Config) -> Tuple[ExponentialDecrease, MarginScheduler]:
    """Learning-rate and margin schedulers; splits the param groups.

    Both are stepped with the global step index before every training
    step: ``scheduler.step(t)``, ``margin_scheduler.step(t)``.

    Args:
        optimizer: the optimiser of ``build_optimizer`` (one param group).
        model: the unwrapped model the optimiser was built from; its
            parameter names decide the group of each parameter.
        cfg: the recipe config; the number of processes is
            ``len(cfg.run.gpus)``.

    Returns:
        (scheduler, margin_scheduler). The margin scheduler sets the head
        margin to ``cfg.model.head.margin`` and keeps it there.
    """
    optim = cfg.optim
    epoch_iter = schema.epoch_iter(cfg)
    scheduler = ExponentialDecrease(
        optimizer, num_epochs=cfg.run.num_epochs, epoch_iter=epoch_iter,
        initial_lr=optim.initial_lr, final_lr=optim.final_lr,
        warm_up_epoch=optim.warm_up_epoch,
        scale_ratio=schema.scale_ratio(cfg),
        warm_from_zero=optim.warm_from_zero,
        frontend_modules=list(optim.frontend_modules),
        frontend_lr_ratio=optim.frontend_lr_ratio, model=model)
    if [g['_is_frontend'] for g in optimizer.param_groups] != [True, False]:
        raise ValueError(
            'expected an encoder and a base param group, got %d group(s); '
            'optim.frontend_modules %s must name a part of the model' %
            (len(optimizer.param_groups), list(optim.frontend_modules)))
    margin = cfg.model.head.margin
    margin_scheduler = MarginScheduler(
        model=model, epoch_iter=epoch_iter, increase_start_epoch=1,
        fix_start_epoch=1, initial_margin=margin, final_margin=margin,
        update_margin=False)
    return scheduler, margin_scheduler


def learning_rates(optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    """(base, encoder) learning rates currently set in the optimiser."""
    rates: Dict[bool, Any] = {
        bool(g['_is_frontend']): g['lr'] for g in optimizer.param_groups}
    return float(rates[False]), float(rates[True])
