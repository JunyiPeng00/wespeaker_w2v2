"""One training step on a batch.

A batch is the dict the data layer collates. The step reads

    wav        (B, 1, T) float waveforms, zero-padded to the longest row
    wav_len    (B,) valid samples per row
    label      (B,) class labels
    pair_role  (B,) 0 plain row, 1 anchor, 2 short view of the row before
    arm        (B,) sampler arm index of each row

Numerics. The encoder and the pooling backend run under bfloat16 autocast
(on CUDA). The classifier head and all losses run in fp32 with autocast
off: in bfloat16 a cosine above about 0.998 rounds to 1 and
``sqrt(1 - cos^2)`` has no finite gradient there. The target margin of row
``i`` is scaled by ``min(1, wav_len_i / (ref_sec * sample_rate))``.

Order of one step (``train_step``):

1. learning-rate and margin schedulers are set for the global step index;
2. forward and losses (``forward_losses``); the running centre of the
   pair-AUC term is updated with this batch inside the loss;
3. non-finite loss on any process: gradients cleared, step skipped;
4. backward, gradient-norm clipping over all parameters;
5. non-finite gradient norm on any process: gradients cleared, step
   skipped (a NaN gradient can come with a finite loss);
6. optimiser step.

bfloat16 needs no loss scaling, so there is no gradient scaler.
"""
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from ovsb.config.schema import Config
from ovsb.engine.guards import all_ranks_finite
from ovsb.losses.bundle import Components, LossBundle, Stats
from ovsb.model.head import margin_scale_from_len

SKIP_LOSS = 'loss'
SKIP_GRAD = 'grad norm'


@dataclass
class ForwardOutput:
    """Result of ``forward_losses``; ``loss`` and ``logits`` carry a graph."""
    loss: torch.Tensor    # scalar objective
    parts: Components     # detached unweighted terms 'ce', 'cons', 'pauc'
    stats: Stats          # pair counts and mean cosines of the batch
    logits: torch.Tensor  # (B, C) fp32 margin logits
    emb: torch.Tensor     # (B, D) raw embeddings, autocast dtype
    labels: torch.Tensor  # (B,) labels on the device


@dataclass
class StepResult:
    """Result of ``train_step``; all tensors are detached.

    ``skipped`` is None when the optimiser stepped, ``SKIP_LOSS`` when the
    loss was not finite (no backward) and ``SKIP_GRAD`` when the gradient
    norm was not finite (backward done, no optimiser step). ``grad_norm``
    is the total gradient norm before clipping, None without a backward.
    """
    loss: torch.Tensor
    parts: Components
    stats: Stats
    logits: torch.Tensor
    labels: torch.Tensor
    skipped: Optional[str]
    grad_norm: Optional[torch.Tensor]


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The model inside a (Distributed)DataParallel wrapper, or itself."""
    return getattr(model, 'module', model)


def forward_losses(model: torch.nn.Module, bundle: LossBundle,
                   batch: Mapping[str, Any], cfg: Config,
                   device: torch.device,
                   amp: Optional[bool] = None) -> ForwardOutput:
    """Forward one batch and evaluate the objective; no optimiser step.

    The mode of the model (train / eval) is left as it is. In training
    mode the call draws the frame mask from the global generator of
    ``device`` and updates the running centre held by ``bundle``.

    Args:
        model: the model, optionally in a DistributedDataParallel wrapper;
            the pooling backend is called through the wrapper.
        bundle: the objective.
        batch: collated batch (fields in the module docstring).
        cfg: the recipe config (duration-margin reference, sample rate).
        device: device of the model.
        amp: bfloat16 autocast for encoder and backend; None means on
            for CUDA and off (fp32) elsewhere.
    """
    net = unwrap(model)
    if amp is None:
        amp = device.type == 'cuda'
    labels = batch['label'].long().to(device)
    wav = batch['wav'].squeeze(1).float().to(device)
    wav_len = batch['wav_len'].long().to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        # flens: valid frames per row, None when no row is padded
        feats, flens = net.frontend(wav, wav_len)
        out = model(feats) if flens is None else model(feats, flens)
        emb = out[-1] if isinstance(out, tuple) else out
    scale = margin_scale_from_len(
        wav_len, cfg.model.head.duration_margin_ref_sec,
        cfg.data.resample_rate)
    with torch.autocast(device.type, enabled=False):
        logits = net.projection(emb.float(), labels, scale)
    loss, parts, stats = bundle(logits, emb, labels,
                                batch['pair_role'].to(device),
                                batch['arm'].to(device))
    return ForwardOutput(loss, parts, stats, logits, emb, labels)


def train_step(model: torch.nn.Module, bundle: LossBundle,
               batch: Mapping[str, Any], cfg: Config, device: torch.device,
               optimizer: torch.optim.Optimizer, scheduler: Any,
               margin_scheduler: Any, cur_iter: int,
               amp: Optional[bool] = None) -> StepResult:
    """One optimisation step (order in the module docstring).

    In a distributed run every process must call this for the same
    ``cur_iter``: the two finite checks are all-reduced, so all processes
    skip a step together.

    Args:
        model, bundle, batch, cfg, device, amp: as in ``forward_losses``;
            the model must be in training mode.
        optimizer: optimiser over the model parameters.
        scheduler: learning-rate scheduler, stepped with ``cur_iter``.
        margin_scheduler: margin scheduler, stepped with ``cur_iter``.
        cur_iter: zero-based global step, ``(epoch - 1) * epoch_iter + i``.
    """
    scheduler.step(cur_iter)
    margin_scheduler.step(cur_iter)
    out = forward_losses(model, bundle, batch, cfg, device, amp)
    skipped, grad_norm = None, None
    if not all_ranks_finite(out.loss):
        skipped = SKIP_LOSS
        optimizer.zero_grad()
    else:
        optimizer.zero_grad()
        out.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg.optim.grad_clip)
        if not all_ranks_finite(grad_norm):
            skipped = SKIP_GRAD
            optimizer.zero_grad()
        else:
            optimizer.step()
    return StepResult(out.loss.detach(), out.parts, out.stats,
                      out.logits.detach(), out.labels, skipped, grad_norm)
