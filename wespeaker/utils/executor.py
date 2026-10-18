# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tableprint as tp

import torch
import torch.distributed as dist
import torchnet as tnt
from wespeaker.dataset.dataset_utils import apply_cmvn, spec_aug



def grad_norm_adam_scale(cur_iter, total_iters, cfg):
    """Loss scale S_t of the legacy run: scale_init * growth_factor^floor(t/interval).

    Knobs (configs['grad_norm_adam']): scale_init (65536), growth_factor (2.0),
    growth_interval (2000 steps), reference_total_iters (None = per-step
    schedule exactly as the legacy GradScaler; int = stretch the interval by
    total_iters / reference_total_iters so the profile follows training
    PROGRESS), max_scale (None = unbounded, as the legacy GradScaler).
    """
    interval = float(cfg.get('growth_interval', 2000))
    ref = cfg.get('reference_total_iters')
    if ref:
        interval *= float(total_iters) / float(ref)
    s = float(cfg.get('scale_init', 65536.0)) * float(
        cfg.get('growth_factor', 2.0)) ** int(cur_iter // max(interval, 1.0))
    if cfg.get('max_scale'):
        s = min(s, float(cfg['max_scale']))
    return s


def apply_grad_norm_adam(model, optimizer, cur_iter, total_iters, cfg):
    """Explicit, exact form of the legacy clip-on-scaled-grads regime.

    Legacy fed AdamW(eps) with u / S_t (u = globally L2-normalised gradient).
    Adam is homogeneous under a CONSTANT gradient factor c with eps -> c*eps,
    so with c = S_0 this equals AdamW(eps * S_0) fed with u * d_t,
    d_t = S_0 / S_t (1, 1/2, 1/4, ... every growth_interval). This keeps the
    Adam moments in the same units as the legacy run, so it is exact across
    the S doublings (an eps-schedule form is not). Interpretation:
    normalised-gradient AdamW with a large eps (eps_eff = eps * S_0 = 6.6e-4
    by default) and a gradient-magnitude decay d_t. frontend_eps_mult scales
    the frontend groups' eps (constant over time). Requires GradScaler off.
    Returns (S_t, pre-normalisation gradient norm).
    """
    params = [p for p in model.parameters() if p.grad is not None]
    total = torch.nn.utils.clip_grad_norm_(params, float('inf'))
    s0 = float(cfg.get('scale_init', 65536.0))
    s_t = grad_norm_adam_scale(cur_iter, total_iters, cfg)
    if torch.isfinite(total) and total > 0:
        # legacy: clip(g*S, 1)/S = g / (S*||g|| + 1e-6); times S_0
        coef = s0 / (s_t * total + 1e-6)
        for p in params:
            p.grad.mul_(coef)
    eps_eff = float(cfg.get('eps', 1e-8)) * s0
    fe_mult = float(cfg.get('frontend_eps_mult', 1.0))
    for group in optimizer.param_groups:
        group['eps'] = eps_eff * (fe_mult if group.get('_is_frontend', False) else 1.0)
    return s_t, total

def run_epoch(dataloader, epoch_iter, model, criterion, optimizer, scheduler,
              margin_scheduler, epoch, logger, scaler, device, configs):
    model.train()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    amp_dtype = getattr(torch, configs.get('amp_dtype', 'float16'))
    grad_clip = configs.get('grad_clip', None)  # e.g. 1.0; None = off
    clip_on_scaled = configs.get('grad_clip_on_scaled_grads', False)
    gna_cfg = configs.get('grad_norm_adam') or {}
    use_gna = bool(gna_cfg.get('enable', False))
    total_iters = epoch_iter * configs['num_epochs']
    for i, batch in enumerate(dataloader):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        utts = batch['key']
        targets = batch['label']
        targets = targets.long().to(device)  # (B)
        if frontend_type == 'fbank':
            features = batch['feat']  # (B,T,F)
            features = features.float().to(device)
        else:  # 's3prl', 'tfmel'
            wavs = batch['wav']  # (B,1,W)
            wavs = wavs.squeeze(1).float().to(device)  # (B,W)
            wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                wavs.shape[0]).to(device)  # (B)
            with torch.autocast('cuda', enabled=configs['enable_amp'],
                               dtype=amp_dtype):
                features, _ = model.module.frontend(wavs, wavs_len)

        with torch.autocast('cuda', enabled=configs['enable_amp'],
                               dtype=amp_dtype):
            # apply cmvn
            if configs['dataset_args'].get('cmvn', True):
                features = apply_cmvn(
                    features, **configs['dataset_args'].get('cmvn_args', {}))
            # spec augmentation
            if configs['dataset_args'].get('spec_aug', False):
                features = spec_aug(features,
                                    **configs['dataset_args']['spec_aug_args'])

            outputs = model(features)  # (embed_a,embed_b) in most cases
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
        # The margin projection runs in fp32: under bf16 autocast a cosine
        # >= ~0.998 rounds to exactly 1.0 and sqrt(1 - cos^2) has a NaN
        # gradient. It is a single (B x C) matmul, so the cost is negligible.
        with torch.autocast('cuda', enabled=False):
            outputs = model.module.projection(embeds.float(), targets)
            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)

        # bf16 has no GradScaler to skip non-finite steps; do it here. The
        # decision is all-reduced so every DDP rank skips (or not) together,
        # otherwise one rank would miss the gradient all-reduce and hang.
        finite = torch.isfinite(loss.detach()).to(torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() == 0:
            logger.warning('non-finite loss at epoch {} batch {}, step '
                           'skipped on all ranks'.format(epoch, i + 1))
            optimizer.zero_grad()
            continue

        # loss, acc
        loss_meter.add(loss.item())
        acc_meter.add(outputs.float().cpu().detach().numpy(),
                      targets.cpu().numpy())

        # updata the model
        optimizer.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(loss).backward()
        if use_gna:
            gna_scale, gna_gnorm = apply_grad_norm_adam(
                model, optimizer, cur_iter, total_iters, gna_cfg)
            if (i + 1) % configs['log_batch_interval'] == 0:
                logger.info('grad_norm_adam: S_t %.6g eps %.3g grad-norm %.4g '
                            'at epoch %d batch %d' % (
                                gna_scale, optimizer.param_groups[0]['eps'],
                                float(gna_gnorm), epoch, i + 1))
        elif grad_clip:
            if clip_on_scaled:
                # LEGACY the original behaviour (the one that produced the
                # released mhfa_W2VB_ft_v2): clip the SCALED gradients to
                # `grad_clip` and let scaler.step() unscale afterwards, so the
                # true gradient norm becomes grad_clip / S (S = loss scale,
                # 2^16 and doubling every 2000 steps). Per-element gradients
                # then sit far below Adam's eps, which turns AdamW into a
                # damped normalised-gradient step (~16x smaller, proportional
                # to the gradient). Requires float16 autocast + GradScaler.
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # log
        if clip_on_scaled and (i + 1) % configs['log_batch_interval'] == 0:
            logger.info('legacy GradScaler scale %.6g at epoch %d batch %d'
                        % (scaler.get_scale(), epoch, i + 1))
        if (i + 1) % configs['log_batch_interval'] == 0:
            logger.info(
                tp.row((epoch, i + 1, scheduler.get_lr(),
                        margin_scheduler.get_margin()) +
                       (loss_meter.value()[0], acc_meter.value()[0]),
                       width=10,
                       style='grid'))

        if (i + 1) == epoch_iter:
            break

    logger.info(
        tp.row(
            (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
            (loss_meter.value()[0], acc_meter.value()[0]),
            width=10,
            style='grid'))
