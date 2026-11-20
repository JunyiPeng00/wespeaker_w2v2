"""Training step: numerics, order, skip semantics, parity with the source.

Parity tests (environment variables in tests/test_build.py):

* tiny model: the original trainer's epoch function runs three fixed
  batches in a subprocess under the source tree, from the same weights;
  losses, loss terms, gradients and weights after every step are compared.
* real model (``OVSB_REAL_MODEL=1``, about 15 GB of memory, a few minutes):
  two steps on the reference run's final weights against the numbers the
  source tree's parity harness recorded.

Both run on the CPU in fp32. bfloat16 autocast is not compared with the
source: its trainer enables autocast for CUDA only.
"""
import copy
import math
import random
import re

import numpy as np
import pytest
import torch

from test_build import (TINY_CLASSES, build_real_model, load_reference,
                        make_tiny_assets, needs_real_model, needs_source,
                        real_config, run_source, synth_batches, tiny_config,
                        write_tiny_legacy, SOURCE_RECIPE)

from ovsb.engine import step as step_mod
from ovsb.engine.optim import build_optimizer, build_schedulers
from ovsb.engine.step import (SKIP_GRAD, SKIP_LOSS, forward_losses,
                              train_step, unwrap)
from ovsb.losses.bundle import LossBundle
from ovsb.losses.pair_auc import RunningCenter
from ovsb.model.build import build_model
from ovsb.model.head import margin_scale_from_len

CPU = torch.device('cpu')
SR = 16000


@pytest.fixture(scope='module')
def tiny_assets(tmp_path_factory):
    return make_tiny_assets(tmp_path_factory.mktemp('tiny_w2vbert'))


def make_bundle(cfg, warmup=None):
    pauc = cfg.loss.pair_auc
    center = RunningCenter(cfg.data.sampler.arm_order.index(pauc.center.arm),
                           pauc.center.momentum, warmup or pauc.center.warmup)
    return LossBundle(cfg.loss.consistency_weight, pauc.weight, pauc.scale,
                      center)


def reseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batch(seed, arm_order, lens, labels, arms, roles=None):
    """A collated batch with the fields of the data layer (and the extra
    constant fields the source collate adds, which the recipe ignores)."""
    gen = torch.Generator().manual_seed(seed)
    rows, width = len(lens), max(lens)
    wav = torch.zeros(rows, 1, width)
    for i, n in enumerate(lens):
        wav[i, 0, :n] = 0.1 * torch.randn(n, generator=gen)
    roles = roles or [0] * rows
    for i, role in enumerate(roles):  # a view is cut from its anchor
        if role == 2:
            wav[i, 0, :lens[i]] = wav[i - 1, 0, 2000:2000 + lens[i]]

    def const(value, dtype=torch.long):
        return torch.full((rows,), value, dtype=dtype)

    return {
        'key': ['utt%d_%d' % (seed, i) for i in range(rows)],
        'label': torch.tensor(labels), 'sample_rate': const(SR), 'wav': wav,
        'wav_len': torch.tensor(lens), 'pair_role': torch.tensor(roles),
        'arm': torch.tensor([arm_order.index(a) for a in arms]),
        'kpos': torch.arange(rows) % 2, 'ptype': const(0),
        'aug_code': const(0), 'chain_code': const(-1),
        'zr_lsb': const(0.0, torch.float64), 'zr_run': const(0),
        'zr_dith': const(0), 'zr_bypass': const(0), 'eg_redraw': const(-1),
        'eg_fail': const(0),
        'label_b': const(-1), 'lam': const(1.0, torch.float64),
        'lang': const(0), 'vid': torch.arange(rows), 'seg': const(-1),
        'dev': const(-1), 'dcls': const(-1), 'dial': const(-1),
        'genre': const(-1)}


def tiny_batches(arm_order):
    """plain ragged (with centre-arm rows), pair, plain equal-length
    without a centre-arm row."""
    plain = make_batch(
        1, arm_order, [16000, 9600, 12800, 16000, 11200, 14400, 9920, 15040],
        [3, 3, 7, 7, 11, 11, 20, 20],
        ['vox2', 'vox2', 'cn1', 'cn1', 'mandarin_t1', 'mandarin_t1', 'vb2',
         'vb2'])
    pair = make_batch(
        2, arm_order, [16000, 8000, 16000, 12000, 16000, 9600, 16000, 11200],
        [3, 3, 3, 3, 9, 9, 9, 9], ['vox2'] * 4 + ['cn1'] * 4,
        roles=[1, 2] * 4)
    equal = make_batch(
        3, arm_order, [14400] * 8, [5, 5, 30, 30, 31, 31, 39, 39],
        ['cn1', 'cn1', 'tvx', 'tvx', '3dspeaker', '3dspeaker', 'kespeech',
         'kespeech'])
    return [plain, pair, equal]


@pytest.fixture()
def setup(tiny_assets):
    """(cfg, model in training mode, optimizer, scheduler, margin, batches)."""
    cfg = tiny_config(tiny_assets)
    torch.manual_seed(21)
    model = build_model(cfg, TINY_CLASSES).train()
    optimizer = build_optimizer(model, cfg)
    scheduler, margin = build_schedulers(optimizer, model, cfg)
    return cfg, model, optimizer, scheduler, margin, \
        tiny_batches(cfg.data.sampler.arm_order)


def params_of(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


# ------------------------------------------------------------------- unit
def test_forward_losses_terms_and_dtypes(setup):
    cfg, model, optimizer, _, _, batches = setup
    before = params_of(model)
    for index, batch in enumerate(batches):
        bundle = make_bundle(cfg, warmup=1)
        reseed(4)
        out = forward_losses(model, bundle, batch, cfg, CPU)
        assert out.loss.dim() == 0 and out.loss.requires_grad
        assert out.logits.shape == (8, TINY_CLASSES)
        assert out.logits.dtype == out.emb.dtype == torch.float32
        assert out.emb.shape == (8, 16) and out.labels.dtype == torch.long
        parts = out.parts
        assert (parts['cons'] is not None) == (index == 1)
        assert parts['pauc'] is not None
        total = parts['ce'] + 0.2 * parts['pauc']
        if parts['cons'] is not None:
            total = total + 0.5 * parts['cons']
        assert out.loss.item() == pytest.approx(total.item(), rel=1e-6)
        # the logits are the head's, with the duration-scaled margin
        scale = margin_scale_from_len(batch['wav_len'], 3.0, SR)
        assert scale.min() < 1.0
        want = model.projection(out.emb.detach(), out.labels, scale)
        assert torch.equal(out.logits.detach(), want)
        full = model.projection(out.emb.detach(), out.labels, torch.ones(8))
        assert not torch.equal(want, full)
    assert all(p.grad is None for p in model.parameters())
    after = params_of(model)
    assert all(torch.equal(before[n], after[n]) for n in before)


def test_forward_losses_updates_the_centre_once(setup):
    cfg, model, _, _, _, batches = setup
    bundle = make_bundle(cfg)
    forward_losses(model, bundle, batches[0], cfg, CPU)
    assert (bundle.center.n_upd, bundle.center.n_rows) == (1, 2)
    forward_losses(model, bundle, batches[2], cfg, CPU)  # no centre-arm row
    assert bundle.center.n_upd == 1
    forward_losses(model, bundle, batches[1], cfg, CPU)
    assert (bundle.center.n_upd, bundle.center.n_rows) == (2, 6)


def test_train_step_order_and_update(setup):
    cfg, model, optimizer, scheduler, margin, batches = setup
    bundle = make_bundle(cfg, warmup=1)
    before = params_of(model)
    seen = {}
    optimizer.register_step_pre_hook(lambda opt, args, kwargs: seen.update(
        lr=[g['lr'] for g in opt.param_groups],
        norm=math.sqrt(sum(float(p.grad.pow(2).sum())
                           for p in model.parameters()))))
    cur = 3645 + 17
    reseed(5)
    res = train_step(model, bundle, batches[0], cfg, CPU, optimizer,
                     scheduler, margin, cur)
    assert res.skipped is None
    # the schedulers were set for this step before the optimiser stepped
    fresh = build_optimizer(model, cfg)
    ref_sched, _ = build_schedulers(fresh, model, cfg)
    ref_sched.step(cur)
    assert seen['lr'] == [g['lr'] for g in fresh.param_groups]
    assert seen['lr'][0] == pytest.approx(0.3 * seen['lr'][1])
    # clipping: the optimiser saw a gradient of norm min(1, total)
    assert float(res.grad_norm) > 0
    assert seen['norm'] == pytest.approx(min(1.0, float(res.grad_norm)),
                                         rel=1e-4)
    after = params_of(model)
    changed = [n for n in before if not torch.equal(before[n], after[n])]
    assert len(changed) == len(before)  # encoder, backend and head all move
    assert not res.loss.requires_grad and not res.logits.requires_grad
    assert res.logits.shape == (8, TINY_CLASSES)
    assert set(res.parts) == {'ce', 'cons', 'pauc'}
    assert bundle.center.n_upd == 1


def test_three_steps_reduce_the_loss_on_one_batch(setup):
    cfg, model, optimizer, scheduler, margin, batches = setup
    bundle = make_bundle(cfg)
    losses = []
    for i in range(6):
        reseed(6)
        losses.append(float(train_step(
            model, bundle, batches[0], cfg, CPU, optimizer, scheduler,
            margin, 600 + i).loss))
    assert losses[-1] < losses[0]


def test_non_finite_loss_skips_before_backward(setup):
    cfg, model, optimizer, scheduler, margin, batches = setup
    bundle = make_bundle(cfg)
    train_step(model, bundle, batches[0], cfg, CPU, optimizer, scheduler,
               margin, 700)  # leaves gradients and a centre behind
    before, acc = params_of(model), bundle.center.acc.clone()
    bad = dict(batches[0])
    bad['wav'] = bad['wav'].clone()
    bad['wav'][0, 0, 100] = float('nan')  # a centre-arm row
    res = train_step(model, bundle, bad, cfg, CPU, optimizer, scheduler,
                     margin, 701)
    assert res.skipped == SKIP_LOSS and res.grad_norm is None
    assert not bool(torch.isfinite(res.loss))
    assert all(p.grad is None for p in model.parameters())
    after = params_of(model)
    assert all(torch.equal(before[n], after[n]) for n in before)
    # a non-finite batch leaves the running centre untouched
    assert torch.equal(bundle.center.acc, acc)
    assert (bundle.center.n_upd, bundle.center.n_skipped) == (1, 1)
    # the next good batch trains again
    res = train_step(model, bundle, batches[0], cfg, CPU, optimizer,
                     scheduler, margin, 702)
    assert res.skipped is None and bundle.center.n_upd == 2


def test_non_finite_gradient_skips_the_optimiser(setup, monkeypatch):
    cfg, model, optimizer, scheduler, margin, batches = setup
    bundle = make_bundle(cfg)
    before = params_of(model)
    clip = torch.nn.utils.clip_grad_norm_
    calls = []

    def nan_norm(parameters, max_norm):
        calls.append(max_norm)
        return clip(parameters, max_norm) * float('nan')

    monkeypatch.setattr(step_mod.torch.nn.utils, 'clip_grad_norm_', nan_norm)
    res = train_step(model, bundle, batches[0], cfg, CPU, optimizer,
                     scheduler, margin, 800)
    assert calls == [cfg.optim.grad_clip] == [1.0]
    assert res.skipped == SKIP_GRAD and bool(torch.isnan(res.grad_norm))
    assert bool(torch.isfinite(res.loss))  # the step still counts in the log
    assert all(p.grad is None for p in model.parameters())
    after = params_of(model)
    assert all(torch.equal(before[n], after[n]) for n in before)
    assert not optimizer.state  # no moment was created


class Wrapper(torch.nn.Module):
    """Stands in for DistributedDataParallel on one process."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.module(*args, **kwargs)


def test_wrapped_model_gives_the_same_loss(setup):
    cfg, model, _, _, _, batches = setup
    reseed(8)
    bare = forward_losses(model, make_bundle(cfg), batches[0], cfg, CPU)
    wrapped = Wrapper(model)
    assert unwrap(wrapped) is model and unwrap(model) is model
    reseed(8)
    out = forward_losses(wrapped, make_bundle(cfg), batches[0], cfg, CPU)
    assert wrapped.calls == 1  # the pooling backend goes through the wrapper
    assert torch.equal(out.loss, bare.loss)


def test_cpu_bfloat16_autocast_keeps_head_and_losses_in_fp32(setup):
    """Recipe side only (the source enables autocast on CUDA only)."""
    cfg, model, _, _, _, batches = setup
    reseed(9)
    ref = forward_losses(model, make_bundle(cfg), batches[1], cfg, CPU,
                         amp=False)
    reseed(9)
    out = forward_losses(model, make_bundle(cfg), batches[1], cfg, CPU,
                         amp=True)
    assert out.emb.dtype == torch.bfloat16
    assert out.logits.dtype == out.loss.dtype == torch.float32
    assert all(v is None or v.dtype == torch.float32
               for v in out.parts.values())
    assert bool(torch.isfinite(out.loss))
    assert out.loss.item() == pytest.approx(ref.loss.item(), rel=0.05)
    out.loss.backward()
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())


# ------------------------------------------------- tiny model vs the source
EPOCH = 2       # cur_iter = 3645 + i: past the warm-up, peak learning rate
STEP_SEED = 77

STEP_DRIVER = r'''
import logging, random
import torch.nn.functional as F
from wespeaker.utils import executor
from wespeaker.utils.pair_auc import pair_auc_center

model = build_model()
model.load_state_dict(torch.load(sys.argv[3]), strict=True)
batches = torch.load(sys.argv[4])
seed, epoch = int(sys.argv[5]), int(sys.argv[6])
ddp, optimizer, scheduler, margin, epoch_iter = build_optim(model)
configs['enable_amp'] = False  # CPU, fp32
steps, cur = [], {}


def clone(value):
    return None if value is None else value.detach().clone()


def grads():
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()}


class Criterion(torch.nn.CrossEntropyLoss):
    def forward(self, logits, labels):
        loss = super().forward(logits, labels)
        cur.update(ce=clone(loss), logits=clone(logits))
        return loss


real_pauc, real_cons = executor.pair_auc_loss, executor.pair_consistency_loss


def pauc(*args, **kwargs):
    res = real_pauc(*args, **kwargs)
    cur.update(pauc=clone(res[0]), pauc_stats=list(res[1:]))
    return res


def cons(*args, **kwargs):
    res = real_cons(*args, **kwargs)
    cur.update(cons=clone(res[0]))
    return res


executor.pair_auc_loss, executor.pair_consistency_loss = pauc, cons
real_backward = torch.Tensor.backward


def backward(tensor, *args, **kwargs):
    cur.update(loss=clone(tensor))
    res = real_backward(tensor, *args, **kwargs)
    cur.update(grads=grads())  # before clipping
    return res


torch.Tensor.backward = backward
optimizer.register_step_pre_hook(lambda opt, a, k: cur.update(
    clipped=grads(), lr=[g['lr'] for g in opt.param_groups],
    margin=margin.get_margin()))


def after_step(opt, a, k):
    cur.update(params={n: p.detach().clone()
                       for n, p in model.named_parameters()})
    steps.append(dict(cur))
    cur.clear()


optimizer.register_step_post_hook(after_step)


def loader():
    for i, batch in enumerate(batches):
        random.seed(seed + i)
        np.random.seed(seed + i)
        torch.manual_seed(seed + i)
        yield batch


executor.run_epoch(loader(), epoch_iter, ddp, Criterion(), optimizer,
                   scheduler, margin, epoch, logging.getLogger('source'),
                   torch.amp.GradScaler('cuda', enabled=False),
                   device=torch.device('cpu'), configs=configs)
center = pair_auc_center(configs)
torch.save({'tree': wespeaker.__file__, 'steps': steps,
            'center': [center.mean, center.n_upd, center.n_rows],
            'time_mask': list(model.frontend.time_mask_stats)}, sys.argv[2])
'''


def max_diff(mine, theirs):
    """(largest abs difference, largest relative-to-norm difference, all
    tensors bit-identical) over two dicts of tensors."""
    worst, worst_rel, same = 0.0, 0.0, True
    assert list(mine) == list(theirs)
    for name in mine:
        a, b = mine[name].double(), theirs[name].double()
        diff = float((a - b).abs().max())
        worst = max(worst, diff)
        worst_rel = max(worst_rel, diff / max(float(b.abs().max()), 1e-30))
        same = same and torch.equal(mine[name], theirs[name])
    return worst, worst_rel, same


@needs_source
def test_three_steps_equal_the_source_executor(tiny_assets, tmp_path):
    legacy, cfg = write_tiny_legacy(tmp_path, tiny_assets, center_warmup=1)
    torch.manual_seed(31)
    model = build_model(cfg, TINY_CLASSES).train()
    batches = tiny_batches(cfg.data.sampler.arm_order)
    state_file, batch_file = tmp_path / 'state.pt', tmp_path / 'batches.pt'
    torch.save(model.state_dict(), state_file)
    torch.save(batches, batch_file)
    src = run_source(STEP_DRIVER, tmp_path, legacy, state_file, batch_file,
                     STEP_SEED, EPOCH)
    assert len(src['steps']) == 3

    optimizer = build_optimizer(model, cfg)
    scheduler, margin = build_schedulers(optimizer, model, cfg)
    bundle = make_bundle(cfg)
    report = []
    for i, (batch, ref) in enumerate(zip(batches, src['steps'])):
        cur_iter = (EPOCH - 1) * 3645 + i
        # (1) fixed-batch entry: loss terms and gradients before clipping
        optimizer.zero_grad()  # the last step left its gradients behind
        reseed(STEP_SEED + i)
        out = forward_losses(model, copy.deepcopy(bundle), batch, cfg, CPU)
        out.loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        optimizer.zero_grad()
        g_abs, g_rel, g_same = max_diff(grads, ref['grads'])
        # (2) the training step itself
        reseed(STEP_SEED + i)
        res = train_step(model, bundle, batch, cfg, CPU, optimizer,
                         scheduler, margin, cur_iter)
        assert res.skipped is None
        assert torch.equal(res.loss, out.loss.detach())
        clipped = {n: p.grad.clone() for n, p in model.named_parameters()}
        c_abs, c_rel, c_same = max_diff(clipped, ref['clipped'])
        p_abs, p_rel, p_same = max_diff(params_of(model), ref['params'])
        report.append((i, float(res.loss), float(ref['loss']), g_abs, g_same,
                       c_abs, c_same, p_abs, p_same))

        assert [g['lr'] for g in optimizer.param_groups] == ref['lr']
        assert margin.get_margin() == ref['margin'] == 0.2
        assert torch.equal(res.loss, ref['loss'])
        assert torch.equal(res.parts['ce'], ref['ce'])
        assert torch.equal(res.logits, ref['logits'])
        assert (res.parts['cons'] is None) == (ref.get('cons') is None) \
            == (i != 1)
        if i == 1:
            assert torch.equal(res.parts['cons'], ref['cons'])
        assert torch.equal(res.parts['pauc'], ref['pauc'])
        assert [res.stats['pauc_pos'], res.stats['pauc_neg']] == \
            ref['pauc_stats'][:2]
        assert [res.stats['pauc_cos_pos'], res.stats['pauc_cos_neg']] == \
            ref['pauc_stats'][2:]
        want_norm = math.sqrt(sum(float(g.double().pow(2).sum())
                                  for g in ref['grads'].values()))
        assert float(res.grad_norm) == pytest.approx(want_norm, rel=1e-6)
        assert g_same and c_same and p_same, report[-1]
    mean, n_upd, n_rows = src['center']
    assert (bundle.center.n_upd, bundle.center.n_rows) == (n_upd, n_rows) \
        == (2, 6)
    assert torch.equal(bundle.center.mean, mean)
    # every batch went through the encoder twice here (entry + step)
    assert list(model.frontend.time_mask_stats) == \
        [2 * n for n in src['time_mask']]
    assert src['time_mask'][0] > 0  # frames were masked in these steps
    print('\ntiny-model step parity (step, loss, source loss, max|dgrad|, '
          'identical, max|dclipped|, identical, max|dparam|, identical):')
    for row in report:
        print('  %d %.9g %.9g %.3g %s %.3g %s %.3g %s' % row)


# ------------------------------------------- real model vs reference dump
REAL_STEP_SEED = 20261121
REAL_EPOCH_ITER = 3645


def top_groups(model):
    """Parameter groups of the reference dump: top-level module, and one
    group per encoder layer."""
    groups = {}
    for name, param in model.named_parameters():
        top = name.split('.')[0]
        groups.setdefault(top, []).append(param)
        match = re.search(r'\.layers\.(\d+)\.', name)
        if top == 'frontend' and match:
            groups.setdefault('frontend.layer_%02d' % int(match.group(1)),
                              []).append(param)
        elif top == 'frontend':
            groups.setdefault('frontend.other', []).append(param)
    return groups


def group_norm(tensors):
    return math.sqrt(sum(float(torch.linalg.vector_norm(t.detach().float()))
                         ** 2 for t in tensors))


@needs_real_model
def test_real_model_steps_equal_reference(monkeypatch):
    monkeypatch.chdir(SOURCE_RECIPE)
    ref_all = load_reference()
    ref, ref_sched = ref_all['model']['train'], ref_all['schedule']
    cfg = real_config()
    assert ref['epoch_iter'] == REAL_EPOCH_ITER
    model = build_real_model(cfg).train()
    optimizer = build_optimizer(model, cfg)
    scheduler, margin = build_schedulers(optimizer, model, cfg)
    # param groups of the real model
    want_groups = ref_sched['uninterrupted']['param_groups']
    for group, want in zip(optimizer.param_groups, want_groups):
        assert len(group['params']) == want['n_tensors']
        assert sum(p.numel() for p in group['params']) == want['n_elements']
        for key in ('betas', 'eps', 'weight_decay', 'amsgrad', 'fused',
                    '_is_frontend'):
            value = group[key]
            assert (list(value) if isinstance(value, tuple) else value) \
                == want[key], key
    assert [w['n_tensors'] for w in want_groups] == [708, 15]

    plain, pair = synth_batches(cfg.data.sampler.arm_order)
    groups = top_groups(model)
    watch = list(ref['plain']['weight_delta_norm'])
    bundle = make_bundle(cfg, warmup=ref['center_warmup_override'])
    lines = []

    def close(name, got, want, rel, abs_tol=0.0):
        lines.append('  %-34s %.9g vs %.9g (rel diff %.2g)' % (
            name, got, want, abs(got - want) / max(abs(want), 1e-30)))
        assert got == pytest.approx(want, rel=rel, abs=abs_tol), name

    # the configured warm-up (20): the first batch is scored uncentred
    reseed(REAL_STEP_SEED)
    with torch.no_grad():
        warm = forward_losses(model, make_bundle(cfg), plain, cfg, CPU)
    close('plain total, centre warm-up 20', float(warm.loss),
          ref['plain']['total_loss_config_warmup'], 1e-4)
    close('plain pair_auc uncentred', float(warm.parts['pauc']),
          ref['plain']['pair_auc_uncentred_config_warmup'], 1e-4, 1e-4)
    del warm
    mask_plain = list(model.frontend.time_mask_stats)

    for tag, batch, epoch, seed in (('plain', plain, 8, REAL_STEP_SEED),
                                    ('pair', pair, 9, REAL_STEP_SEED + 1)):
        want = ref[tag]
        before = {g: [p.detach().clone() for p in groups[g]] for g in watch}
        reseed(seed)
        res = train_step(model, bundle, batch, cfg, CPU, optimizer,
                         scheduler, margin, (epoch - 1) * REAL_EPOCH_ITER)
        assert res.skipped is None
        close(tag + ' total loss', float(res.loss), want['total_loss'], 1e-4)
        close(tag + ' ce', float(res.parts['ce']), want['ce_margin'], 1e-4)
        close(tag + ' pair_auc centred', float(res.parts['pauc']),
              want['pair_auc_centered'], 1e-4, 1e-4)
        if tag == 'pair':
            close(tag + ' consistency', float(res.parts['cons']),
                  want['consistency'], 1e-4, 1e-4)
        else:
            assert res.parts['cons'] is None
        assert [g['lr'] for g in optimizer.param_groups] == want['lr']
        # gradients: total norm before clipping, per-group norms after it
        total = float(res.grad_norm)
        want_total = math.sqrt(sum(
            v ** 2 for k, v in want['grad_norm'].items()
            if '#' not in k and '.' not in k))
        close(tag + ' grad norm before clipping', total, want_total, 1e-4)
        coef = min(1.0, cfg.optim.grad_clip / (total + 1e-6))
        close(tag + ' grad norm at the step', total * coef,
              want['grad_norm_total_at_step'], 1e-4)
        worst = 0.0
        for name, params in groups.items():
            got = group_norm([p.grad for p in params]) / coef
            assert got == pytest.approx(want['grad_norm'][name], rel=1e-4), \
                name
            worst = max(worst, abs(got / want['grad_norm'][name] - 1.0))
            assert want['grad_norm'][name + '#tensors_with_grad'] == \
                len(params)
        lines.append('  %-34s %d groups, worst rel diff %.2g' % (
            tag + ' per-group grad norms', len(groups), worst))
        worst = 0.0
        for name in watch:
            got = group_norm([b - p.detach() for b, p in
                              zip(before[name], groups[name])])
            assert got == pytest.approx(want['weight_delta_norm'][name],
                                        rel=1e-3), name
            worst = max(worst,
                        abs(got / want['weight_delta_norm'][name] - 1.0))
        lines.append('  %-34s %d groups, worst rel diff %.2g' % (
            tag + ' weight-change norms', len(watch), worst))
        del before
    # the reference forwarded each batch three times with the same seed
    # (two loss-only calls and the step), this test the plain batch twice
    mask_all = list(model.frontend.time_mask_stats)
    assert [3 * (a - b) for a, b in zip(mask_all, mask_plain)] == \
        ref['time_mask_stats']
    assert bundle.center.n_upd == 2
    print('\nreal-model steps vs reference dump:\n' + '\n'.join(lines))
