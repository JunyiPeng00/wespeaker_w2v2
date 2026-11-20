"""Pair-AUC loss and running centre: maths, state, guards, source parity."""
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

RECIPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RECIPE_DIR not in sys.path:
    sys.path.insert(0, RECIPE_DIR)

from ovsb.losses.pair_auc import RunningCenter, pair_auc_loss  # noqa: E402

torch.set_num_threads(4)
SOURCE_TREE = os.environ.get('OVSB_SOURCE_TREE')
needs_source = pytest.mark.skipif(
    not SOURCE_TREE, reason='OVSB_SOURCE_TREE is not set')

# Source side of the parity tests: loss cases, then a centre sequence.
SOURCE_DRIVER = r'''
import sys
import numpy as np
import torch
import wespeaker
from wespeaker.utils.pair_auc import PairAucCenter, pair_auc_loss

torch.set_num_threads(4)
data, out = np.load(sys.argv[1]), {'tree': np.array(wespeaker.__file__)}
for i in range(int(data['n_cases'])):
    x = torch.from_numpy(data['x%d' % i]).requires_grad_(True)
    role = torch.from_numpy(data['role%d' % i])
    label = torch.from_numpy(data['label%d' % i])
    loss, n_pos, n_neg, m_pos, m_neg = pair_auc_loss(
        x, label, float(data['scale']), role, None, 'both')
    out['stat%d' % i] = np.array([n_pos, n_neg, m_pos, m_neg], np.float64)
    if loss is None:
        continue
    loss.backward()
    out['loss%d' % i] = loss.detach().numpy()
    out['grad%d' % i] = x.grad.numpy()
center = PairAucCenter(int(data['c_arm']), float(data['c_momentum']),
                       int(data['c_warmup']))
for t in range(int(data['c_steps'])):
    x = torch.from_numpy(data['cx%d' % t])
    y = center(x, {'arm': torch.from_numpy(data['carm%d' % t])})
    out['cout%d' % t] = y.numpy()
    out['cready%d' % t] = np.array(center.ready())
    if center.mean is not None:
        out['cmean%d' % t] = center.mean.numpy()
out['c_counts'] = np.array([center.n_upd, center.n_rows])
np.savez(sys.argv[2], **out)
'''


def run_source(driver, tmp_path, inputs):
    """Run `driver` under the source tree; returns its saved arrays."""
    script, inp, out = (tmp_path / n for n in ('drv.py', 'in.npz', 'out.npz'))
    script.write_text(driver)
    np.savez(inp, **inputs)
    env = dict(os.environ, PYTHONPATH=SOURCE_TREE, PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, str(script), str(inp), str(out)],
                   check=True, env=env, cwd=str(tmp_path))
    res = dict(np.load(out))
    assert os.path.abspath(str(res.pop('tree'))).startswith(
        os.path.abspath(SOURCE_TREE))
    return res


def reference_loss(x, labels, scale, drop=()):
    """Direct transcription of the formula; `drop` = excluded (i, j)."""
    e = F.normalize(x, dim=1)
    pos, neg = [], []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if (i, j) in drop:
                continue
            (pos if labels[i] == labels[j] else neg).append((e[i] * e[j]).sum())
    pos, neg = torch.stack(pos), torch.stack(neg)
    return torch.log1p(torch.exp(scale * (neg[None] - pos[:, None])).sum(1)) \
        .mean(), len(pos), len(neg)


def test_formula_plain_batch():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(24, 16, generator=gen)
    labels = torch.arange(24) // 2
    loss, stats = pair_auc_loss(x, labels, 16.0)
    ref, n_pos, n_neg = reference_loss(x, labels, 16.0)
    assert (stats.n_pos, stats.n_neg) == (n_pos, n_neg) == (12, 264)
    assert torch.allclose(loss, ref, rtol=1e-5)
    same = pair_auc_loss(x, labels, 16.0, torch.zeros(24, dtype=torch.long))
    assert torch.equal(same[0], loss)   # all-zero roles change nothing


def test_same_utterance_couples_are_not_target_pairs():
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(16, 16, generator=gen)
    labels = (torch.arange(8) // 2).repeat_interleave(2)
    role = torch.tensor([1, 2] * 8)
    loss, stats = pair_auc_loss(x, labels, 16.0, role)
    drop = {(i, i + 1) for i in range(0, 16, 2)}
    ref, n_pos, n_neg = reference_loss(x, labels, 16.0, drop)
    assert (stats.n_pos, stats.n_neg) == (n_pos, n_neg) == (16, 96)
    assert torch.allclose(loss, ref, rtol=1e-5)
    full, full_stats = pair_auc_loss(x, labels, 16.0)
    assert full_stats.n_pos == 24 and not torch.equal(full, loss)


def test_gradient_reaches_both_pair_kinds():
    gen = torch.Generator().manual_seed(2)
    labels = torch.arange(24) // 2
    labels[-4:] = torch.tensor([100, 101, 102, 103])  # in no target pair
    x = torch.randn(24, 16, generator=gen, requires_grad=True)
    loss, _ = pair_auc_loss(x, labels, 16.0)
    grad, = torch.autograd.grad(loss, x)
    assert torch.isfinite(grad).all()
    assert grad[-4:].abs().max() > 0    # moved as nontarget rows only
    assert grad[:-4].abs().min() > 0


def test_no_term_without_both_pair_kinds():
    x = torch.randn(6, 8)
    loss, stats = pair_auc_loss(x, torch.arange(6), 16.0)
    assert loss is None and (stats.n_pos, stats.n_neg) == (0, 15)
    loss, stats = pair_auc_loss(x, torch.zeros(6, dtype=torch.long), 16.0)
    assert loss is None and (stats.n_pos, stats.n_neg) == (15, 0)


def test_center_ema_warmup_and_gradient():
    torch.manual_seed(0)
    n, dim, scale = 24, 16, 16.0
    center = RunningCenter(1, momentum=0.9, warmup=3)
    labels, arm = torch.arange(n) // 2, torch.arange(n) % 2
    offset, acc = torch.randn(dim) * 3.0, torch.zeros(dim)
    for t in range(5):
        x = (torch.randn(n, dim) + offset).requires_grad_(True)
        out = center(x, arm)
        acc = 0.9 * acc + 0.1 * x.detach()[arm == 1].mean(0)
        ref = acc / (1 - 0.9 ** (t + 1))
        assert torch.allclose(center.mean, ref, atol=1e-6)
        if t < 2:   # warm-up: the input itself comes back
            assert out is x and not center.ready
            continue
        assert center.ready and torch.allclose(out, x - ref, atol=1e-6)
        loss = pair_auc_loss(out, labels, scale)[0]
        detached = pair_auc_loss(x.detach() - ref, labels, scale)[0]
        uncentred = pair_auc_loss(x.detach(), labels, scale)[0]
        assert torch.allclose(loss, detached, atol=1e-6)
        assert abs(float(loss - uncentred)) > 1e-3
        grad, = torch.autograd.grad(loss, x)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
        assert not center.mean.requires_grad
    before = center.mean.clone()
    other_arm = torch.zeros(n, dtype=torch.long)
    assert not center.update(torch.randn(n, dim) + 100, other_arm)
    assert torch.equal(before, center.mean) and center.n_upd == 5
    assert center.n_rows == 5 * 12 and 'on' in center.describe()


def test_center_skips_non_finite_arm_mean():
    torch.manual_seed(1)
    center = RunningCenter(0, momentum=0.9, warmup=1)
    arm = torch.tensor([0, 0, 1, 1])
    center.update(torch.randn(4, 8), arm)
    mean, acc = center.mean.clone(), center.acc.clone()
    for bad in (float('nan'), float('inf'), float('-inf')):
        x = torch.randn(4, 8)
        x[1, 3] = bad
        assert not center.update(x, arm)
    assert center.n_skipped == 3 and center.n_upd == 1 and center.n_rows == 2
    assert torch.equal(center.mean, mean) and torch.equal(center.acc, acc)
    x = torch.randn(4, 8)
    x[2, 0] = float('nan')      # a row of another arm: not this object's job
    assert center.update(x, arm) and center.n_upd == 2
    assert torch.isfinite(center.mean).all() and center.n_skipped == 3
    fresh = RunningCenter(0)
    assert not fresh.update(torch.full((4, 8), float('nan')), arm)
    assert fresh.acc is None and fresh.mean is None and fresh.n_skipped == 1
    assert 'skipped' in fresh.describe()


def test_center_state_dict_round_trip():
    torch.manual_seed(2)
    arm = torch.arange(12) % 3
    batches = [torch.randn(12, 8) + 2.0 for _ in range(9)]
    full = RunningCenter(2, momentum=0.95, warmup=4)
    first = RunningCenter(2, momentum=0.95, warmup=4)
    empty = RunningCenter(2, momentum=0.95, warmup=4)
    empty.load_state_dict(first.state_dict())   # nothing seen yet
    assert empty.mean is None and empty.n_upd == 0
    for x in batches[:5]:
        full(x, arm)
        first(x, arm)
    first.update(torch.full((12, 8), float('nan')), arm)
    state = first.state_dict()
    resumed = RunningCenter(2, momentum=0.95, warmup=4)
    resumed.load_state_dict(state)
    assert resumed.ready and torch.equal(resumed.mean, first.mean)
    assert resumed.n_skipped == 1
    state['acc'].add_(1.0)                       # the state is a copy
    assert torch.equal(resumed.acc, first.acc)
    for x in batches[5:]:
        assert torch.equal(full(x, arm), resumed(x, arm))
    assert torch.equal(full.mean, resumed.mean)
    assert (full.n_upd, full.n_rows) == (resumed.n_upd, resumed.n_rows)
    with pytest.raises(ValueError):
        RunningCenter(2, momentum=0.9, warmup=4).load_state_dict(state)
    with pytest.raises(ValueError):
        RunningCenter(1, momentum=0.95, warmup=4).load_state_dict(state)


def test_center_rejects_bad_settings():
    for kwargs in ({'momentum': 1.0}, {'momentum': 0.0}, {'warmup': 0}):
        with pytest.raises(ValueError):
            RunningCenter(0, **kwargs)


def loss_cases(gen):
    """Plain, pair, singleton-heavy and degenerate batches, B = 48."""
    cases = []
    for i in range(12):
        x = torch.randn(48, 256, generator=gen) * (0.3 + i) \
            + torch.randn(256, generator=gen)
        if i % 3 == 0:    # plain batch, two rows per speaker
            label = torch.arange(24).repeat_interleave(2)
            role = torch.zeros(48, dtype=torch.long)
        elif i % 3 == 1:  # pair batch, two utterances per speaker
            label = torch.arange(12).repeat_interleave(4)
            role = torch.tensor([1, 2] * 24)
        else:             # plain rows followed by couples, some singletons
            label = torch.randint(0, 30, (48,), generator=gen)
            role = torch.tensor([0] * 16 + [1, 2] * 16)
            label[17::2] = label[16::2]
        cases.append((x, role, label))
    x = torch.randn(48, 256, generator=gen)
    cases.append((x, torch.zeros(48, dtype=torch.long), torch.arange(48)))
    cases.append((x, torch.tensor([1, 2] * 24),
                  torch.arange(24).repeat_interleave(2)))  # couples only
    return cases


@needs_source
def test_parity_with_source(tmp_path):
    gen = torch.Generator().manual_seed(4321)
    cases = loss_cases(gen)
    scale, arm_index, momentum, warmup, steps = 16.0, 3, 0.99, 20, 40
    inputs = {'n_cases': np.array(len(cases)), 'scale': np.array(scale),
              'c_arm': np.array(arm_index), 'c_momentum': np.array(momentum),
              'c_warmup': np.array(warmup), 'c_steps': np.array(steps)}
    for i, (x, role, label) in enumerate(cases):
        inputs.update({'x%d' % i: x.numpy(), 'role%d' % i: role.numpy(),
                       'label%d' % i: label.numpy()})
    offset = torch.randn(256, generator=gen) * 2.0
    seq = []
    for t in range(steps):
        x = torch.randn(48, 256, generator=gen) + offset
        arm = torch.randint(0, 8, (48,), generator=gen)
        if t in (5, 27):    # no row of the centre's arm
            arm[arm == arm_index] = 0
        seq.append((x, arm))
        inputs.update({'cx%d' % t: x.numpy(), 'carm%d' % t: arm.numpy()})
    src = run_source(SOURCE_DRIVER, tmp_path, inputs)

    n_none, grad_diff = 0, 0.0
    for i, (x, role, label) in enumerate(cases):
        x = x.clone().requires_grad_(True)
        loss, stats = pair_auc_loss(x, label, scale, role)
        assert np.array_equal(np.array(stats, np.float64), src['stat%d' % i])
        if loss is None:
            n_none += 1
            assert 'loss%d' % i not in src
            continue
        loss.backward()
        assert np.array_equal(loss.detach().numpy(), src['loss%d' % i])
        assert np.array_equal(x.grad.numpy(), src['grad%d' % i])
        grad_diff = max(grad_diff, float(
            np.abs(x.grad.numpy() - src['grad%d' % i]).max()))
    assert n_none == 2

    center = RunningCenter(arm_index, momentum, warmup)
    for t, (x, arm) in enumerate(seq):
        out = center(x, arm)
        assert center.ready == bool(src['cready%d' % t])
        assert np.array_equal(out.numpy(), src['cout%d' % t])
        assert np.array_equal(center.mean.numpy(), src['cmean%d' % t])
    assert center.ready and center.n_skipped == 0
    assert [center.n_upd, center.n_rows] == src['c_counts'].tolist()
    assert center.n_upd == steps - 2
    print('pair-auc parity: %d loss cases (%d without a term), %d centre '
          'steps, max abs diff %.3g (bit-identical)'
          % (len(cases), n_none, steps, grad_diff))
