"""Duration-margin head: maths, checkpoint keys, clamp, source parity."""
import math
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

from ovsb.model.head import (COS_EPS, DurationMarginSubcenterHead,  # noqa: E402
                             margin_scale_from_len)
from wespeaker.models.projections import (  # noqa: E402
    ArcMarginProduct_intertopk_subcenter)

torch.set_num_threads(4)
SOURCE_TREE = os.environ.get('OVSB_SOURCE_TREE')
needs_source = pytest.mark.skipif(
    not SOURCE_TREE, reason='OVSB_SOURCE_TREE is not set')
DIM, CLASSES, K, REF_SEC, SR = 64, 50, 3, 3.0, 16000

# Source side: the head as the source trainer builds and calls it
# (duration_margin set, margin_scale from the row lengths, no second label).
SOURCE_DRIVER = r'''
import sys
import numpy as np
import torch
import torch.nn.functional as F
import wespeaker
from wespeaker.models.projections import get_projection
from wespeaker.utils.phase1 import duration_margin_scale

torch.set_num_threads(4)
data, out = np.load(sys.argv[1]), {'tree': np.array(wespeaker.__file__)}
for i in range(int(data['n_cases'])):
    x = torch.from_numpy(data['x%d' % i]).requires_grad_(True)
    w = torch.from_numpy(data['w%d' % i])
    head = get_projection({
        'project_type': 'arc_margin_intertopk_subcenter',
        'embed_dim': w.shape[1], 'num_class': int(data['classes']),
        'scale': 32.0, 'easy_margin': bool(data['easy%d' % i]),
        'K': int(data['k']), 'mp': 0.06, 'k_top': 5, 'do_lm': False,
        'duration_margin': {'ref_sec': float(data['ref_sec'])}})
    with torch.no_grad():
        head.weight.copy_(w)
    head.update(float(data['margin%d' % i]))
    label = torch.from_numpy(data['label%d' % i])
    scale = duration_margin_scale(
        head, torch.from_numpy(data['len%d' % i]), int(data['sr']))
    logits = head(x.float(), label, margin_scale=scale)
    F.cross_entropy(logits, label).backward()
    out['scale%d' % i] = scale.numpy()
    out['logits%d' % i] = logits.detach().numpy()
    out['gx%d' % i] = x.grad.numpy()
    out['gw%d' % i] = head.weight.grad.numpy()
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


def make_head(weight, margin=0.2, easy_margin=False):
    head = DurationMarginSubcenterHead(
        weight.shape[1], weight.shape[0] // K, scale=32.0, margin=0.0,
        easy_margin=easy_margin, K=K, mp=0.06, k_top=5)
    with torch.no_grad():
        head.weight.copy_(weight)
    head.update(margin)
    return head


def class_cosine(x, weight):
    """Largest sub-centre cosine per class, as the head computes it."""
    cos = F.linear(F.normalize(x), F.normalize(weight))
    return cos.reshape(len(x), -1, K).max(2)[0]


def run_new(x, weight, label, wav_len, margin=0.2, easy_margin=False):
    """(logits, grad x, grad weight) of CE on the head's logits."""
    head = make_head(weight, margin, easy_margin)
    x = x.clone().requires_grad_(True)
    logits = head(x, label, margin_scale_from_len(wav_len, REF_SEC, SR))
    F.cross_entropy(logits, label).backward()
    return logits.detach(), x.grad, head.weight.grad


def unit_cosine_case(which):
    """A batch with one cosine of exactly 1.0, by construction.

    which: 'target' (the row's own class), 'class0' (class 0 of a row with
    another label) or 'other' (some other non-target class).
    """
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(8, DIM, generator=gen)
    weight = torch.randn(CLASSES * K, DIM, generator=gen)
    label = torch.arange(8) + 10
    cls = {'target': 12, 'class0': 0, 'other': 30}[which]
    x[2] = 0.0
    x[2, 5] = 2.0           # normalises to the basis vector e_5 exactly
    weight[cls * K + 1] = 0.0
    weight[cls * K + 1, 5] = 0.5
    wav_len = torch.full((8,), 2 * SR)
    assert float(class_cosine(x, weight)[2, cls]) == 1.0
    return x, weight, label, wav_len


def test_margin_scale_from_len():
    wav_len = torch.tensor([0, 8000, 24000, 48000, 96000, 48001])
    scale = margin_scale_from_len(wav_len, 3.0, 16000)
    assert scale.dtype == torch.float32
    assert scale.tolist() == pytest.approx([0, 1 / 6, 0.5, 1, 1, 1], abs=1e-7)
    assert margin_scale_from_len(torch.tensor([8000]), 2.0, 8000).item() == 0.5


def test_state_dict_matches_core_class():
    core = ArcMarginProduct_intertopk_subcenter(DIM, CLASSES, K=K)
    head = DurationMarginSubcenterHead(DIM, CLASSES, K=K)
    assert list(head.state_dict()) == list(core.state_dict()) == ['weight']
    assert head.weight.shape == core.weight.shape == (CLASSES * K, DIM)
    head.load_state_dict(core.state_dict(), strict=True)
    core.load_state_dict(head.state_dict(), strict=True)
    holder = torch.nn.Module()
    holder.add_module('projection', head)
    assert list(holder.state_dict()) == ['projection.weight']


def test_without_scale_is_the_core_forward():
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(16, DIM, generator=gen)
    label = torch.randint(0, CLASSES, (16,), generator=gen)
    head = make_head(torch.randn(CLASSES * K, DIM, generator=gen))
    core = ArcMarginProduct_intertopk_subcenter(
        DIM, CLASSES, scale=32.0, margin=0.0, K=K, mp=0.06, k_top=5)
    core.load_state_dict(head.state_dict())
    core.update(0.2)
    assert torch.equal(head(x, label), core(x, label))
    # full-length rows get the full margin: the core logits up to rounding
    ones = head(x, label, torch.ones(16))
    assert torch.allclose(ones, core(x, label), atol=1e-4)


def test_per_row_margin_formula():
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(32, DIM, generator=gen)
    weight = torch.randn(CLASSES * K, DIM, generator=gen)
    label = torch.randint(0, CLASSES, (32,), generator=gen)
    g = torch.rand(32, generator=gen)
    head = make_head(weight, margin=0.2)
    logits = head(x, label, g) / 32.0
    cos = class_cosine(x, weight)
    rows = torch.arange(32)
    theta = torch.acos(cos[rows, label])
    assert torch.allclose(logits[rows, label], torch.cos(theta + 0.2 * g),
                          atol=1e-5)
    rest = cos.clone()
    rest[rows, label] = -2.0
    top = rest.topk(5, dim=1)
    assert torch.allclose(logits.gather(1, top.indices),
                          torch.cos(torch.acos(top.values) - 0.06), atol=1e-5)
    untouched = torch.ones_like(cos, dtype=torch.bool)
    untouched[rows, label] = False
    untouched.scatter_(1, top.indices, False)
    assert torch.equal(logits[untouched], (cos * 32.0)[untouched] / 32.0)
    # a zero scale removes the target margin (the logit is cos itself)
    plain = head(x, label, torch.zeros(32)) / 32.0
    assert torch.allclose(plain[rows, label], cos[rows, label], atol=1e-6)


def test_wrapped_branch_below_the_margin_edge():
    """c <= cos(pi - m_i): the logit continues as c - (1 + cos(pi - m_i))."""
    weight = torch.randn(CLASSES * K, DIM)
    x = torch.randn(4, DIM)
    label = torch.tensor([3, 4, 5, 6])
    for row, cls in enumerate(label.tolist()):
        weight[cls * K:(cls + 1) * K] = -x[row] + 0.01 * torch.randn(K, DIM)
    g = torch.tensor([1.0, 0.75, 0.5, 0.25])
    logits = make_head(weight, margin=0.2)(x, label, g) / 32.0
    c = class_cosine(x, weight)[torch.arange(4), label]
    edge = torch.cos(math.pi - 0.2 * g)
    assert bool((c <= edge).all())
    assert torch.allclose(logits[torch.arange(4), label], c - (1.0 + edge),
                          atol=1e-6)


@pytest.mark.parametrize('which', ['target', 'class0', 'other'])
def test_unit_cosine_has_finite_gradients(which):
    x, weight, label, wav_len = unit_cosine_case(which)
    logits, grad_x, grad_w = run_new(x, weight, label, wav_len)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(grad_x).all() and torch.isfinite(grad_w).all()
    assert grad_x.abs().sum() > 0 and grad_w.abs().sum() > 0
    assert float(logits.max()) <= 32.0 * (1.0 - COS_EPS)


def parity_cases(gen, n_seeds):
    """Random batches, plus rows near both ends of the cosine range."""
    cases = []
    for i in range(n_seeds):
        x = torch.randn(48, DIM, generator=gen) * (0.2 + i % 7)
        weight = torch.randn(CLASSES * K, DIM, generator=gen)
        label = torch.randint(0, CLASSES, (48,), generator=gen)
        wav_len = torch.randint(SR // 2, 8 * SR, (48,), generator=gen)
        if i % 4 == 1:   # targets close to +1 (below the clamp)
            rows = torch.arange(0, 48, 3)
            x[rows] = weight[label[rows] * K + 1] \
                + 2e-2 * torch.randn(len(rows), DIM, generator=gen)
        if i % 4 == 2:   # targets close to -1: the wrapped branch
            for row in range(0, 48, 5):
                cls = int(label[row])
                weight[cls * K:(cls + 1) * K] = -x[row] \
                    + 2e-2 * torch.randn(K, DIM, generator=gen)
        if i % 4 == 3:   # hard negatives close to +1
            rows = torch.arange(1, 48, 4)
            x[rows] = weight[((label[rows] + 1) % CLASSES) * K] \
                + 2e-2 * torch.randn(len(rows), DIM, generator=gen)
        cases.append(dict(x=x, w=weight, label=label, len=wav_len,
                          margin=(0.2, 0.2, 0.2, 0.1, 0.35)[i % 5],
                          easy=(i % 10 == 9)))
    return cases


def pack(cases):
    inputs = {'n_cases': np.array(len(cases)), 'classes': np.array(CLASSES),
              'k': np.array(K), 'ref_sec': np.array(REF_SEC),
              'sr': np.array(SR)}
    for i, case in enumerate(cases):
        inputs.update({'%s%d' % (k, i): np.asarray(v) for k, v in case.items()})
    return inputs


@needs_source
def test_parity_with_source(tmp_path):
    gen = torch.Generator().manual_seed(99)
    cases = parity_cases(gen, 40)
    src = run_source(SOURCE_DRIVER, tmp_path, pack(cases))
    limit = float(torch.tensor(1.0 - COS_EPS))   # fp32 clamp bound
    worst = {'logits': 0.0, 'gx': 0.0, 'gw': 0.0}
    top_cos, low_cos, n_wrapped = 0.0, 0.0, 0
    for i, case in enumerate(cases):
        cos = class_cosine(case['x'], case['w'])
        assert float(cos.abs().max()) < limit    # the clamp is inactive
        top_cos = max(top_cos, float(cos.max()))
        low_cos = min(low_cos, float(cos.min()))
        target = cos.gather(1, case['label'][:, None])
        n_wrapped += int((target <= math.cos(math.pi - case['margin'])).sum())
        scale = margin_scale_from_len(case['len'], REF_SEC, SR)
        assert np.array_equal(scale.numpy(), src['scale%d' % i])
        new = dict(zip(('logits', 'gx', 'gw'), run_new(
            case['x'], case['w'], case['label'], case['len'],
            case['margin'], case['easy'])))
        for name, value in new.items():
            ref = src['%s%d' % (name, i)]
            assert np.isfinite(ref).all()
            worst[name] = max(worst[name],
                              float(np.abs(value.numpy() - ref).max()))
            assert np.array_equal(value.numpy(), ref), (i, name, worst)
    assert top_cos > 0.999 and low_cos < -0.999 and n_wrapped > 0
    print('head parity: %d seeds, cos range [%.6f, %.6f], %d wrapped '
          'targets, max abs diff logits %.3g grad-emb %.3g grad-weight %.3g'
          % (len(cases), low_cos, top_cos, n_wrapped, worst['logits'],
             worst['gx'], worst['gw']))


@needs_source
def test_source_gradient_is_nan_at_unit_cosine(tmp_path):
    """Documents the defect: one cosine of exactly 1 anywhere in the batch
    makes the source gradients NaN (d sqrt(1 - c^2) at c = 1, multiplied by
    a zero), while its logits stay finite. The port stays finite."""
    kinds = ('target', 'class0', 'other')
    cases = []
    for which in kinds:
        x, weight, label, wav_len = unit_cosine_case(which)
        cases.append(dict(x=x, w=weight, label=label, len=wav_len,
                          margin=0.2, easy=False))
    src = run_source(SOURCE_DRIVER, tmp_path, pack(cases))
    for i, (which, case) in enumerate(zip(kinds, cases)):
        assert np.isfinite(src['logits%d' % i]).all()
        assert np.isnan(src['gx%d' % i]).any(), which
        assert np.isnan(src['gw%d' % i]).any(), which
        logits, grad_x, grad_w = run_new(case['x'], case['w'], case['label'],
                                         case['len'])
        assert torch.isfinite(grad_x).all() and torch.isfinite(grad_w).all()
        # away from the clamped entry the logits are the source's
        diff = np.abs(logits.numpy() - src['logits%d' % i])
        hit = diff > 0
        assert hit.sum() <= 1 and diff.max() < 1e-2
        print('unit cosine (%s): source NaN grads %d emb / %d weight entries;'
              ' port finite; logits differ in %d entry (max %.3g)'
              % (which, np.isnan(src['gx%d' % i]).sum(),
                 np.isnan(src['gw%d' % i]).sum(), hit.sum(), diff.max()))
