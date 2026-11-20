"""Consistency loss: formula, stop-gradient, plain batches, source parity."""
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

from ovsb.losses.consistency import pair_consistency_loss  # noqa: E402

torch.set_num_threads(4)
SOURCE_TREE = os.environ.get('OVSB_SOURCE_TREE')
needs_source = pytest.mark.skipif(
    not SOURCE_TREE, reason='OVSB_SOURCE_TREE is not set')

# Runs the source implementation on the saved inputs (own interpreter, so
# the two trees' `wespeaker` packages never meet in one process).
SOURCE_DRIVER = r'''
import sys
import numpy as np
import torch
import torch.nn.functional as F
import wespeaker
from wespeaker.utils.pair_cons import pair_consistency_loss

torch.set_num_threads(4)
data, out = np.load(sys.argv[1]), {'tree': np.array(wespeaker.__file__)}
for i in range(int(data['n_cases'])):
    x = torch.from_numpy(data['x%d' % i]).requires_grad_(True)
    role = torch.from_numpy(data['role%d' % i])
    label = torch.from_numpy(data['label%d' % i])
    loss, n, _ = pair_consistency_loss(
        F.normalize(x.float(), dim=1), role, label, 'same')
    out['n%d' % i] = np.array(n)
    if loss is None:
        continue
    loss.backward()
    out['loss%d' % i] = loss.detach().numpy()
    out['grad%d' % i] = x.grad.numpy()
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


def pair_batch(gen, n_couples, dim):
    """(x, role, label): couples of one speaker pair per two utterances."""
    x = torch.randn(2 * n_couples, dim, generator=gen)
    role = torch.tensor([1, 2] * n_couples)
    label = torch.arange(n_couples).div(2, rounding_mode='floor') \
        .repeat_interleave(2)
    return x, role, label


def test_formula_and_stop_gradient():
    gen = torch.Generator().manual_seed(0)
    x, role, _ = pair_batch(gen, 8, 16)
    x.requires_grad_(True)
    e = F.normalize(x, dim=1)
    a = torch.nonzero(role == 1).squeeze(1)
    ref = (1.0 - (e[a + 1] * e[a].detach()).sum(1)).mean()
    loss, n = pair_consistency_loss(e, role)
    assert torch.equal(loss, ref) and n == 8
    grad, = torch.autograd.grad(loss, x)
    assert grad[a].abs().max() == 0       # anchors: no gradient
    assert grad[a + 1].abs().min() > 0    # every view moves


def test_identical_views_give_zero():
    e = F.normalize(torch.randn(6, 8), dim=1)
    e = e.repeat_interleave(2, dim=0)
    loss, n = pair_consistency_loss(e, torch.tensor([1, 2] * 6))
    assert n == 6 and abs(float(loss)) < 1e-6


def test_plain_batch_has_no_term():
    e = F.normalize(torch.randn(16, 8), dim=1)
    loss, n = pair_consistency_loss(e, torch.zeros(16, dtype=torch.long))
    assert loss is None and n == 0


def test_mixed_rows_use_only_the_couples():
    gen = torch.Generator().manual_seed(1)
    e = F.normalize(torch.randn(10, 8, generator=gen), dim=1)
    role = torch.tensor([0, 1, 2, 0, 0, 1, 2, 0, 1, 2])
    loss, n = pair_consistency_loss(e, role)
    ref = sum(1.0 - (e[v] * e[v - 1]).sum() for v in (2, 6, 9)) / 3
    assert n == 3 and torch.allclose(loss, ref, atol=1e-7)


@needs_source
def test_parity_with_source(tmp_path):
    gen = torch.Generator().manual_seed(1234)
    cases = []
    for i in range(12):
        x, role, label = pair_batch(gen, 24, 256)
        if i % 4 == 3:  # a plain batch: the term is absent
            role = torch.zeros_like(role)
        cases.append((x * (0.5 + i), role, label))
    inputs = {'n_cases': np.array(len(cases))}
    for i, (x, role, label) in enumerate(cases):
        inputs.update({'x%d' % i: x.numpy(), 'role%d' % i: role.numpy(),
                       'label%d' % i: label.numpy()})
    src = run_source(SOURCE_DRIVER, tmp_path, inputs)
    worst = 0.0
    for i, (x, role, _) in enumerate(cases):
        x = x.clone().requires_grad_(True)
        loss, n = pair_consistency_loss(F.normalize(x.float(), dim=1), role)
        assert n == int(src['n%d' % i])
        if loss is None:
            assert 'loss%d' % i not in src
            continue
        loss.backward()
        assert np.array_equal(loss.detach().numpy(), src['loss%d' % i])
        assert np.array_equal(x.grad.numpy(), src['grad%d' % i])
        worst = max(worst, float(np.abs(x.grad.numpy()
                                        - src['grad%d' % i]).max()))
    print('consistency parity: %d cases, max abs diff %.3g'
          % (len(cases), worst))
