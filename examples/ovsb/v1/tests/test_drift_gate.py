"""local/drift_gate.py: value, exit codes, fail-closed behaviour.

Real-value check (skipped unless set): ``OVSB_DRIFT_MODEL`` (a checkpoint),
``OVSB_DRIFT_REF`` (its reference) and ``OVSB_DRIFT_EXPECT`` (encoder drift
printed with 5 decimals, e.g. 0.04429).
"""

import importlib.util
import os
import subprocess
import sys

import pytest
import torch

RECIPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(RECIPE_DIR, 'local', 'drift_gate.py')


def _load():
    spec = importlib.util.spec_from_file_location('drift_gate', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dg = _load()


def _state(seed=0, scale=0.0):
    g = torch.Generator().manual_seed(seed)
    ref = {
        'frontend.a.weight': torch.randn(4, 3, generator=g),
        'frontend.b.bias': torch.randn(5, generator=g),
        'weights_k': torch.randn(3, generator=g),
        'pool.weight': torch.randn(2, 2, generator=g),
        'frontend.steps': torch.tensor(7),  # not floating point
        'projection.weight': torch.randn(6, 2, generator=g),
    }
    if not scale:
        return ref
    out = {k: v.clone() for k, v in ref.items()}
    for k in ('frontend.a.weight', 'frontend.b.bias', 'weights_k',
              'pool.weight'):
        out[k] = out[k] * (1 + scale)
    out['projection.weight'] = out['projection.weight'] * 5  # ignored
    return out


def _save(tmp_path, name, state, wrap=None):
    path = str(tmp_path / name)
    torch.save({wrap: state} if wrap else state, path)
    return path


def test_value_matches_definition():
    ref, model = _state(), _state(scale=0.05)
    d = dg.drift(ref, model)
    # theta = 1.05 * theta_ref in both groups -> relative drift 0.05
    assert d['encoder'] == pytest.approx(0.05, rel=1e-5)
    assert d['mhfa'] == pytest.approx(0.05, rel=1e-5)
    num = sum(float(((model[k] - ref[k]) ** 2).sum())
              for k in ('frontend.a.weight', 'frontend.b.bias'))
    den = sum(float((ref[k] ** 2).sum())
              for k in ('frontend.a.weight', 'frontend.b.bias'))
    assert d['encoder'] == (num / den) ** 0.5  # same summation, same value


def test_head_and_integer_tensors_are_ignored():
    ref, model = _state(), _state()
    model['projection.weight'] = torch.zeros(99, 2)  # other class list
    model['frontend.steps'] = torch.tensor(8)
    d = dg.drift(ref, model)
    assert d == {'encoder': 0.0, 'mhfa': 0.0}


def test_reference_only_tensors_do_not_count():
    ref, model = _state(), _state(scale=0.05)
    ref['frontend.layers.23.weight'] = torch.ones(3)  # deeper reference
    assert dg.drift(ref, model)['encoder'] == pytest.approx(0.05, rel=1e-5)


@pytest.mark.parametrize('damage', ['missing', 'shape', 'dtype'])
def test_incomparable_model_tensor_is_an_error(damage):
    ref, model = _state(), _state(scale=0.01)
    if damage == 'missing':
        del ref['frontend.b.bias']
    elif damage == 'shape':
        ref['frontend.b.bias'] = torch.zeros(6)
    else:
        ref['frontend.b.bias'] = torch.zeros(5, dtype=torch.long)
    with pytest.raises(dg.DriftError, match='frontend.b.bias'):
        dg.drift(ref, model)


def test_empty_group_is_an_error():
    ref = {'frontend.a.weight': torch.ones(2)}
    with pytest.raises(dg.DriftError, match='mhfa'):
        dg.drift(ref, {'frontend.a.weight': torch.ones(2)})


def test_exit_codes_single_model(tmp_path):
    ref = _save(tmp_path, 'ref.pt', _state())
    model = _save(tmp_path, 'm.pt', _state(scale=0.05), wrap='model')
    assert dg.main(['--model', model, '--ref', ref, '--gate', '0.08']) == 0
    assert dg.main(['--model', model, '--ref', ref, '--gate', '0.02']) == 3


def test_missing_or_unreadable_reference_fails_closed(tmp_path):
    model = _save(tmp_path, 'm.pt', _state())
    assert dg.main(['--model', model, '--ref', str(tmp_path / 'no.pt')]) == 2
    garbage = tmp_path / 'garbage.pt'
    garbage.write_bytes(b'not a checkpoint')
    assert dg.main(['--model', model, '--ref', str(garbage)]) == 2
    not_state = _save(tmp_path, 'list.pt', [1, 2, 3])
    assert dg.main(['--model', model, '--ref', not_state]) == 2


def test_missing_tensor_exit_code(tmp_path):
    ref_state = _state()
    del ref_state['pool.weight']
    ref = _save(tmp_path, 'ref.pt', ref_state)
    model = _save(tmp_path, 'm.pt', _state())
    assert dg.main(['--model', model, '--ref', ref, '--gate', '0.5']) == 2


def test_exp_mode_appends_and_latches(tmp_path):
    exp = tmp_path / 'exp'
    (exp / 'models').mkdir(parents=True)
    ref = _save(tmp_path, 'ref.pt', _state())
    torch.save(_state(), str(exp / 'models' / 'model_0.pt'))
    torch.save(_state(scale=0.01), str(exp / 'models' / 'model_1.pt'))
    torch.save(_state(scale=0.01), str(exp / 'models' / 'partial_model.pt'))
    args = ['--exp', str(exp), '--ref', ref, '--gate', '0.02']
    assert dg.main(args) == 0
    table = (exp / 'drift.tsv').read_text().splitlines()
    assert table == ['epoch\tencoder\tmhfa\tgate',
                     '1\t0.01000\t0.01000\tok']
    assert dg.main(args) == 0  # nothing new, nothing appended
    assert len((exp / 'drift.tsv').read_text().splitlines()) == 2
    torch.save(_state(scale=0.05), str(exp / 'models' / 'model_2.pt'))
    torch.save(_state(scale=0.01), str(exp / 'models' / 'model_10.pt'))
    assert dg.main(args) == 3
    rows = (exp / 'drift.tsv').read_text().splitlines()
    assert [r.split('\t')[0] for r in rows[1:]] == ['1', '2', '10']
    assert rows[2].endswith('EXCEEDED') and rows[3].endswith('ok')
    assert dg.main(args) == 3  # an exceeded epoch keeps the gate closed


def test_exp_mode_error_does_not_pass(tmp_path):
    exp = tmp_path / 'exp'
    (exp / 'models').mkdir(parents=True)
    ref = _save(tmp_path, 'ref.pt', _state())
    bad = _state()
    bad['frontend.new.weight'] = torch.ones(2)  # not in the reference
    torch.save(bad, str(exp / 'models' / 'model_1.pt'))
    assert dg.main(['--exp', str(exp), '--ref', ref]) == 2
    assert dg.main(['--exp', str(tmp_path / 'none'), '--ref', ref]) == 2


def test_process_exit_codes(tmp_path):
    ref = _save(tmp_path, 'ref.pt', _state())
    model = _save(tmp_path, 'm.pt', _state(scale=0.05))

    def run(*extra):
        return subprocess.run([sys.executable, SCRIPT, '--model', model,
                               *extra], capture_output=True, text=True)
    ok = run('--ref', ref, '--gate', '0.08')
    assert ok.returncode == 0 and 'encoder\t0.05000' in ok.stdout
    assert run('--ref', ref, '--gate', '0.02').returncode == 3
    missing = run('--ref', str(tmp_path / 'no.pt'))
    assert missing.returncode not in (0, 3)
    assert 'DRIFT GATE ERROR' in missing.stderr
    assert run().returncode not in (0, 3)  # no --ref


@pytest.mark.skipif(not all(os.environ.get(k) for k in (
    'OVSB_DRIFT_MODEL', 'OVSB_DRIFT_REF', 'OVSB_DRIFT_EXPECT')),
    reason='OVSB_DRIFT_MODEL / OVSB_DRIFT_REF / OVSB_DRIFT_EXPECT not set')
def test_real_value():
    d = dg.drift(dg.load_state(os.environ['OVSB_DRIFT_REF']),
                 dg.load_state(os.environ['OVSB_DRIFT_MODEL']))
    assert '%.5f' % d['encoder'] == os.environ['OVSB_DRIFT_EXPECT']
