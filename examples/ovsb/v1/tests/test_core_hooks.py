"""The two hooks added to the wespeaker core leave its behaviour unchanged.

* ``W2VBertSVFrontend._mask_frames`` is an identity extension point: the
  equal-length and the ragged forward give bit-identical outputs to the
  frontend of ``origin/master`` and the state-dict keys are the same.
* ``extract.py`` calls ``maybe_truncate_frontend``, which does nothing when
  ``frontend_num_layers`` is absent from the dataset config.

The frontend is built from a tiny randomly initialised Wav2Vec2-BERT saved
to a temporary directory, so no pretrained weights are needed.
"""

import ast
import importlib.util
import os
import subprocess

import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from conftest import REPO_ROOT  # noqa: E402

MASTER_REF = os.environ.get('OVSB_MASTER_REF', 'origin/master')
FRONTEND_FILE = 'wespeaker/frontend/w2vbert_sv.py'
EXTRACT_FILE = 'wespeaker/bin/extract.py'


def _git(*args):
    try:
        out = subprocess.run(('git', '-C', REPO_ROOT) + args, check=True,
                             capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip('git %s failed: %s' % (' '.join(args), exc))
    return out.stdout


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def tiny_assets(tmp_path_factory):
    """A 3-layer random Wav2Vec2-BERT in the layout the frontend loads."""
    path = tmp_path_factory.mktemp('tiny_w2vbert')
    torch.manual_seed(0)
    config = transformers.Wav2Vec2BertConfig(
        hidden_size=32, num_hidden_layers=3, num_attention_heads=2,
        intermediate_size=64, feature_projection_input_dim=160,
        conv_depthwise_kernel_size=5, left_max_position_embeddings=8,
        right_max_position_embeddings=4, add_adapter=False,
        vocab_size=8)
    transformers.Wav2Vec2BertModel(config).save_pretrained(str(path))
    return str(path)


@pytest.fixture(scope='module')
def frontends(tiny_assets, tmp_path_factory):
    """(frontend of the working tree, frontend of the master file)."""
    master_src = _git('show', '%s:%s' % (MASTER_REF, FRONTEND_FILE))
    master_py = tmp_path_factory.mktemp('master') / 'w2vbert_sv_master.py'
    master_py.write_text(master_src)
    master = _load_module('w2vbert_sv_master', str(master_py))
    from wespeaker.frontend import w2vbert_sv as current
    built = []
    for module in (current, master):
        frontend = module.W2VBertSVFrontend(
            upstream_args={'path_or_url': tiny_assets})
        built.append(frontend.eval())
    return built


def _wav(seed, batch, samples):
    gen = torch.Generator().manual_seed(seed)
    return 0.1 * torch.randn(batch, samples, generator=gen)


def test_state_dict_keys_and_values_unchanged(frontends):
    current, master = frontends
    cur, ref = current.state_dict(), master.state_dict()
    assert list(cur) == list(ref)
    for key in ref:
        assert torch.equal(cur[key], ref[key]), key


def test_constructor_signature_unchanged(frontends):
    import inspect
    current, master = frontends
    assert inspect.signature(type(current).__init__) == \
        inspect.signature(type(master).__init__)
    assert not hasattr(master, '_mask_frames')


@pytest.mark.parametrize('train', [False, True])
def test_forward_bit_identical(frontends, train):
    current, master = frontends
    wav = _wav(1, 3, 8000)
    outs = []
    for frontend in (current, master):
        frontend.train(train)
        torch.manual_seed(7)  # same dropout stream in training mode
        with torch.no_grad():
            feats, flens = frontend(wav, torch.full((3,), 8000))
        outs.append(feats)
        assert flens is None
        frontend.eval()
    assert outs[0].shape == (3, 32, outs[0].shape[2], 4)
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize('train', [False, True])
def test_ragged_forward_bit_identical(frontends, train):
    current, master = frontends
    wav = _wav(2, 3, 8000)
    lens = torch.tensor([8000, 5000, 2600])
    for row, n in enumerate(lens.tolist()):
        wav[row, n:] = 0.0
    outs = []
    for frontend in (current, master):
        frontend.train(train)
        torch.manual_seed(7)
        with torch.no_grad():
            outs.append(frontend(wav, lens))
        frontend.eval()
    assert outs[0][1] is not None
    assert torch.equal(outs[0][1], outs[1][1])
    assert torch.equal(outs[0][0], outs[1][0])


def test_mask_frames_is_called_and_is_identity(frontends, monkeypatch):
    current, _ = frontends
    h = torch.randn(2, 5, 32)
    assert current._mask_frames(h) is h
    assert current._mask_frames(h, torch.tensor([5, 3])) is h
    calls = []

    def record(h, flens=None):
        calls.append(None if flens is None else flens.tolist())
        return h

    monkeypatch.setattr(current, '_mask_frames', record)
    wav = _wav(3, 2, 6000)
    with torch.no_grad():
        current(wav)
        _, flens = current(wav, torch.tensor([6000, 3000]))
    assert calls == [None, flens.tolist()]


def test_truncation_is_noop_without_frontend_num_layers(frontends):
    from wespeaker.utils.frontend_utils import maybe_truncate_frontend
    current, _ = frontends
    model = torch.nn.Module()
    model.add_module('frontend', current)
    layers = list(current.upstream.encoder.layers)
    keys = list(model.state_dict())
    for dataset_args in ({}, {'frontend_num_layers': None},
                         {'frontend': 'w2vbert_sv', 'resample_rate': 16000}):
        assert maybe_truncate_frontend(model, dataset_args, 25) == 0
    assert list(current.upstream.encoder.layers) == layers
    assert all(a is b for a, b in zip(current.upstream.encoder.layers, layers))
    assert list(model.state_dict()) == keys
    assert current.n_hidden_states == 4


def test_truncation_applies_when_requested(tiny_assets):
    from wespeaker.frontend.w2vbert_sv import W2VBertSVFrontend
    from wespeaker.utils.frontend_utils import maybe_truncate_frontend
    model = torch.nn.Module()
    model.add_module('frontend', W2VBertSVFrontend(
        upstream_args={'path_or_url': tiny_assets}))
    assert maybe_truncate_frontend(model, {'frontend_num_layers': 2}, 3) == 1
    assert len(model.frontend.upstream.encoder.layers) == 2
    with torch.no_grad():
        feats, _ = model.frontend.eval()(_wav(4, 1, 6000))
    assert feats.shape[-1] == 3


def _truncation_calls(source):
    return [node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and getattr(node.func, 'id', None) == 'maybe_truncate_frontend']


def test_extract_calls_truncation_with_the_dataset_config():
    with open(os.path.join(REPO_ROOT, EXTRACT_FILE)) as fin:
        calls = _truncation_calls(fin.read())
    assert len(calls) == 1
    model_arg, conf_arg = calls[0].args[:2]
    assert (model_arg.id, conf_arg.id) == ('model', 'test_conf')
    master = _git('show', '%s:%s' % (MASTER_REF, EXTRACT_FILE))
    assert _truncation_calls(master) == []


def test_core_diff_is_additive_and_limited_to_the_two_hooks():
    stat = _git('diff', '--numstat', MASTER_REF, '--', 'wespeaker', 'tools')
    rows = [line.split('\t') for line in stat.splitlines() if line]
    assert sorted(row[2] for row in rows) == [EXTRACT_FILE, FRONTEND_FILE]
    assert all(row[1] == '0' for row in rows), rows
    assert sum(int(row[0]) for row in rows) <= 14
