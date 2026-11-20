"""local/init/truncate_ckpt.py on a synthetic checkpoint.

Real check (skipped unless set): ``OVSB_STAGE1_CKPT`` (last stage-1
checkpoint, 24 layers) and ``OVSB_REAL_INIT`` (a 22-layer checkpoint
derived from it): every tensor but the head must be bit-identical.
"""

import importlib.util
import json
import os

import pytest
import torch

RECIPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(RECIPE_DIR, 'local', 'init', 'truncate_ckpt.py')


def _load():
    spec = importlib.util.spec_from_file_location('truncate_ckpt', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tc = _load()
LAYERS, K, DIM = 6, 3, 4
SPEAKERS = ['id3', 'id1', 'vb2_x']


def _state():
    g = torch.Generator().manual_seed(0)
    state = {'frontend.upstream.feature_projection.weight':
             torch.randn(2, 2, generator=g)}
    for i in range(LAYERS):
        prefix = 'frontend.upstream.encoder.layers.%d.' % i
        state[prefix + 'ffn.weight'] = torch.randn(2, 2, generator=g)
        state[prefix + 'ffn.bias'] = torch.randn(2, generator=g)
    state['weights_k'] = torch.randn(LAYERS + 1, generator=g)
    state['weights_v'] = torch.randn(LAYERS + 1, generator=g)
    state['pool.weight'] = torch.randn(3, generator=g)
    state['projection.weight'] = torch.randn(K * len(SPEAKERS), DIM,
                                             generator=g)
    return state


def _label(tmp_path, speakers=SPEAKERS):
    path = tmp_path / 'utt2spk'
    path.write_text(''.join('%s/a/u%d %s\n' % (s, i, s)
                            for s in speakers for i in range(2)))
    return str(path)


def test_truncate_state():
    state = _state()
    out, dropped, mass = tc.truncate_state(state, 4)
    assert dropped == 2 * (LAYERS - 4)
    assert not any('layers.4.' in k or 'layers.5.' in k for k in out)
    kept = [k for k in state if 'layers.4.' not in k and 'layers.5.' not in k]
    assert list(out) == kept  # order kept
    for k in kept:
        if k in ('weights_k', 'weights_v'):
            assert torch.equal(out[k], state[k][:5])
        else:
            assert torch.equal(out[k], state[k])
    w = torch.softmax(state['weights_v'], -1)
    assert mass['weights_v'] == pytest.approx(float(w[:5].sum()))
    # softmax over the kept entries = the old weights renormalised
    assert torch.allclose(torch.softmax(out['weights_v'], -1),
                          w[:5] / w[:5].sum(), atol=1e-6)


def test_truncate_to_all_layers_is_identity():
    state = _state()
    out, dropped, _ = tc.truncate_state(state, LAYERS)
    assert dropped == 0 and list(out) == list(state)
    assert all(torch.equal(out[k], state[k]) for k in state)


@pytest.mark.parametrize('n', [0, LAYERS + 1])
def test_bad_layer_count(n):
    with pytest.raises(ValueError):
        tc.truncate_state(_state(), n)


def test_main_writes_checkpoint_and_sidecar(tmp_path):
    src = str(tmp_path / 'src.pt')
    out = str(tmp_path / 'sub' / 'out.pt')
    torch.save({'model': _state()}, src)  # wrapped checkpoints are accepted
    label = _label(tmp_path)
    assert tc.main([src, '4', out, '--train-label', label]) == 0
    got = torch.load(out, weights_only=False)
    assert got['weights_k'].shape == (5, )
    assert torch.equal(got['projection.weight'],
                       _state()['projection.weight'])
    with open(out + '.head.json') as f:
        side = json.load(f)
    assert side['num_classes'] == 3 and side['rows'] == 9
    assert side['rows_per_class'] == K and side['frontend_num_layers'] == 4
    from ovsb.model.head_record import verify_head
    verify_head(side, SPEAKERS, 9, 'sidecar')


def test_main_without_label_writes_no_sidecar(tmp_path):
    src, out = str(tmp_path / 'src.pt'), str(tmp_path / 'out.pt')
    torch.save(_state(), src)
    assert tc.main([src, '4', out]) == 0
    assert os.path.isfile(out) and not os.path.exists(out + '.head.json')


def test_main_refuses_another_class_list(tmp_path):
    src, out = str(tmp_path / 'src.pt'), str(tmp_path / 'out.pt')
    torch.save(_state(), src)
    label = _label(tmp_path, SPEAKERS + ['id9'])
    assert tc.main([src, '4', out, '--train-label', label]) == 1
    assert not os.path.exists(out)


def test_main_checks_the_source_sidecar(tmp_path):
    from ovsb.model.head_record import HeadMismatchError, write_head_sidecar
    src, out = str(tmp_path / 'src.pt'), str(tmp_path / 'out.pt')
    torch.save(_state(), src)
    write_head_sidecar(src, ['a', 'b', 'c'], 9)  # same size, other speakers
    with pytest.raises(HeadMismatchError):
        tc.main([src, '4', out, '--train-label', _label(tmp_path)])


@pytest.mark.skipif(not all(os.environ.get(k) for k in (
    'OVSB_STAGE1_CKPT', 'OVSB_REAL_INIT')),
    reason='OVSB_STAGE1_CKPT / OVSB_REAL_INIT not set')
def test_real_truncation_equals_reference():
    src = tc.unwrap(torch.load(os.environ['OVSB_STAGE1_CKPT'],
                               map_location='cpu', weights_only=False,
                               mmap=True))
    real = tc.unwrap(torch.load(os.environ['OVSB_REAL_INIT'],
                                map_location='cpu', weights_only=False,
                                mmap=True))
    out, _, _ = tc.truncate_state(src, 22)
    assert list(out) == list(real)
    differing = [k for k in out if k != tc.HEAD_KEY
                 and not torch.equal(out[k], real[k])]
    assert not differing
