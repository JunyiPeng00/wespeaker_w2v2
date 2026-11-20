"""Time masking of the frontend: rate, spans, padding, hook, source parity."""
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

RECIPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RECIPE_DIR not in sys.path:
    sys.path.insert(0, RECIPE_DIR)

from ovsb.model.frontend import (TimeMaskedW2VBertSV,  # noqa: E402
                                 draw_time_mask)
from wespeaker.frontend.w2vbert_sv import W2VBertSVFrontend  # noqa: E402

torch.set_num_threads(4)
SOURCE_TREE = os.environ.get('OVSB_SOURCE_TREE')
needs_source = pytest.mark.skipif(
    not SOURCE_TREE, reason='OVSB_SOURCE_TREE is not set')
HAS_CORE_HOOK = hasattr(W2VBertSVFrontend, '_mask_frames') and \
    '_mask_frames' in inspect.getsource(W2VBertSVFrontend.forward)

# Source side: its masking method on a stand-in object (no encoder needed),
# several calls per seed, then one more draw to pin the generator position.
SOURCE_DRIVER = r'''
import sys
import numpy as np
import torch
import wespeaker
from wespeaker.frontend.w2vbert_sv import W2VBertSVFrontend as Frontend

torch.set_num_threads(4)


class Dummy:
    _time_mask = Frontend._time_mask
    time_mask_stats = Frontend.time_mask_stats

    def __init__(self, prob, span):
        self.time_mask_prob, self.time_mask_span = prob, span
        self.training, self._tm_acc = True, None


data, out = np.load(sys.argv[1]), {'tree': np.array(wespeaker.__file__)}
for i in range(int(data['n_cases'])):
    d = Dummy(float(data['prob%d' % i]), int(data['span%d' % i]))
    torch.manual_seed(int(data['seed%d' % i]))
    for j in range(int(data['calls%d' % i])):
        h = torch.from_numpy(data['h%d_%d' % (i, j)])
        key = 'flens%d_%d' % (i, j)
        flens = torch.from_numpy(data[key]) if key in data else None
        out['o%d_%d' % (i, j)] = d._time_mask(h, flens).numpy()
    out['next%d' % i] = torch.rand(4).numpy()
    out['stats%d' % i] = np.array(d.time_mask_stats)
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


class Masker:
    """The masking half of TimeMaskedW2VBertSV without an encoder."""
    _mask_frames = TimeMaskedW2VBertSV._mask_frames
    time_mask_stats = TimeMaskedW2VBertSV.time_mask_stats

    def __init__(self, prob, span, training=True):
        self.time_mask_prob, self.time_mask_span = prob, span
        self.training, self._masked_valid = training, None


def test_off_and_eval_are_identity_without_a_draw():
    h = torch.randn(4, 100, 8)
    torch.manual_seed(0)
    first = torch.rand(3)
    torch.manual_seed(0)
    assert Masker(0.0, 10)._mask_frames(h) is h
    assert Masker(0.05, 10, training=False)._mask_frames(h) is h
    assert torch.equal(torch.rand(3), first)   # generator untouched
    assert Masker(0.0, 10).time_mask_stats == [0, 0]


def test_rate_spans_and_padding():
    torch.manual_seed(0)
    d = Masker(0.05, 10)
    flens = torch.tensor([300, 200, 100, 25] * 8)
    pad = torch.arange(300)[None] >= flens[:, None]
    total = 0
    for _ in range(200):
        o = d._mask_frames(torch.ones(32, 300, 4), flens)
        z = (o == 0).all(-1)
        assert not (z & pad).any()          # padding never counted / touched
        assert ((o == 0) | (o == 1)).all()
        total += int(z.sum())
    masked, valid = d.time_mask_stats
    assert total == masked and valid == 200 * int(flens.sum())
    assert 0.040 < masked / valid < 0.052   # <= prob: overlaps only lower it
    d2 = Masker(0.05, 10)                   # equal-length path
    for _ in range(100):
        d2._mask_frames(torch.ones(16, 150, 4))
    masked, valid = d2.time_mask_stats
    assert valid == 100 * 16 * 150 and 0.040 < masked / valid < 0.052


def test_span_contiguity():
    torch.manual_seed(1)
    o = Masker(0.2, 5)._mask_frames(torch.ones(1, 50, 2))
    z = (o == 0).all(-1)[0].int().tolist()
    runs = [len(r) for r in ''.join(map(str, z)).split('0') if r]
    assert runs and min(runs) >= 5


def test_shorter_than_span_is_safe_and_lighter():
    torch.manual_seed(2)
    d = Masker(0.05, 10)
    flens = torch.tensor([1, 5, 9, 10] * 250)
    z = torch.zeros(4)
    for _ in range(50):
        o = d._mask_frames(torch.ones(1000, 10, 2), flens)
        m = (o == 0).all(-1)
        assert not (m & (torch.arange(10)[None] >= flens[:, None])).any()
        z += m.view(250, 4, 10).sum((0, 2)).float()
    frac = z / (50 * 250 * torch.tensor([1., 5., 9., 10.]))
    assert frac[0] < 0.01 and frac[1] < 0.035 and 0.04 < frac[3] < 0.06, frac


def test_draw_consumes_two_draws_in_order():
    valid = torch.tensor([120.0, 80.0, 200.0])
    torch.manual_seed(5)
    mask = draw_time_mask(200, valid, 0.05, 10)
    after = torch.rand(2)
    torch.manual_seed(5)
    u = torch.rand(3)
    v = torch.rand(3, int(200 * 0.05 / 10) + 1)
    assert torch.equal(torch.rand(2), after)
    n = torch.floor(valid * 0.05 / 10 + u).long()
    ref = torch.zeros(3, 200, dtype=torch.bool)
    for b in range(3):
        for k in range(int(n[b])):
            start = int(v[b, k] * (valid[b] - 10 + 1))
            ref[b, start:min(start + 10, int(valid[b]))] = True
    assert torch.equal(mask, ref)


def tiny_assets(path):
    """A two-layer random W2V-BERT checkpoint directory for the wrapper."""
    from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel
    torch.manual_seed(0)
    Wav2Vec2BertModel(Wav2Vec2BertConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, feature_projection_input_dim=160,
        add_adapter=False, vocab_size=None)).save_pretrained(str(path))
    return {'path_or_url': str(path)}


@pytest.mark.skipif(not HAS_CORE_HOOK,
                    reason='core frontend has no _mask_frames hook yet')
def test_hook_masks_both_forward_paths(tmp_path):
    args = tiny_assets(tmp_path / 'assets')
    front = TimeMaskedW2VBertSV(args, time_mask={'prob': 0.2, 'span': 4})
    base = W2VBertSVFrontend(args)
    assert list(front.state_dict()) == list(base.state_dict())
    base.load_state_dict(front.state_dict())
    wav = torch.randn(3, 16000)
    lens = torch.tensor([16000, 12000, 8000])
    seen = []
    hook = front.upstream.encoder.layers[0].register_forward_hook(
        lambda mod, inp, out: seen.append(inp[0].detach()))
    for kwargs in ({}, {'input_lengths': lens}):
        front.eval()
        base.eval()
        with torch.no_grad():
            ref, ref_len = base(wav, **kwargs)
            out, out_len = front(wav, **kwargs)
        assert torch.equal(out, ref)        # eval: the stock frontend
        assert (out_len is None) == (ref_len is None)
        front.train()
        before = front.time_mask_stats
        torch.manual_seed(3)
        out, out_len = front(wav, **kwargs)
        masked, valid = (a - b for a, b in
                         zip(front.time_mask_stats, before))
        frames = out.shape[2]
        n_valid = 3 * frames if out_len is None else int(out_len.sum())
        assert valid == n_valid and 0 < masked < valid
        zero = (seen[-1] == 0).all(-1)      # encoder input rows set to 0
        pad = torch.zeros_like(zero) if out_len is None else \
            torch.arange(frames)[None] >= out_len[:, None]
        assert int((zero & ~pad).sum()) == masked
        assert not torch.equal(out, ref)
    hook.remove()


@pytest.mark.skipif(not HAS_CORE_HOOK,
                    reason='core frontend has no _mask_frames hook yet')
def test_constructor_checks(tmp_path):
    args = tiny_assets(tmp_path / 'assets')
    assert TimeMaskedW2VBertSV(args).time_mask_prob == 0.0
    for bad in ({'prob': 0.6}, {'prob': -0.1}, {'span': 0}, {'ratio': 0.1}):
        with pytest.raises(ValueError):
            TimeMaskedW2VBertSV(args, time_mask=bad)
    with pytest.raises(ValueError):
        TimeMaskedW2VBertSV(args, frozen=True, time_mask={'prob': 0.05})
    TimeMaskedW2VBertSV(args, frozen=True, time_mask={'prob': 0.0})


@needs_source
def test_parity_with_source(tmp_path):
    gen = torch.Generator().manual_seed(77)
    cases = []
    for i in range(16):
        prob, span = ((0.05, 10), (0.05, 10), (0.2, 5), (0.5, 3))[i % 4]
        calls = []
        for j in range(5):   # alternate padded and equal-length batches
            frames = int(torch.randint(60, 400, (1,), generator=gen))
            batch = (48, 24, 7)[(i + j) % 3]
            h = torch.randn(batch, frames, 6, generator=gen)
            flens = None
            if (i + j) % 2 == 0:
                flens = torch.randint(1, frames + 1, (batch,), generator=gen)
                flens[0] = frames
            calls.append((h, flens))
        cases.append((prob, span, 1000 + i, calls))
    inputs = {'n_cases': np.array(len(cases))}
    for i, (prob, span, seed, calls) in enumerate(cases):
        inputs.update({'prob%d' % i: np.array(prob), 'span%d' % i:
                       np.array(span), 'seed%d' % i: np.array(seed),
                       'calls%d' % i: np.array(len(calls))})
        for j, (h, flens) in enumerate(calls):
            inputs['h%d_%d' % (i, j)] = h.numpy()
            if flens is not None:
                inputs['flens%d_%d' % (i, j)] = flens.numpy()
    src = run_source(SOURCE_DRIVER, tmp_path, inputs)
    n_calls, n_masked = 0, 0
    for i, (prob, span, seed, calls) in enumerate(cases):
        d = Masker(prob, span)
        torch.manual_seed(seed)
        for j, (h, flens) in enumerate(calls):
            out = d._mask_frames(h, flens).numpy()
            assert np.array_equal(out, src['o%d_%d' % (i, j)])
            n_calls += 1
        assert np.array_equal(torch.rand(4).numpy(), src['next%d' % i])
        assert d.time_mask_stats == src['stats%d' % i].tolist()
        n_masked += d.time_mask_stats[0]
    assert n_masked > 0
    print('time-mask parity: %d seeded sequences, %d calls, %d masked frames,'
          ' outputs / generator position / statistics identical'
          % (len(cases), n_calls, n_masked))
