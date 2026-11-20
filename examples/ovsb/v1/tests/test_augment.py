"""Noise / reverberation stage: type draw, application, source parity."""

import collections
import io
import os
import pickle
import random

import numpy as np
import pytest
import torch
from scipy import signal
from scipy.io import wavfile

import proc_parity as pp
from ovsb.config import schema
from ovsb.data import augment as A
from ovsb.data.lmdb_multi import MultiLmdbData
from wespeaker.dataset import processor as core

SR = pp.SR
PROBS = {'clean': 0.35, 'noise': 0.2, 'reverb': 0.15, 'reverb_noise': 0.25}

needs_source = pytest.mark.skipif(
    not pp.have_source(), reason='set OVSB_SOURCE_TREE for parity tests')
needs_recipe = pytest.mark.skipif(
    not (pp.have_source() and pp.SOURCE_RECIPE and os.path.isdir(
        os.path.join(pp.SOURCE_RECIPE, 'data/musan/lmdb'))),
    reason='set OVSB_SOURCE_TREE and OVSB_SOURCE_RECIPE (real LMDBs)')


@pytest.fixture
def fake_rir():
    """Delayed impulse plus a short decaying tail; two rooms, four each."""
    rng = np.random.RandomState(0)
    items = []
    for room in ('smallroom/Room001', 'mediumroom/Room002'):
        for j in range(4):
            h = np.zeros(2000, dtype=np.float32)
            h[400] = 1.0
            h[401:1400] = 0.05 * rng.randn(999) * np.exp(
                -np.arange(999) / 200.0)
            items.append(('{}/{}-{:05d}.wav'.format(room, room[-7:], j),
                          pp.wav_bytes(h)))
    return pp.FakeSource(items)


@pytest.fixture
def fake_noise():
    rng = np.random.RandomState(1)
    items = []
    for cat in ('noise', 'music', 'speech'):
        x = (rng.randn(SR * 8) * 3000).astype(np.int16)
        items.append(('{}/x/{}-0001.wav'.format(cat, cat), pp.wav_bytes(x)))
    x = (rng.randn(8000 * 8) * 3000).astype(np.int16)
    items.append(('noise/x/noise-8k.wav', pp.wav_bytes(x, 8000)))
    return pp.FakeSource(items)


def _samples(n, length=8000, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        wav = (0.1 * rng.randn(length)).astype(np.float32)
        out.append(dict(key='utt{:05d}'.format(i), sample_rate=SR,
                        wav=torch.from_numpy(wav).unsqueeze(0)))
    return out


def _rirs(src):
    out = {}
    for key, value in src.items:
        _, h = wavfile.read(io.BytesIO(value))
        h = h.astype(np.float32)
        out[key] = h / np.sqrt(np.sum(h**2))
    return out


def _classify(x, y, rirs):
    """(reverberated?, SNR in dB of the residual) of output y for input x."""
    best = None
    for h in list(rirs.values()) + [None]:
        r = x if h is None else signal.convolve(x, h)[:len(x)]
        g = np.dot(y, r) / np.dot(r, r)
        e = y - g * r
        snr = 10 * np.log10(np.sum((g * r)**2) / (np.sum(e**2) + 1e-20))
        if best is None or snr > best[1]:
            best = (h is not None, snr)
    return best


def test_cumulative_thresholds_of_the_recipe():
    cum = A._aug_cum(pp.AUG_PROBS)
    acc = 0.0
    want = []
    for w in (1.0 - (0.2 + 0.2 + 0.2), 0.2, 0.2, 0.2):
        acc += w
        want.append(acc)
    assert cum[:3] == want[:3] and cum[3] == 1.0
    assert A._aug_cum({'noise': 0.2}) == [0.8, 1.0, 1.0, 1.0]
    assert A._aug_cum({}) == [1.0, 1.0, 1.0, 1.0]
    with pytest.raises(AssertionError):
        A._aug_cum({'overlap': 0.1})
    with pytest.raises(AssertionError):
        A._aug_cum({'noise': 0.7, 'reverb': 0.4})
    # same type order as the integer codes of the batch
    assert schema.AUG_TYPES == ('clean', 'noise', 'reverb', 'reverb_noise')


def test_draw_probabilities():
    random.seed(0)
    n = 20000
    probs = {'noise': 0.12, 'reverb': 0.12, 'reverb_noise': 0.16}
    counts = collections.Counter(
        s['aug_type'] for s in A.draw_aug_type(
            ({'key': str(i)} for i in range(n)), probs))
    for t, p in dict(probs, clean=0.6).items():
        assert abs(counts[t] / n - p) < 0.015, (t, counts[t] / n)


def test_draw_per_arm_anchors_clean_one_draw_per_row():
    n = 6000
    rows = [dict(key=str(i), arm=i % 3, pair_role=(i // 3) % 3)
            for i in range(n)]
    random.seed(7)
    out = list(A.draw_aug_type(iter([dict(r) for r in rows]), PROBS,
                               {1: {'noise': 1.0}, 2: {}}))
    state = random.getstate()
    random.seed(7)
    us = [random.random() for _ in range(n)]
    assert random.getstate() == state  # exactly one draw per row
    cum = A._aug_cum(PROBS)
    for r, u in zip(out, us):
        if r['pair_role'] == 1:
            assert r['aug_type'] == 'clean'
        elif r['arm'] == 1:
            assert r['aug_type'] == 'noise'
        elif r['arm'] == 2:
            assert r['aug_type'] == 'clean'
        else:
            want = [t for t, c in zip(schema.AUG_TYPES, cum) if u < c][0]
            assert r['aug_type'] == want
    views = collections.Counter(r['aug_type'] for r in out
                                if r['pair_role'] == 2 and r['arm'] == 0)
    assert len(views) == 4  # views are augmented like plain rows


def test_apply_noise_equals_core_without_noise_rir(fake_noise):
    rng = np.random.RandomState(2)
    for i in range(40):
        x = (0.1 * rng.randn(rng.randint(4000, 40000))).astype(np.float32)
        outs = []
        for fn in (core.apply_noise, A.apply_noise):
            random.seed(100 + i)
            y, key, snr = fn(x.copy(), fake_noise, SR)
            outs.append((y.tobytes(), y.dtype, key, snr, random.getstate()))
        assert outs[0] == outs[1]
    random.seed(1)
    a = core.apply_noise(x, fake_noise, SR, {'noise': [3, 4], 'music': [3, 4],
                                             'speech': [3, 4]})
    random.seed(1)
    b = A.apply_noise(x, fake_noise, SR, {'noise': [3, 4], 'music': [3, 4],
                                          'speech': [3, 4]})
    assert a[0].tobytes() == b[0].tobytes() and a[1:] == b[1:]
    assert 3 <= b[2] <= 4


def test_noise_rir_sets_the_snr_after_reverberation(fake_noise, fake_rir):
    rirs = _rirs(fake_rir)
    h = rirs[fake_rir.keys[0]]
    x = (0.3 * np.random.RandomState(3).randn(2 * SR)).astype(np.float32)
    random.seed(4)
    dry, key, snr = A.apply_noise(x, fake_noise, SR)
    state = random.getstate()
    random.seed(4)
    wet, key2, snr2 = A.apply_noise(x, fake_noise, SR, noise_rir=h)
    assert (key, snr) == (key2, snr2) and random.getstate() == state
    assert not np.array_equal(dry, wet)
    for y in (dry, wet):  # both noises sit at the drawn SNR
        got = 10 * np.log10(np.mean(x**2) / np.mean((y - x)**2))
        assert abs(got - snr) < 0.2
    # the wet noise is the dry chunk through the RIR, up to the gain
    n_dry, n_wet = (dry - x).astype(np.float64), (wet - x).astype(np.float64)
    ref = signal.convolve(n_dry, h)[:len(x)]
    corr = np.dot(n_wet, ref) / np.sqrt(np.dot(ref, ref) * np.dot(n_wet,
                                                                  n_wet))
    assert corr > 0.999


def test_types_are_applied(fake_rir, fake_noise):
    random.seed(1)
    n = 2000
    samples = _samples(n, length=1600, seed=3)
    clean = [s['wav'][0].numpy().copy() for s in samples]
    wavs = [s['wav'] for s in samples]
    out = list(A.add_reverb_noise(A.draw_aug_type(iter(samples), PROBS),
                                  fake_rir, fake_noise, SR))
    rirs = _rirs(fake_rir)
    counts = collections.Counter()
    for x, w, s in zip(clean, wavs, out):
        y = s['wav'][0].numpy()
        assert y.shape == x.shape and y.dtype == np.float32
        if s['aug_type'] == 'clean':
            assert s['wav'] is w and np.array_equal(y, x)
            continue
        assert np.abs(y).max() <= 1.0
        reverbed, snr = _classify(x, y.astype(np.float64), rirs)
        assert reverbed == (s['aug_type'] != 'noise')
        assert (snr < 40) == (s['aug_type'] != 'reverb')
        counts[s['aug_type']] += 1
    assert min(counts.values()) > 200 and len(counts) == 3


def test_reverberant_noise_uses_the_room_of_the_speech(fake_rir, fake_noise,
                                                       monkeypatch):
    rooms = []
    real = A._rir_same_room

    def spy(source, key):
        got = real(source, key)
        rooms.append((key.rsplit('/', 1)[0], got[0].rsplit('/', 1)[0]))
        return got

    monkeypatch.setattr(A, '_rir_same_room', spy)
    multi = MultiLmdbData([fake_rir], [1.0])
    types = ['reverb_noise', 'noise', 'reverb'] * 20

    def run(**kw):
        samples = _samples(60, length=8000, seed=5)
        for s, t in zip(samples, types):
            s['aug_type'] = t
        random.seed(2)
        return list(A.add_reverb_noise(iter(samples), multi, fake_noise, SR,
                                       **kw))

    wet = run(reverberant_noise=True)
    assert len(rooms) == 20  # reverb_noise rows only: plain noise stays dry
    assert all(a == b for a, b in rooms)
    assert {a for a, _ in rooms} == {'smallroom/Room001',
                                     'mediumroom/Room002'}
    dry = run()
    assert len(rooms) == 20  # option off: no second RIR
    assert not torch.equal(wet[0]['wav'], dry[0]['wav'])
    assert wet[0]['wav'].shape == dry[0]['wav'].shape


def test_draw_args():
    cfg = schema.build_section(schema.AugmentCfg, pp.augment_cfg(dict(
        noise_lmdb='n', reverb_lmdb='r', reverb_real='x',
        noise_exclude_list=None)), 'a')
    kw = A.draw_args(cfg, pp.ARM_INDEX)
    assert kw['aug_probs'] == pp.AUG_PROBS
    assert kw['arm_probs'] == {pp.ARM_INDEX[n]: {'noise': 0.2} for n in (
        '3dspeaker', 'cn1', 'cn2', 'tvx')}
    cfg.arm_probs = {}
    assert A.draw_args(cfg, pp.ARM_INDEX)['arm_probs'] is None


# ------------------------------------------------------------------ parity
CHAIN = ['filter', 'spk_to_id', 'resample', 'crop', 'zero_run', 'draw',
         'reverb_noise']


def _specs(root):
    def lmdbs(name, seed=0):
        return pp.make_aug_lmdbs(os.path.join(root, name), seed)

    base = dict(stages=CHAIN, batch_size=48, crop_seed=42, epoch=1,
                crop=pp.CROP, zero_run=pp.ZERO_RUN, filter=pp.FILTER,
                seed=31, stream=dict(seed=21, n=600, rate_8k=0.05))
    recipe = pp.augment_cfg(lmdbs('a'))
    dense = dict(noise=0.34, reverb=0.33, reverb_noise=0.33)
    return [
        dict(base, name='recipe', augment=recipe),
        dict(base, name='recipe_epoch3_worker5', augment=recipe, epoch=3,
             worker_id=5, seed=32, stream=dict(seed=22, n=600)),
        dict(base, name='dense', seed=33, stream=dict(seed=23, n=500),
             augment=pp.augment_cfg(lmdbs('b', 1), probs=dense,
                                    arm_probs={})),
        dict(base, name='dry_noise_one_source', seed=34,
             stream=dict(seed=24, n=500),
             augment=pp.augment_cfg(
                 lmdbs('c', 2), probs=dense, reverberant_noise=False,
                 reverb_sources=['simulated'], reverb_weights=[1.0],
                 noise_exclude_list=None)),
        dict(base, name='augment_only', seed=35,
             stages=['draw', 'reverb_noise'],
             stream=dict(seed=25, n=300, profile='speech'),
             augment=pp.augment_cfg(lmdbs('d', 3), probs=dense)),
    ]


@pytest.fixture(scope='module')
def parity(tmp_path_factory):
    root = str(tmp_path_factory.mktemp('aug'))
    specs = _specs(root)
    source, new = pp.run_both(specs, root)
    return {s['name']: (a, b) for s, a, b in zip(specs, source, new)}


NAMES = ['recipe', 'recipe_epoch3_worker5', 'dense', 'dry_noise_one_source',
         'augment_only']


@needs_source
@pytest.mark.parametrize('name', NAMES)
def test_parity_with_source(parity, name):
    source, new = parity[name]
    assert len(new['rows']) >= 300
    assert pp.diff_rows(source['rows'], new['rows']) == []
    assert source['rng'] == new['rng']
    types = collections.Counter(r['aug_type'] for r in new['rows'])
    assert min(types[t] for t in ('noise', 'reverb', 'reverb_noise')) > 20


@needs_recipe
def test_parity_with_source_on_the_real_lmdbs(tmp_path):
    """One chain against the recipe's noise and RIR LMDBs (read-only)."""
    import lmdb
    data = os.path.join(pp.SOURCE_RECIPE, 'data')
    noise_lmdb = os.path.realpath(os.path.join(data, 'musan/lmdb'))
    env = lmdb.open(noise_lmdb, readonly=True, lock=False, readahead=False)
    with env.begin(write=False) as txn:
        keys = pickle.loads(txn.get(b'__keys__'))
    env.close()
    exclude = tmp_path / 'exclude.txt'
    exclude.write_text('\n'.join(keys[::7]) + '\n')
    synth = pp.make_aug_lmdbs(str(tmp_path / 'synth'))
    aug = dict(probs=dict(noise=0.3, reverb=0.3, reverb_noise=0.3),
               arm_probs=dict(pp.AUG_ARM_PROBS), reverberant_noise=True,
               reverb_sources=['simulated', synth['reverb_real']],
               reverb_weights=[0.8, 0.2], noise_lmdb=noise_lmdb,
               reverb_lmdb=os.path.realpath(os.path.join(data, 'rirs/lmdb')),
               noise_exclude_list=str(exclude))
    spec = dict(stages=CHAIN, batch_size=48, crop_seed=42, epoch=1,
                crop=pp.CROP, zero_run=pp.ZERO_RUN, filter=pp.FILTER,
                seed=41, stream=dict(seed=31, n=240), augment=aug)
    (source, ), (new, ) = pp.run_both([spec], str(tmp_path))
    assert len(new['rows']) >= 150
    assert pp.diff_rows(source['rows'], new['rows']) == []
    assert source['rng'] == new['rng']
    types = collections.Counter(r['aug_type'] for r in new['rows'])
    assert min(types[t] for t in ('noise', 'reverb', 'reverb_noise')) > 10
