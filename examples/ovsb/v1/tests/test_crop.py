"""Variable-length crop: batch structure, random streams, source parity."""

import collections
import random

import numpy as np
import pytest
import torch

import proc_parity as pp
from ovsb.config import schema
from ovsb.data import crop as C

SR = pp.SR
LENS = [2 * SR, 3 * SR, 6 * SR]
WEIGHTS = [0.3, 0.4, 0.3]

needs_source = pytest.mark.skipif(
    not pp.have_source(), reason='set OVSB_SOURCE_TREE for parity tests')


def _crop(data, batch_size=8, **kw):
    args = dict(full_prob=0.5, onset_prob=0.25, onset_max_len=SR,
                pair_prob=0.3, pair_anchor_len=6 * SR,
                pair_view_lens=[SR // 2, 2 * SR], sample_rate=SR)
    args.update(kw)
    return C.random_chunk_var(data, LENS, WEIGHTS, batch_size, SR // 2,
                              **args)


def _const(n, length=10 * SR, arm=0):
    for i in range(n):
        yield dict(key=str(i), arm=arm,
                   wav=torch.full((1, length), 0.01))


def test_frame_helpers():
    assert C._frame_active(np.zeros(7 * 320, np.float32), SR)[:2] == (None,
                                                                      None)
    x = np.zeros(2 * SR, np.float32)
    x[SR:] = 0.1  # 1 s of silence, then signal
    e, active, hop = C._frame_active(x, SR)
    assert hop == 320 and e.shape == (100, ) and active.sum() == 50
    assert C._active_frac(active, hop, 0, SR) == 0.0
    assert C._active_frac(active, hop, SR // 2, SR) == 0.5
    assert C._active_frac(active, hop, 0, 100) == 1.0  # no full frame
    random.seed(0)
    starts = {C._onset_start(x, SR // 2, SR) for _ in range(50)}
    # the onset is at 1 s; the crop starts up to 100 ms before it
    assert all(SR - SR // 10 <= s <= SR for s in starts) and len(starts) > 5


def test_onset_start_without_onset_draws_nothing():
    random.seed(3)
    state = random.getstate()
    assert C._onset_start(np.full(SR, 0.1, np.float32), SR // 2, SR) is None
    assert C._onset_start(np.zeros(100, np.float32), 50, SR) is None
    assert random.getstate() == state


def test_batches_are_rectangular_and_pairs_alternate():
    random.seed(0)
    bs = 8
    out = list(_crop(pp.make_stream(1, 400, 'speech'), bs))
    n_pair = 0
    for b in range(len(out) // bs):
        batch = out[b * bs:(b + 1) * bs]
        assert len({x['pad_to'] for x in batch}) == 1
        roles = [x['pair_role'] for x in batch]
        if 1 in roles:
            n_pair += 1
            assert roles == [1, 2] * (bs // 2)
            assert batch[0]['pad_to'] == 6 * SR
            for a, v in zip(batch[0::2], batch[1::2]):
                assert a['key'] == v['key'] and a['spk'] == v['spk']
                assert a['wav'].shape[1] <= 6 * SR
                assert SR // 2 <= v['wav'].shape[1] <= 2 * SR
        else:
            assert roles == [0] * bs and batch[0]['pad_to'] in LENS
        for x in batch:
            assert x['wav'].dim() == 2 and x['wav'].shape[0] == 1
            assert x['wav'].shape[1] <= x['pad_to']
            assert 'eg_redraw' not in x  # no gate configured
    assert 0 < n_pair < len(out) // bs


def test_pair_batch_consumes_half_as_many_rows():
    random.seed(1)
    bs = 8
    out = list(_crop(_const(40), bs, pair_prob=1.0))
    assert len(out) == 80  # two rows per input utterance
    assert [x['key'] for x in out[:bs]] == ['0', '0', '1', '1', '2', '2',
                                            '3', '3']
    out = list(_crop(_const(40), bs, pair_prob=0.0))
    assert len(out) == 40 and {x['pair_role'] for x in out} == {0}


def test_batch_sequence_follows_the_seed_string():
    bs = 4

    def run(global_seed, seed=7, epoch=3, **kw):
        random.seed(global_seed)
        out = list(_crop(_const(200), bs, seed=seed, epoch=epoch, **kw))
        return [(x['pad_to'], x['pair_role']) for x in out][::bs]

    ref = random.Random('osvb-varcrop-7-3-0')
    want = []
    for _ in range(200):
        cap = ref.choices(LENS, weights=WEIGHTS)[0]
        paired = ref.random() < 0.3
        want.append((6 * SR, 1) if paired else (cap, 0))
    got = run(11)
    assert got == want[:len(got)]
    assert got == run(22)  # the same on every rank
    assert got != run(11, epoch=4) and got != run(11, seed=8)
    assert got != run(11, stream=1)
    assert run(11, stream=1) == run(22, stream=1)


def test_short_utterance_is_kept_whole():
    random.seed(2)
    out = list(_crop(_const(64, length=SR), 8, pair_prob=0.5,
                     full_prob=1.0))
    assert all(x['wav'].shape[1] == SR for x in out
               if x['pair_role'] != 2)
    assert all(x['wav'].shape[1] <= SR for x in out)


def test_arm_caps_follow_their_distribution_below_the_batch_cap():
    random.seed(4)
    caps = {5: ([2 * SR, 3 * SR, 6 * SR], [0.6, 0.3, 0.1])}
    out = list(_crop(_const(24000, arm=5), 8, pair_prob=0.0, full_prob=1.0,
                     arm_caps=caps))
    assert all(x['wav'].shape[1] <= x['pad_to'] for x in out)
    counts = collections.Counter(x['wav'].shape[1] for x in out)
    for length, p in zip(*caps[5]):
        assert abs(counts[length] / len(out) - p) < 0.03, counts
    # an unlisted arm is cropped to the batch cap and draws nothing extra
    random.seed(4)
    a = list(_crop(_const(80, arm=1), 8, arm_caps=caps))
    state = random.getstate()
    random.seed(4)
    b = list(_crop(_const(80, arm=1), 8))
    assert random.getstate() == state
    assert all(torch.equal(x['wav'], y['wav']) for x, y in zip(a, b))


def test_energy_gate_redraws_short_crops():
    random.seed(5)
    gate = C.EnergyGate(gate_below=int(1.5 * SR), min_active_frac=0.5,
                        max_redraws=4)
    stream = pp.make_stream(6, 600, 'crop')
    out = list(_crop(stream, 8, energy_gate=gate, full_prob=0.0))
    assert all('eg_redraw' in x and 'eg_fail' in x for x in out)
    seen = collections.Counter(x['eg_redraw'] for x in out)
    assert seen[-1] > 0 and seen[0] > 0 and seen[4] > 0
    for x in out:
        assert -1 <= x['eg_redraw'] <= 4
        if x['pair_role'] != 0 or x['wav'].shape[1] >= gate.gate_below:
            assert (x['eg_redraw'], x['eg_fail']) == (-1, 0)
        if x['eg_fail']:
            assert x['eg_redraw'] == 4
    assert any(x['eg_fail'] for x in out)


def test_crop_args_convert_seconds_and_arm_names():
    cfg = schema.build_section(schema.CropCfg, pp.CROP, 'data.crop')
    kw = C.crop_args(cfg, pp.ARM_INDEX, 48, SR, seed=42)
    assert kw['max_lens'] == [32000, 48000, 64000, 96000]
    assert kw['max_weights'] == [0.15, 0.45, 0.25, 0.15]
    assert (kw['batch_size'], kw['min_len'], kw['seed']) == (48, 8000, 42)
    assert kw['onset_max_len'] == 16000 and kw['pair_anchor_len'] == 96000
    assert kw['pair_view_lens'] == [8000, 32000]
    assert set(kw['arm_caps']) == {pp.ARM_INDEX['cn1'], pp.ARM_INDEX['cn2']}
    assert kw['arm_caps'][pp.ARM_INDEX['cn1']] == (
        [32000, 48000, 64000, 96000], [0.3, 0.4, 0.2, 0.1])
    assert kw['energy_gate'] == C.EnergyGate(24000, 0.5, 4)
    plain = dict(pp.CROP, arm_caps={}, energy_gate=None)
    kw = C.crop_args(schema.build_section(schema.CropCfg, plain, 'c'),
                     pp.ARM_INDEX, 48, SR, seed=1)
    assert kw['arm_caps'] is None and kw['energy_gate'] is None
    out = list(C.random_chunk_var(pp.make_stream(1, 96, 'speech'), epoch=1,
                                  **kw))
    assert len(out) >= 96


# ------------------------------------------------------------------ parity
def _specs():
    base = dict(stages=['crop'], batch_size=48, crop_seed=42, epoch=1,
                crop=pp.CROP, seed=11,
                stream=dict(seed=1, n=720, profile='crop'))
    plain = dict(pp.CROP, arm_caps={}, energy_gate=None,
                 pair_view_onset_prob=None)
    return [
        dict(base, name='recipe'),
        dict(base, name='epoch2_seed', epoch=2, seed=12, crop_seed=716,
             stream=dict(seed=2, n=720, profile='crop')),
        dict(base, name='worker3', worker_id=3, seed=13),
        dict(base, name='batch6', batch_size=6, seed=14,
             stream=dict(seed=3, n=300, profile='crop')),
        dict(base, name='no_caps_no_gate', crop=plain, seed=15),
        dict(base, name='no_pairs', crop=dict(pp.CROP, pair_prob=0.0),
             seed=16),
        dict(base, name='all_pairs', crop=dict(pp.CROP, pair_prob=1.0),
             seed=17, stream=dict(seed=4, n=360, profile='crop')),
        dict(base, name='capped_arms_sparse', seed=18,
             crop=dict(pp.CROP, full_prob=0.1, onset_prob=1.0),
             stream=dict(seed=5, n=720, profile='crop',
                         arms=[pp.ARM_INDEX['cn1'], pp.ARM_INDEX['cn2'],
                               pp.ARM_INDEX['vox2']])),
        dict(base, name='filtered', seed=19, filter=pp.FILTER,
             stages=['filter', 'resample', 'crop'],
             stream=dict(seed=6, n=720, profile='mixed', rate_8k=0.05)),
    ]


@pytest.fixture(scope='module')
def parity(tmp_path_factory):
    specs = _specs()
    source, new = pp.run_both(specs, str(tmp_path_factory.mktemp('crop')))
    return {s['name']: (a, b) for s, a, b in zip(specs, source, new)}


@needs_source
@pytest.mark.parametrize('name', [s['name'] for s in _specs()])
def test_parity_with_source(parity, name):
    source, new = parity[name]
    assert len(new['rows']) >= 300
    assert pp.diff_rows(source['rows'], new['rows']) == []
    assert source['rng'] == new['rng']


@needs_source
def test_parity_cases_cover_the_branches(parity):
    rows = [r for _, new in parity.values() for r in new['rows']]
    roles = collections.Counter(r['pair_role'] for r in rows)
    assert min(roles[0], roles[1], roles[2]) > 500
    redraw = collections.Counter(r.get('eg_redraw') for r in rows)
    assert redraw[-1] > 0 and redraw[0] > 0 and redraw[4] > 0
    assert sum(r.get('eg_fail', 0) for r in rows) > 10
    assert 'eg_redraw' not in parity['no_caps_no_gate'][1]['rows'][0]
