"""Zero-run guard: statistics, dither stream, silent rows, source parity."""

import collections

import numpy as np
import pytest
import torch

import proc_parity as pp
from ovsb.config import schema
from ovsb.data import zero_run as Z

SR = pp.SR
LSB = 1.0 / 32768.0

needs_source = pytest.mark.skipif(
    not pp.have_source(), reason='set OVSB_SOURCE_TREE for parity tests')


def _guard(x, dither=True, **kw):
    data = [dict(key='k', wav=torch.from_numpy(x.copy()).unsqueeze(0), arm=3)]
    args = dict(arms=None, dither=dither, min_run_ms=50, sample_rate=SR,
                seed=717, epoch=1, rank=0, stats=True)
    args.update(kw)
    return list(Z.zero_run_guard(iter(data), **args))[0]


def _rows():
    rng = np.random.default_rng(0)
    speech = (rng.standard_normal(SR) * 2000).astype(np.int16).astype(
        np.float32) / 32768.0
    zeros = np.zeros(SR, np.float32)
    lsb = rng.integers(-4, 2, 2 * SR).astype(np.float32) / 32768.0
    faint_run = lsb.copy()
    faint_run[SR:] = 0.0
    return dict(all_zero=np.concatenate([zeros, zeros]),
                speech_zero=np.concatenate([speech, zeros]),
                lsb_noise=lsb, faint_run=faint_run,
                speech=np.concatenate([speech, speech]))


def test_zero_run_stats():
    x = np.array([0.5, 0, 0, 0, 3 * LSB, 0, 0, 0.2, 0, 0, 0, 0], np.float32)
    lsb, runs = Z.zero_run_stats(x, 3)
    assert runs == [(1, 4), (8, 12)]
    assert lsb == pytest.approx(10 / 12)
    assert Z.zero_run_stats(x, 4)[1] == [(8, 12)]
    assert Z.zero_run_stats(x, 3, lsb_level=2)[0] == pytest.approx(9 / 12)
    assert Z.zero_run_stats(np.zeros(0, np.float32), 3) == (0.0, [])


def test_speech_with_a_zero_run_is_dithered_from_its_own_stream():
    rows = _rows()
    s = _guard(rows['speech_zero'])
    y = s['wav'][0].numpy()
    assert (s['zr_dith'], s['zr_silent'], s['zr_run']) == (1, 0, 1)
    assert 'zr_bypass' not in s
    assert np.array_equal(y[:SR], rows['speech_zero'][:SR])
    want = np.random.default_rng((717, 1, 0, 31)).uniform(
        -LSB, LSB, SR).astype(np.float32)
    assert np.array_equal(y[SR:], want)
    assert np.abs(y[SR:]).max() <= LSB and (y[SR:] != 0).mean() > 0.99
    # a loud crop whose zero run covers all but one sample is dithered too
    one = np.zeros(2 * SR, np.float32)
    one[0] = 1000 * LSB
    s = _guard(one)
    assert (s['zr_dith'], s['zr_silent']) == (1, 0)


def test_silent_rows_are_flagged_and_left_untouched():
    rows = _rows()
    out = {k: _guard(v) for k, v in rows.items()}
    for k in ('all_zero', 'lsb_noise', 'faint_run'):  # peak below 8 LSB
        assert out[k]['zr_bypass'] == 1 and out[k]['zr_dith'] == 0, k
        assert out[k]['zr_silent'] == 1
        assert np.array_equal(out[k]['wav'][0].numpy(), rows[k])
    assert out['all_zero']['zr_run'] == 1
    assert out['lsb_noise']['zr_run'] == 0
    assert 'zr_bypass' not in out['speech']
    assert (out['speech']['zr_dith'], out['speech']['zr_run']) == (0, 0)
    assert Z.DITHER_MIN_PEAK_LSB == 8.0
    # a silent row consumes nothing of the dither stream
    data = [dict(key=str(i), arm=0, wav=torch.from_numpy(x.copy())[None])
            for i, x in enumerate([rows['all_zero'], rows['speech_zero']])]
    two = list(Z.zero_run_guard(iter(data), seed=717, epoch=1, rank=0))
    assert torch.equal(two[1]['wav'], out['speech_zero']['wav'])


def test_without_dither_only_statistics():
    for key, x in _rows().items():
        s = _guard(x, dither=False)
        assert 'zr_bypass' not in s and s['zr_dith'] == 0
        assert s['zr_silent'] == 0
        assert np.array_equal(s['wav'][0].numpy(), x)
    s = _guard(_rows()['speech_zero'], stats=False)
    assert 'zr_lsb' not in s and 'zr_run' not in s


def test_dither_stream_per_rank_epoch_and_stream():
    x = np.r_[np.ones(10) * .1, np.zeros(1600), np.ones(10) * -.2].astype(
        'float32')

    def run(**kw):
        data = [dict(wav=torch.tensor(x).unsqueeze(0))]
        return next(Z.zero_run_guard(iter(data), seed=314,
                                     **kw))['wav'][0].numpy()

    r_none, r0, r1, r2 = run(), run(rank=0), run(rank=1), run(rank=2)
    want = x.copy()
    want[10:1610] = np.random.default_rng((314, 0, 0, 31)).uniform(
        -LSB, LSB, 1600)
    assert np.array_equal(r_none, r0) and np.array_equal(r0, want)
    want[10:1610] = np.random.default_rng((314, 0, 0, 31, 2)).uniform(
        -LSB, LSB, 1600)
    assert np.array_equal(r2, want)
    assert not np.array_equal(r0, r1) and not np.array_equal(r1, r2)
    for y in (r1, r2):
        assert np.array_equal(y[x != 0], x[x != 0])
        assert np.abs(y - x).max() <= LSB and not (y[10:1610] == 0).any()
    assert not np.array_equal(r0, run(rank=0, epoch=1))
    # a resumed epoch (stream > 0) gets streams distinct from every rank's
    s1, s1r2 = run(rank=0, stream=1), run(rank=2, stream=1)
    assert not any(np.array_equal(s1, y) for y in (r0, r1, r2, s1r2))
    want[10:1610] = np.random.default_rng((314, 0, 0, 31, 2, 1)).uniform(
        -LSB, LSB, 1600)
    assert np.array_equal(s1r2, want)


def test_arm_list_limits_the_dither():
    x = _rows()['speech_zero']
    listed = _guard(x, arms=[3])
    other = _guard(x, arms=[1, 2])
    assert listed['zr_dith'] == 1 and other['zr_dith'] == 0
    assert other['zr_run'] == 1 and other['zr_silent'] == 0
    assert np.array_equal(other['wav'][0].numpy(), x)
    silent = _guard(_rows()['all_zero'], arms=[1])
    assert 'zr_bypass' not in silent  # unlisted arm: never flagged


def test_zero_run_args():
    cfg = schema.build_section(schema.ZeroRunCfg, pp.ZERO_RUN, 'z')
    kw = Z.zero_run_args(cfg, pp.ARM_INDEX, SR, seed=42)
    assert kw == dict(arms=None, dither=True, min_run_ms=50.0, lsb_level=4.0,
                      dither_lsb=1.0, sample_rate=SR, stats=True, seed=42)
    cfg = schema.build_section(
        schema.ZeroRunCfg, dict(pp.ZERO_RUN, arms=['cn1', 'vox2']), 'z')
    assert Z.zero_run_args(cfg, pp.ARM_INDEX, SR, 1)['arms'] == [
        pp.ARM_INDEX['cn1'], pp.ARM_INDEX['vox2']]


# ------------------------------------------------------------------ parity
def _specs():
    base = dict(stages=['crop', 'zero_run'], batch_size=48, crop_seed=42,
                epoch=1, crop=pp.CROP, zero_run=pp.ZERO_RUN, seed=21,
                stream=dict(seed=11, n=600, profile='zeros'))
    return [
        dict(base, name='recipe'),
        dict(base, name='rank3_epoch5', rank=3, epoch=5, seed=22),
        dict(base, name='worker2', worker_id=2, seed=23),
        dict(base, name='listed_arms', seed=24,
             zero_run=dict(pp.ZERO_RUN, arms=['cn1', 'mandarin_t1', 'vox2'],
                           min_run_ms=20.0, dither_lsb=2.0, lsb_level=2.0)),
        dict(base, name='stats_only', dither=False, seed=25),
        dict(base, name='uncropped', stages=['zero_run'], seed=26,
             stream=dict(seed=12, n=300, profile='mixed')),
    ]


@pytest.fixture(scope='module')
def parity(tmp_path_factory):
    specs = _specs()
    source, new = pp.run_both(specs, str(tmp_path_factory.mktemp('zr')))
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
    rows = [r for name in ('recipe', 'rank3_epoch5', 'worker2')
            for r in parity[name][1]['rows']]
    seen = collections.Counter(
        (r['zr_run'], r['zr_dith'], r.get('zr_bypass', 0)) for r in rows)
    for case in ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 0, 1)):
        assert seen[case] > 5, seen
    listed = parity['listed_arms'][1]['rows']
    assert any(r['zr_run'] and not r['zr_dith'] and not r['zr_silent']
               for r in listed)
