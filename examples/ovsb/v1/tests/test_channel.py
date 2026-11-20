"""Codec / channel chain: draws, row selection, failures, isolation, parity.

The tests marked ``needs_ffmpeg`` run the real codecs and skip when
torchaudio cannot load the FFmpeg libraries (load the FFmpeg 6 module
first). The parity tests also need OVSB_SOURCE_TREE.
"""

import collections
import logging
import os
import random
import signal
import time

import numpy as np
import pytest
import torch

import proc_parity as pp
from ovsb.config import schema
from ovsb.data import channel as CH
from ovsb.data import ffmpeg_isolation as FI

SR = pp.SR
ASYM = dict(eq_prob=0.6, bandlimit_prob=0.4, clip_prob=0.0, loss_prob=0.0)
HAVE_FFMPEG = CH._ffmpeg_ok()

needs_ffmpeg = pytest.mark.skipif(
    not HAVE_FFMPEG, reason='FFmpeg libraries not loadable by torchaudio')
needs_source = pytest.mark.skipif(
    not (HAVE_FFMPEG and pp.have_source()),
    reason='needs FFmpeg and OVSB_SOURCE_TREE')


class Boom:
    """An effector whose call always fails."""

    def __init__(self, *args, **kw):
        pass

    def apply(self, *args, **kw):
        raise RuntimeError('injected codec failure')


def _reset_stats():
    CH._STATS['pid'] = None  # the next use starts new counters
    return CH.get_stats()


@pytest.fixture
def fresh_stats():
    """Chain counters and the failure-log limiter start from zero."""
    _reset_stats()
    return CH.get_stats


def _speech(seed=0, seconds=3.0):
    return pp.make_wave(np.random.RandomState(seed), 'speech')[:int(
        seconds * SR)].copy()


def _run(x, seed, **kw):
    random.seed(seed)
    info = {}
    y = CH.apply_channel(x.copy(), SR, info=info, **kw)
    return y, random.getstate(), info


def _rows(arms, n=8, seed=0, **extra):
    rs = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        x = (rs.randn(SR) * 0.1).astype(np.float32)
        rows.append(dict(key=str(i), spk='s%d' % (i // 2), arm=arms[i % len(
            arms)], kpos=i % 2, pair_role=0,
            wav=torch.from_numpy(x).unsqueeze(0), **extra))
    return rows


# ---------------------------------------------------------- without FFmpeg
def test_codec_table_matches_the_batch_codes():
    assert CH.CHAIN_CODECS == ('mp3', 'opus', 'vorbis', 'mulaw8k', 'g722')
    assert tuple(schema.CODECS) == CH.CHAIN_CODECS
    assert set(CH._CODECS) == set(CH.CHAIN_CODECS)


def test_zero_probabilities_still_consume_their_draws(monkeypatch):
    monkeypatch.setattr(CH, '_FFMPEG_OK', False)  # numpy path: no codec lib
    calls = []
    real = random.random
    monkeypatch.setattr(random, 'random', lambda: calls.append(1) or real())
    x = _speech(1)
    random.seed(9)
    y = CH.apply_channel(x, SR, eq_prob=0.0, bandlimit_prob=0.0,
                         codec_prob=1.0, codecs=('g722', ), clip_prob=0.0,
                         loss_prob=0.0)
    state = random.getstate()
    # EQ, band-limit, codec, clip and loss decisions: one draw each
    assert len(calls) == 5
    random.seed(9)
    for _ in range(3):
        real()
    random.choice(['g722'])
    real()
    real()
    assert random.getstate() == state
    assert y.shape == x.shape and y.dtype == np.float32
    assert np.abs(y).max() == pytest.approx(1.0, abs=1e-3)


def test_numpy_fallback_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(CH, '_FFMPEG_OK', False)
    x = _speech(2)
    wide, _, info = _run(x, 1, eq_prob=0.0, bandlimit_prob=0.0,
                         codec_prob=0.0, clip_prob=0.0, loss_prob=0.0)
    assert info == dict(codec=None, codec_applied=False)
    assert np.allclose(wide, x / (np.abs(x).max() + 1e-6) * 0.9 /
                       (0.9 + 1e-4), atol=1e-5)
    narrow, _, info = _run(x, 1, codec_prob=1.0, codecs=('mulaw8k', ),
                           **ASYM)
    assert info == dict(codec='mulaw8k', codec_applied=False)
    spec = np.abs(np.fft.rfft(narrow))
    assert spec[len(spec) * 3 // 4:].sum() < 0.01 * spec.sum()
    clipped, _, _ = _run(x, 1, eq_prob=0.0, bandlimit_prob=0.0,
                         codec_prob=0.0, clip_prob=1.0, loss_prob=1.0,
                         loss_rate=(0.5, 0.5))
    frames = clipped[:len(clipped) // 320 * 320].reshape(-1, 320)
    assert 0.3 < (np.abs(frames).max(1) == 0).mean() < 0.7
    assert CH._mulaw(np.array([0.0, 0.5, -1.0])) == pytest.approx(
        [0.0, 0.5, -1.0], abs=0.02)


def test_row_selection_and_distinct_codecs(monkeypatch):
    calls = []

    def fake(audio, sr, info=None, **kw):
        calls.append(kw)
        info.update(codec=kw['codecs'][0], codec_applied=True)
        return audio * 0.5

    monkeypatch.setattr(CH, 'apply_channel', fake)
    rows = _rows([3, 3, 3, 3, 1, 1], n=24)
    rows[3]['pair_role'], rows[9]['pair_role'] = 1, 2  # anchor, view
    rows[7]['zr_bypass'] = 1  # silent crop
    before = [r['wav'].clone() for r in rows]
    random.seed(3)
    out = list(CH.add_channel_asym(iter(rows), arms=[3], prob=1.0,
                                   kpos=(1, ), distinct=True,
                                   resample_rate=SR, eq_prob=0.25))
    chained = [i for i, r in enumerate(out) if 'chain' in r]
    want = [i for i in range(24)
            if i % 6 < 4 and i % 2 == 1 and i not in (3, 7)]
    assert chained == want and len(calls) == len(want)
    for i, r in enumerate(out):
        if i in chained:
            assert torch.equal(r['wav'], before[i] * 0.5)
            assert r['chain'] in CH.CHAIN_CODECS and 'chain_fail' not in r
        else:
            assert torch.equal(r['wav'], before[i]) and 'chain' not in r
    assert all(c['codec_prob'] == 1.0 and c['eq_prob'] == 0.25 and len(
        c['codecs']) == 1 for c in calls)
    # one decision and one codec draw per chained row, nothing for the rest
    state = random.getstate()
    random.seed(3)
    for _ in want:
        random.random()
        random.choice(list(CH.CHAIN_CODECS))
    assert random.getstate() == state
    # every position chained: codecs differ inside a speaker group
    random.seed(4)
    both = list(CH.add_channel_asym(iter(_rows([3], n=400)), arms=[3],
                                    kpos=(0, 1)))
    assert all(a['chain'] != b['chain']
               for a, b in zip(both[0::2], both[1::2]))
    assert len({r['chain'] for r in both}) == 5
    none = list(CH.add_channel_asym(iter(_rows([3], n=40)), arms=[3],
                                    prob=0.0))
    assert not any('chain' in r for r in none)


def test_failed_codec_is_not_tagged_and_is_counted(monkeypatch, caplog,
                                                   fresh_stats):
    import torchaudio.io
    monkeypatch.setattr(CH, '_FFMPEG_OK', True)
    monkeypatch.setattr(torchaudio.io, 'AudioEffector', Boom)
    x = _speech(3)
    with caplog.at_level(logging.WARNING):
        y, _, info = _run(x, 5, codec_prob=1.0, codecs=('g722', ),
                          eq_prob=0.0, bandlimit_prob=0.0, clip_prob=0.0,
                          loss_prob=0.0)
    assert info == dict(codec='g722', codec_applied=False)
    # the crop passes through (peak-normalised), nothing else changes
    assert np.allclose(y, x / np.abs(x).max(), atol=1e-3)
    assert 'g722 failed, codec not applied' in caplog.text
    assert 'injected codec failure' in caplog.text
    st = fresh_stats()
    assert (st['rows'], st['effector_calls'], st['effector_failures'],
            st['codec_failures']) == (1, 1, 1, 1)
    random.seed(6)
    out = list(CH.add_channel_asym(iter(_rows([3], n=8)), arms=[3], **ASYM))
    for r in out[1::2]:
        assert 'chain' not in r and r['chain_fail'] in CH.CHAIN_CODECS
    for r in out[0::2]:
        assert 'chain' not in r and 'chain_fail' not in r
    assert fresh_stats()['codec_failures'] == 5


def test_failure_warnings_are_rate_limited(monkeypatch, caplog, fresh_stats):
    import torchaudio.io
    monkeypatch.setattr(CH, '_FFMPEG_OK', True)
    monkeypatch.setattr(torchaudio.io, 'AudioEffector', Boom)
    x = _speech(4, 0.5)
    fresh_stats()
    with caplog.at_level(logging.WARNING):
        for i in range(60):
            _run(x, i, codec_prob=1.0, codecs=('mp3', ), **ASYM)
        lines = [r for r in caplog.records if 'codec not applied' in r.msg]
        assert len(lines) == CH._MAX_FAIL_LINES == 20
        assert all(r.levelno == logging.WARNING for r in lines)
        assert CH._FAIL_LOG['suppressed'] == 40
        # after the quiet interval one line reports what was held back
        monkeypatch.setitem(CH._FAIL_LOG, 'last',
                            time.time() - CH._FAIL_LOG_EVERY - 1)
        _run(x, 99, codec_prob=1.0, codecs=('mp3', ), **ASYM)
        lines = [r for r in caplog.records if 'codec not applied' in r.msg]
        assert len(lines) == 21
        assert '(+40 failures not logged)' in lines[-1].getMessage()
        assert 'codec_failures=61' in lines[-1].getMessage()
    assert fresh_stats()['codec_failures'] == 61


def test_require_ffmpeg(monkeypatch):
    cfg = schema.build_section(schema.ChannelCfg, pp.CHANNEL, 'c')
    monkeypatch.setattr(CH, '_FFMPEG_OK', False)
    with pytest.raises(RuntimeError, match='FFmpeg is unavailable'):
        CH.require_ffmpeg(cfg)
    cfg.isolate_ffmpeg = False
    with pytest.raises(RuntimeError, match='FFmpeg is unavailable'):
        CH.require_ffmpeg(cfg)
    # chain off: no probe at all
    probes = []
    monkeypatch.setattr(CH, '_ffmpeg_ok', lambda: probes.append(1) or False)
    cfg.prob = 0.0
    CH.require_ffmpeg(cfg)
    assert probes == []
    monkeypatch.setattr(CH, '_ffmpeg_ok', lambda: True)
    cfg.prob, cfg.isolate_ffmpeg = 1.0, True
    CH.require_ffmpeg(cfg)


def test_channel_args():
    cfg = schema.build_section(schema.ChannelCfg, pp.CHANNEL, 'c')
    kw = CH.channel_args(cfg, pp.ARM_INDEX, SR)
    assert kw == dict(
        arms=[pp.ARM_INDEX['mandarin_t1']], prob=1.0, kpos=(1, ),
        distinct=True, resample_rate=SR, codecs=CH.CHAIN_CODECS, eq_prob=0.6,
        eq_gain_db=6.0, bandlimit_prob=0.4, clip_prob=0.0,
        clip_level=(0.3, 0.7), loss_prob=0.0, loss_rate=(0.02, 0.08),
        isolate_ffmpeg=True)


def test_helper_script_imports_no_package():
    """The helper runs this file as a script: stdlib imports only on top."""
    src = open(FI.__file__).read()
    top = [line for line in src.splitlines()
           if line.startswith(('import ', 'from '))]
    assert not [line for line in top if 'ovsb' in line or 'wespeaker' in line
                or 'torch' in line or 'numpy' in line]
    assert "'--ffx-helper' in sys.argv" in src
    stats = FI.get_stats()
    assert stats['calls'] == (stats['ok'] + stats['child_errors'] +
                              stats['fallbacks'])
    assert FI.stats_line().startswith('calls=')
    assert set('ffx_' + k for k in stats) <= set(CH.get_stats())


# ------------------------------------------------------------- with FFmpeg
def _delta(before):
    after = FI.get_stats()
    return {k: after[k] - before[k] for k in after if after[k] != before[k]}


KW = dict(ASYM, codec_prob=1.0, codecs=('opus', ))


@needs_ffmpeg
@pytest.mark.parametrize('codec', CH.CHAIN_CODECS)
def test_isolated_call_equals_in_process(codec):
    for i, x in enumerate(pp.channel_inputs(7, 6)):
        kw = dict(ASYM, codec_prob=1.0, codecs=(codec, ))
        a, ra, ia = _run(x, 50 + i, **kw)
        b, rb, ib = _run(x, 50 + i, isolate_ffmpeg=True, **kw)
        assert a.tobytes() == b.tobytes() and ra == rb
        assert ia == ib == dict(codec=codec, codec_applied=True)
        assert a.shape == x.shape and not np.array_equal(a, x)
    assert FI.helper_pid() is not None


@needs_ffmpeg
def test_helper_runs_from_this_package():
    FI.shutdown()
    before = FI.get_stats()
    _run(_speech(5), 1, isolate_ffmpeg=True, **KW)
    pid = FI.helper_pid()
    assert _delta(before) == dict(calls=1, ok=1, starts=1)
    with open('/proc/%d/cmdline' % pid, 'rb') as f:
        cmd = f.read().split(b'\0')
    assert os.path.realpath(cmd[2].decode()) == os.path.realpath(FI.__file__)
    assert b'--ffx-helper' in cmd
    FI.shutdown()
    assert FI.helper_pid() is None


@needs_ffmpeg
def test_crash_is_retried_once():
    x = _speech(6)
    want = _run(x, 5, **KW)
    FI.set_inject(['segv'])
    before = FI.get_stats()
    got = _run(x, 5, isolate_ffmpeg=True, **KW)
    d = _delta(before)
    assert got[0].tobytes() == want[0].tobytes() and got[1] == want[1]
    assert got[2]['codec_applied']
    assert d['crashes'] == 1 and d['retries'] == 1 and d['retry_ok'] == 1
    assert d.get('fallbacks', 0) == 0


@needs_ffmpeg
def test_two_crashes_fall_back_untagged(monkeypatch, caplog, fresh_stats):
    import torchaudio.io
    x = _speech(7)
    normal = _run(x, 6, **KW)
    with monkeypatch.context() as m:
        m.setattr(torchaudio.io, 'AudioEffector', Boom)
        want = _run(x, 6, **KW)  # in-process failure branch
    _reset_stats()
    FI.set_inject(['segv', 'segv'])
    before = FI.get_stats()
    with caplog.at_level(logging.WARNING):
        got = _run(x, 6, isolate_ffmpeg=True, **KW)
    d = _delta(before)
    assert got[0].tobytes() == want[0].tobytes() and got[1] == want[1]
    assert got[1] == normal[1]  # same draws as a successful call
    assert got[0].tobytes() != normal[0].tobytes()
    assert got[2] == dict(codec='opus', codec_applied=False)
    assert d['crashes'] == 2 and d['fallbacks'] == 1
    assert 'opus failed, codec not applied' in caplog.text
    st = fresh_stats()
    assert st['codec_failures'] == 1 and st['ffx_fallbacks'] >= 1
    # through the stage: the row is tagged as failed, not as chained
    rows = _rows([3], n=2, seed=1)
    FI.set_inject(['segv', 'segv'])
    out = list(CH.add_channel_asym(iter(rows), arms=[3], codecs=('opus', ),
                                   isolate_ffmpeg=True, **ASYM))
    assert 'chain' not in out[1] and out[1]['chain_fail'] == 'opus'
    # the next call gets a fresh helper and the normal output
    again = _run(x, 6, isolate_ffmpeg=True, **KW)
    assert again[0].tobytes() == normal[0].tobytes()


@needs_ffmpeg
def test_hang_times_out_and_is_retried():
    x = _speech(8)
    want = _run(x, 9, **KW)
    FI.set_inject(['hang'])
    before = FI.get_stats()
    t0 = time.time()
    got = _run(x, 9, isolate_ffmpeg=True, isolate_timeout=2.0, **KW)
    d = _delta(before)
    assert got[0].tobytes() == want[0].tobytes()
    assert d['timeouts'] == 1 and d['retry_ok'] == 1
    assert 1.5 < time.time() - t0 < 60


@needs_ffmpeg
def test_helper_killed_while_idle_is_restarted():
    x = _speech(9)
    want = _run(x, 11, **KW)
    _run(x, 12, isolate_ffmpeg=True, **KW)
    os.kill(FI.helper_pid(), signal.SIGSEGV)
    time.sleep(0.5)
    before = FI.get_stats()
    got = _run(x, 11, isolate_ffmpeg=True, **KW)
    d = _delta(before)
    assert got[0].tobytes() == want[0].tobytes()
    assert d['crashes'] == 1 and d['retry_ok'] == 1


@needs_ffmpeg
def test_python_error_in_the_helper_is_not_retried():
    before = FI.get_stats()
    with pytest.raises(FI.EffectorError):
        FI.IsolatedEffector(format='wav', encoder='no_such_enc').apply(
            torch.zeros(1600, 1), SR)
    d = _delta(before)
    assert d['child_errors'] == 1 and d.get('retries', 0) == 0


@needs_ffmpeg
def test_forked_process_gets_its_own_helper():
    x = _speech(10)
    _run(x, 13, isolate_ffmpeg=True, **KW)
    mine = FI.helper_pid()
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            _run(x, 13, isolate_ffmpeg=True, **KW)
            stats = FI.get_stats()
            os.write(w, ('%d %d' % (FI.helper_pid(),
                                    stats['calls'])).encode())
        finally:
            os._exit(0)
    os.close(w)
    child_helper, child_calls = (int(v) for v in os.read(r, 64).split())
    os.waitpid(pid, 0)
    assert child_helper and child_helper != mine
    assert child_calls == 1  # counters are per process
    assert FI.helper_pid() == mine


class _ChainedRows(torch.utils.data.IterableDataset):
    """Rows through the isolated chain; reports the worker's helper pid."""

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rows = _rows([3], n=6, seed=info.id)
        for row in CH.add_channel_asym(iter(rows), arms=[3],
                                       isolate_ffmpeg=True, **ASYM):
            yield dict(wav=row['wav'], worker=info.id, pid=os.getpid(),
                       chained=int('chain' in row),
                       helper=FI.helper_pid() or 0,
                       ok=CH.get_stats()['ffx_ok'])


@needs_ffmpeg
def test_each_loader_worker_owns_a_helper():
    FI.shutdown()
    loader = torch.utils.data.DataLoader(_ChainedRows(), batch_size=6,
                                         num_workers=2)
    batches = list(loader)
    assert len(batches) == 2
    helpers = set()
    for b in batches:
        assert b['chained'].tolist() == [0, 1, 0, 1, 0, 1]
        assert torch.isfinite(b['wav']).all()
        assert len(set(b['pid'].tolist())) == 1
        assert b['ok'].tolist() == [0, 1, 1, 2, 2, 3]  # worker counters
        helpers.add(b['helper'][-1].item())
        assert b['helper'][-1].item() not in (0, b['pid'][0].item())
    assert len(helpers) == 2
    assert FI.helper_pid() is None  # the main process started none


# ------------------------------------------------------------------ parity
@pytest.fixture(scope='module')
def parity(tmp_path_factory):
    root = str(tmp_path_factory.mktemp('channel'))
    lmdbs = pp.make_aug_lmdbs(os.path.join(root, 'lmdb'))
    t1 = pp.ARM_INDEX['mandarin_t1']
    chain = dict(
        stages=['filter', 'spk_to_id', 'resample', 'crop', 'zero_run',
                'draw', 'reverb_noise', 'channel'],
        batch_size=48, crop_seed=42, epoch=1, crop=pp.CROP,
        zero_run=pp.ZERO_RUN, filter=pp.FILTER, augment=pp.augment_cfg(lmdbs),
        channel=pp.CHANNEL)
    specs = [
        dict(kind='apply_channel', name='codecs', seed=100, input_seed=5,
             n_inputs=10, codecs=list(CH.CHAIN_CODECS),
             isolate=[False, True], channel_args=ASYM, repeats=2),
        dict(kind='apply_channel', name='clip_loss', seed=200, input_seed=6,
             n_inputs=6, codecs=list(CH.CHAIN_CODECS), isolate=[True],
             channel_args=dict(eq_prob=1.0, bandlimit_prob=0.5,
                               clip_prob=0.5, loss_prob=0.5), repeats=2),
        dict(chain, name='chain', seed=51,
             stream=dict(seed=41, n=600, arms=[t1, t1, t1, 0, 3, 7])),
        dict(chain, name='chain_all_positions', seed=52, epoch=2,
             channel=dict(pp.CHANNEL, kpos=[0, 1], prob=0.7,
                          isolate_ffmpeg=False,
                          codecs=['opus', 'mp3', 'g722']),
             stream=dict(seed=42, n=300, arms=[t1, t1, 0])),
    ]
    source, new = pp.run_both(specs, root)
    return {s['name']: (a, b) for s, a, b in zip(specs, source, new)}


@needs_source
@pytest.mark.parametrize('name', ['codecs', 'clip_loss'])
def test_codec_parity_with_source(parity, name):
    source, new = parity[name]
    assert len(new['calls']) == len(source['calls']) >= 30
    seen = collections.Counter()
    for a, b in zip(source['calls'], new['calls']):
        assert (a['codec'], a['input'], a['isolate']) == (
            b['codec'], b['input'], b['isolate'])
        # reproducible run to run inside each tree, then equal across
        assert a['out'][0].tobytes() == a['out'][1].tobytes()
        assert b['out'][0].tobytes() == b['out'][1].tobytes()
        assert a['out'][0].dtype == b['out'][0].dtype == np.float32
        assert a['out'][0].tobytes() == b['out'][0].tobytes()
        assert a['rng'] == b['rng']
        seen[a['codec'], a['isolate']] += 1
    assert len(seen) == 5 * (2 if name == 'codecs' else 1)


@needs_source
@pytest.mark.parametrize('name', ['chain', 'chain_all_positions'])
def test_chain_parity_with_source(parity, name):
    source, new = parity[name]
    assert len(new['rows']) >= 150
    assert pp.diff_rows(source['rows'], new['rows']) == []
    assert source['rng'] == new['rng']
    chains = collections.Counter(r.get('chain') for r in new['rows'])
    assert not any('chain_fail' in r for r in new['rows'])
    want = 5 if name == 'chain' else 3
    assert len([c for c in chains if c]) == want and chains[None] > 0
    assert min(v for c, v in chains.items() if c) >= 5
