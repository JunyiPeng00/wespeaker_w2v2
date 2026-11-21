"""EpisodicSampler: cells, slots and quotas, speaker cycles, audio, logging.

The parity tests at the end compare the full draw stream with the SOURCE
sampler (see ``data_parity`` for the environment variables). The source
sampler runs in a subprocess with the source tree on PYTHONPATH and dumps,
per sampler cell, an int64 array (episodes, P * K, len(FIELDS)) plus its
sha256; the port's array must hash to the same digest.

  OVSB_SAMPLER_CACHE   directory for the dumps (default <tmp>/ovsb_samp)
"""

import collections
import copy
import dataclasses
import gc
import hashlib
import itertools
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_parity as dp  # noqa: E402

from ovsb.data import sampler as sm  # noqa: E402
from ovsb.data.index import build_index, decode_wav  # noqa: E402
from ovsb.data.policies import PT, PTYPES  # noqa: E402
from ovsb.data.speakers import SpeakerTable  # noqa: E402
from ovsb.data.weights import build_weights  # noqa: E402

CACHE = os.environ.get('OVSB_SAMPLER_CACHE') or os.path.join(
    tempfile.gettempdir(), 'ovsb_samp')
# per-row integer fields of the compared stream (EpisodicSampler.describe)
FIELDS = ('label', 'row', 'spk', 'arm', 'lang', 'vid', 'kpos', 'ptype', 'seg',
          'dev', 'dcls', 'dial', 'genre')
# the fields a source sampler without a construction block emits
PLAIN_FIELDS = ('label', 'row', 'spk', 'arm', 'lang', 'vid')
INT_SAMPLE_FIELDS = ('sample_rate', 'arm', 'lang', 'vid', 'kpos', 'ptype',
                     'seg', 'dev', 'dcls', 'dial', 'genre')


# ----------------------------------------------------------------------------
# fixtures and helpers


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    t = dp.make_toy(tmp_path_factory.mktemp('toy_sampler'))
    build_index(t['files']['shard.list'], os.path.join(t['root'], 'index'),
                nj=1, root=t['root'])
    t['table'], t['weights'] = dp.toy_table(t)
    return t


def make_sampler(toy, episodic=None, seed=3, index_root=None, **kw):
    """The port's sampler for a (modified) toy ``episodic_sampler`` block"""
    ep = toy['episodic'] if episodic is None else episodic
    spec = dp.specs_from_episodic(ep, toy['root'])
    if index_root is not None:
        spec['index_root'] = index_root
    table = SpeakerTable.from_files(toy['files']['utt2spk'],
                                    min_len=dp.TOY_MIN_LEN, **spec)
    kw.setdefault('log_every', 0)
    return sm.EpisodicSampler(table, build_weights(table), ep['P'], ep['K'],
                              seed=seed, **kw)


def variant(toy, **changes):
    """Deep copy of the toy block with dotted-path changes (None = delete)"""
    ep = copy.deepcopy(toy['episodic'])
    for path, value in changes.items():
        d = ep
        keys = path.split('__')
        for k in keys[:-1]:
            d = d[k]
        if value is None:
            d.pop(keys[-1], None)
        else:
            d[keys[-1]] = value
    return ep


def groups(S, n_ep, epoch=1, rank=0, world=1, wid=0, nw=1):
    """[(arm name, speaker id, positions, ptype)] of n_ep episodes"""
    t = S.table
    out = []
    for ep in S.iter_episodes(epoch, rank, world, wid, nw, n_ep):
        for i in range(0, len(ep.pos), S.K):
            out.append((t.arm_names[ep.arm[i]], int(t.spk_of[ep.pos[i]]),
                        ep.pos[i:i + S.K].tolist(), int(ep.ptype[i])))
    return out


def stream(S, cell, fields=FIELDS):
    """(episodes, P * K, len(fields)) int64 of cell = (epoch, rank, world,
    worker, workers, episodes)"""
    out = []
    for ep in S.iter_episodes(*cell):
        d = S.describe(ep)
        out.append(np.stack([d[f] for f in fields], 1))
    return np.stack(out).astype('<i8')


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, '<i8').tobytes()
                          ).hexdigest()


def arm_id(table, name):
    return table.arm_names.index(name)


# ----------------------------------------------------------------------------
# quotas and slots


def test_stratified_counts_sum_and_exact_expectation():
    w = np.array([0.1684, 0.1439, 0.0539, 0.0748, 0.1489, 0.0401, 0.08, 0.25,
                  0.04])
    rng = np.random.default_rng(0)
    tot = np.zeros(len(w))
    n = 20000
    for _ in range(n):
        c = sm.stratified_counts(24, w, rng)
        assert c.sum() == 24 and (c >= np.floor(24 * w / w.sum())).all()
        assert (c <= np.ceil(24 * w / w.sum())).all()
        tot += c
    assert np.abs(tot / n - 24 * w / w.sum()).max() < 0.02


def test_stratified_counts_zero_weight_and_rng_use():
    rng = np.random.default_rng(1)
    for _ in range(200):
        c = sm.stratified_counts(7, [0.5, 0.0, 0.3, 0.0, 0.2], rng)
        assert c[1] == 0 and c[3] == 0 and c.sum() == 7
    # integral quotas: no random number is consumed
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    assert sm.stratified_counts(10, [0.5, 0.3, 0.2], a).tolist() == [5, 3, 2]
    assert a.random() == b.random()
    # a remainder consumes exactly one
    sm.stratified_counts(10, [0.55, 0.25, 0.2], a)
    b.random()
    assert a.random() == b.random()


def test_slot_weights_take_the_share_from_the_arm(toy):
    t = toy['table']
    w = sm.slot_weights(t.arm_weights, t.share_cells)
    wn = t.arm_weights / t.arm_weights.sum()
    vb2 = arm_id(t, 'vb2')
    assert len(w) == len(wn) + 1 and abs(w.sum() - 1) < 1e-12
    assert w[-1] == 0.04 and abs(w[vb2] - (wn[vb2] - 0.04)) < 1e-15
    keep = np.arange(len(wn)) != vb2
    assert np.array_equal(w[:-1][keep], wn[keep])
    assert np.array_equal(sm.slot_weights([2.0, 6.0], ()), [0.25, 0.75])


def test_constructor_checks(toy):
    t, w = toy['table'], toy['weights']
    with pytest.raises(ValueError, match='table K'):
        sm.EpisodicSampler(t, w, 12, 3)
    with pytest.raises(ValueError, match='systematic'):
        sm.EpisodicSampler(t, w, 12, 2, stratify='choice')
    S = sm.EpisodicSampler(t, w, 12, 2)
    # default epoch length: the lines of the label file
    assert S.sample_num_per_epoch == len(toy['label'])
    S = sm.EpisodicSampler(t, w, 24, 2, sample_num_per_epoch=1400000)
    assert S.episodes_per_worker(8, 12) == 304
    assert S.slot_names[-1] == 'vb2_zh' and S.slot_arm[-1] == arm_id(t, 'vb2')


# ----------------------------------------------------------------------------
# cells


def test_cells_partition_the_speakers_and_change_with_the_epoch(toy):
    S = make_sampler(toy)
    n = S.table.n_spk
    cells = [S.cell_speakers(1, r, 2, w, 3) for r in range(2)
             for w in range(3)]
    allspk = np.concatenate(cells)
    assert sorted(allspk.tolist()) == list(range(n))
    assert max(map(len, cells)) - min(map(len, cells)) <= 1
    assert all((np.diff(c) > 0).all() for c in cells)
    again = [S.cell_speakers(1, r, 2, w, 3) for r in range(2)
             for w in range(3)]
    assert all(np.array_equal(a, b) for a, b in zip(cells, again))
    other = S.cell_speakers(2, 0, 2, 0, 3)
    assert not np.array_equal(other, cells[0])
    # the split is the permutation keyed (seed, epoch, 7), dealt round-robin
    perm = np.random.default_rng((3, 1, 7)).permutation(n)
    assert np.array_equal(cells[4], np.sort(perm[4::6]))
    assert np.array_equal(S.cell_speakers(1, 0, 1, 0, 1), np.arange(n))


def test_workers_draw_only_from_their_cell(toy):
    S = make_sampler(toy)
    seen = []
    for r in range(2):
        for w in range(3):
            cell = set(S.cell_speakers(1, r, 2, w, 3).tolist())
            spk = {s for _, s, _, _ in groups(S, 40, 1, r, 2, w, 3)}
            assert spk <= cell
            seen.append(spk)
    for a, b in itertools.combinations(seen, 2):
        assert not a & b


def test_rank_split_arm_is_shared_by_the_workers_of_a_rank(toy):
    """A rank-split arm: every worker of a rank holds the arm's speakers of
    all cells of the rank; other arms and the default are untouched."""
    plain = make_sampler(toy)
    t = plain.table
    name = t.arm_names[int(np.bincount(t.spk_arm[t.spk_arm >= 0]).argmax())]
    a = arm_id(t, name)
    S = make_sampler(toy, rank_split_arms=[name])
    assert S.rank_split_arms == (name, )
    per_rank = []
    for r in range(2):
        old = [plain.cell_speakers(1, r, 2, w, 3) for w in range(3)]
        new = [S.cell_speakers(1, r, 2, w, 3) for w in range(3)]
        union = np.concatenate(old)
        want = np.sort(union[t.spk_arm[union] == a])
        assert len(want) > max(int((t.spk_arm[c] == a).sum()) for c in old)
        for o, n in zip(old, new):
            assert (np.diff(n) > 0).all()
            assert np.array_equal(n[t.spk_arm[n] == a], want)
            assert np.array_equal(n[t.spk_arm[n] != a], o[t.spk_arm[o] != a])
        per_rank.append(set(want.tolist()))
        # the draws of the arm stay inside the rank, other arms in the cell
        for w in range(3):
            for arm, s, _, _ in groups(S, 40, 1, r, 2, w, 3):
                if arm == name:
                    assert s in per_rank[-1]
                else:
                    assert s in set(old[w].tolist())
    assert not per_rank[0] & per_rank[1]
    assert len(per_rank[0] | per_rank[1]) == int((t.spk_arm == a).sum())
    # one rank, one worker: nothing to share, the stream is the default one
    cell = (1, 0, 1, 0, 1, 20)
    assert digest(stream(S, cell)) == digest(stream(plain, cell))
    with pytest.raises(ValueError, match='rank_split_arms'):
        make_sampler(toy, rank_split_arms=['nope'])


def test_rank_split_removes_the_repeats_of_a_small_arm(toy):
    """The repeats of small cells shrink for a rank-split arm: the rank
    holds more speakers of the arm than any of its cells."""
    ep = variant(toy, P=40)

    def repeats(S):
        total = np.zeros(len(S.table.arm_names), np.int64)
        for r in range(2):
            for w in range(3):
                list(S.iter_episodes(1, r, 2, w, 3, 60))
                total += S.stats.repeat_speakers
        return total

    plain = make_sampler(toy, ep)
    before = repeats(plain)
    worst = int(before.argmax())
    assert before[worst] > 0
    after = repeats(make_sampler(
        toy, ep, rank_split_arms=[plain.table.arm_names[worst]]))
    assert after[worst] < before[worst]


# ----------------------------------------------------------------------------
# episodes


def test_episode_layout(toy):
    S = make_sampler(toy)
    t = S.table
    for ep in S.iter_episodes(1, 0, 1, 0, 1, 50):
        assert all(len(x) == S.P * S.K and x.dtype == np.int64 for x in ep)
        spk = t.spk_of[ep.pos].reshape(S.P, S.K)
        assert (spk == spk[:, :1]).all()  # K consecutive rows per speaker
        assert len(set(spk[:, 0].tolist())) == S.P  # distinct speakers
        assert np.array_equal(ep.arm, t.spk_arm[t.spk_of[ep.pos]])
        assert np.array_equal(ep.kpos, np.tile(np.arange(S.K), S.P))
        pt = ep.ptype.reshape(S.P, S.K)
        assert (pt == pt[:, :1]).all() and pt.max() < len(PTYPES)
        assert (np.diff(ep.arm) >= 0).sum() >= len(ep.arm) - 2  # by slot
        d = S.describe(ep)
        assert set(FIELDS) <= set(d)
        assert all(v.shape == (S.P * S.K, ) for v in d.values())
        assert np.array_equal(t.class_names[d['label']],
                              t.spk_names[d['spk']])
    st = S.stats
    assert st.episodes == 50 and st.spk == 50 * S.P
    assert st.arm_draws.sum() == 50 * S.P == st.ptype.sum()
    assert not st.repeat_speakers.any() and not st.empty_slot_episodes.any()
    no_arm = int(np.flatnonzero(t.spk_names == 'zzz_0001')[0])
    assert all(s != no_arm for _, s, _, _ in groups(S, 50))


def test_streams_are_reproducible_and_keyed_by_cell(toy):
    S = make_sampler(toy)
    a = stream(S, (1, 0, 2, 1, 3, 30))
    assert np.array_equal(a, stream(S, (1, 0, 2, 1, 3, 30)))
    assert np.array_equal(a[:10], stream(S, (1, 0, 2, 1, 3, 10)))
    assert digest(a) != digest(stream(S, (2, 0, 2, 1, 3, 30)))
    assert digest(a) != digest(stream(S, (1, 1, 2, 1, 3, 30)))
    assert digest(a) != digest(stream(S, (1, 0, 2, 2, 3, 30)))
    assert digest(a) != digest(stream(make_sampler(toy, seed=4),
                                      (1, 0, 2, 1, 3, 30)))
    # defaults: the epoch of set_epoch, this process as rank 0 / worker 0
    S.set_epoch(5)
    b = np.stack([ep.pos for ep in S.iter_episodes(max_episodes=5)])
    c = np.stack([ep.pos for ep in S.iter_episodes(5, 0, 1, 0, 1, 5)])
    assert np.array_equal(b, c)


def test_stream_id_zero_is_the_default_and_others_differ(toy):
    S = make_sampler(toy)
    cell = (1, 1, 2, 2, 3, 20)
    base = stream(S, cell)
    spk0 = S.cell_speakers(1, 1, 2, 2, 3)
    S.set_epoch(1, stream=0)
    assert np.array_equal(base, stream(S, cell))
    assert S._key(1, 7) == (3, 1, 7) and S._key(1, 1, 2) == (3, 1, 1, 2)
    assert S._key(1, 1, 2, 1) == (3, 1, 1, 2, 1)
    S.set_epoch(1, stream=2)
    keys = [S._key(1, 7), S._key(1, 0, 0), S._key(1, 0, 0, 1),
            S._key(1, 7, 0), S._key(1, 7, 0, 1)]
    assert len(set(keys)) == 5 and all(len(k) == 6 for k in keys)
    assert not np.array_equal(spk0, S.cell_speakers(1, 1, 2, 2, 3))
    assert digest(base) != digest(stream(S, cell))
    S.set_epoch(1)
    assert S.stream == 0 and np.array_equal(base, stream(S, cell))


def test_arm_shares_follow_the_weights(toy):
    S = make_sampler(toy)
    t = S.table
    list(S.iter_episodes(1, 0, 1, 0, 1, 3000))
    share = S.stats.arm_draws / S.stats.arm_draws.sum()
    assert np.abs(share - t.arm_weights / t.arm_weights.sum()).max() < 0.003


def test_uniform_arm_cycles_through_all_speakers(toy):
    # P 4 x 0.2: at most one vox2 speaker per episode, so no collision
    # (and no redraw) can break a cycle
    S = make_sampler(toy, variant(toy, P=4))
    spk = S.table.arm_speakers(arm_id(S.table, 'vox2'))
    drawn = [s for a, s, _, _ in groups(S, 400) if a == 'vox2']
    n = len(spk)
    assert len(drawn) > 10 * n
    for i in range(0, len(drawn) - n, n):
        assert sorted(drawn[i:i + n]) == spk.tolist()


def test_share_cell_gets_its_share(toy):
    S = make_sampler(toy)
    t = S.table
    zh = t.share_cells[0].mask
    n_ep = 20000 // S.P + 1
    arm = collections.Counter()
    n_zh = 0
    for a, s, _, _ in groups(S, n_ep):
        arm[a] += 1
        n_zh += bool(zh[s])
    tot = sum(arm.values())
    assert abs(n_zh / tot - 0.04) < 0.005
    assert abs(arm['vb2'] / tot - 0.2) < 0.005  # financed inside the arm
    st = S.cell_stats()['vb2_zh']
    assert st == dict(draws=n_zh, share=round(n_zh / tot, 4), target=0.04)
    assert S.stats.cell_draws.tolist() == [n_zh]
    # without the cell the zh speakers get their population share of vb2
    S0 = make_sampler(toy, variant(toy, cells=None))
    n0 = sum(bool(zh[s]) for _, s, _, _ in groups(S0, n_ep))
    assert abs(n0 / tot - 0.2 * 3 / 6) < 0.01
    assert S0.cell_stats() == {} and len(S0.slot_arm) == len(t.arm_names)


def test_bridge_weight_is_kept_under_collisions(toy):
    # P 30: 3 cn2 speakers per episode out of 8 (4 of weight 2), so a
    # weight-2 speaker often meets itself; the colliding pool entry is
    # deferred to a later episode and the realised ratio stays 2
    S = make_sampler(toy, variant(toy, P=30))
    t = S.table
    cnt = collections.Counter(s for a, s, _, _ in groups(S, 3000)
                              if a == 'cn2')
    cn = t.arm_speakers(arm_id(t, 'cn2'))
    br = np.mean([cnt[s] for s in cn if t.spk_ngenre[s] >= 2])
    sg = np.mean([cnt[s] for s in cn if t.spk_ngenre[s] < 2])
    assert 1.9 < br / sg < 2.1, br / sg


def test_dialect_mass_shares(toy):
    S = make_sampler(toy, variant(toy, P=4))
    t = S.table
    cnt = collections.Counter(s for a, s, _, _ in groups(S, 12000)
                              if a == '3dspeaker')
    d3 = t.arm_speakers(arm_id(t, '3dspeaker'))
    m12 = sum(cnt[s] for s in d3 if t.spk_dial[s] == 12)
    m9 = sum(cnt[s] for s in d3 if t.spk_dial[s] == 9)
    std = np.mean([cnt[s] for s in d3 if t.spk_dial[s] == 0])
    assert 0.85 < m12 / m9 < 1.15 and 3.8 < m12 / std < 5.2, (m12, m9, std)


def test_weight_file_share_holds_in_every_cell(toy):
    S = make_sampler(toy)
    t = S.table
    listed = S.weights.spk_listed
    a = arm_id(t, 'mandarin_t1')
    hits = collections.Counter()
    for e in range(1, 9):
        for arm, s, _, _ in groups(S, 400, epoch=e):
            if arm == 'mandarin_t1':
                hits[bool(listed[s])] += 1
    assert abs(hits[True] / sum(hits.values()) - 0.5) < 0.02
    # two cells: whatever the listed speakers of the cell, their share of
    # the arm's draws in the cell is the target
    for wid in range(2):
        cell = S.cell_speakers(1, 0, 1, wid, 2)
        mine = cell[t.spk_arm[cell] == a]
        if not listed[mine].any() or listed[mine].all():
            continue
        h = collections.Counter(bool(listed[s]) for arm, s, _, _ in
                                groups(S, 1500, 1, 0, 1, wid, 2)
                                if arm == 'mandarin_t1')
        assert abs(h[True] / sum(h.values()) - 0.5) < 0.03, (wid, h)


def test_small_cells_repeat_speakers_and_redistribute_quotas(toy):
    """The kept limitation: a slot with fewer speakers in the cell than its
    quota repeats speakers inside an episode; a slot without any speaker in
    the cell gives its quota to the others. Both are counted."""
    S = make_sampler(toy, variant(toy, P=40))
    t = S.table
    n_arms = len(t.arm_names)
    total_rep = total_empty = 0
    for r in range(2):
        for w in range(3):
            eps = list(S.iter_episodes(1, r, 2, w, 3, 60))
            st = S.stats
            cell = S.cell_speakers(1, r, 2, w, 3)
            # recount the repeats from the episodes
            rep = np.zeros(n_arms, np.int64)
            for ep in eps:
                spk = t.spk_of[ep.pos[::S.K]]
                arm = ep.arm[::S.K]
                assert len(spk) == S.P
                for a in range(n_arms):
                    x = spk[arm == a]
                    rep[a] += len(x) - len(set(x.tolist()))
            assert np.array_equal(rep, st.repeat_speakers)
            # empty slots: weight > 0, no speaker of the slot in the cell
            for a in range(n_arms):
                mine = cell[t.spk_arm[cell] == a]
                if a == t.share_cells[0].arm:
                    mine = mine[~t.share_cells[0].mask[mine]]
                want = 60 if len(mine) == 0 else 0
                assert st.empty_slot_episodes[a] == want
                assert (st.arm_draws[a] == 0) == (
                    len(cell[t.spk_arm[cell] == a]) == 0)
            e = st.empty_slot_episodes > 0
            w_slot = sm.slot_weights(t.arm_weights, t.share_cells)
            assert np.allclose(st.redistributed, 60 * S.P * w_slot * e)
            assert st.arm_draws.sum() == 60 * S.P  # quotas still sum to P
            total_rep += int(rep.sum())
            total_empty += int(e.sum())
    assert total_rep > 0 and total_empty > 0


def test_cell_without_any_weighted_speaker_is_an_error(toy):
    S = make_sampler(toy)
    with pytest.raises(ValueError, match='no speaker'):
        next(S.iter_episodes(1, 39, 40, 39, 40, 1))


# ----------------------------------------------------------------------------
# audio


def test_samples_fields_order_and_audio(toy):
    S = make_sampler(toy)
    t = S.table
    S.set_epoch(1)
    eps = list(S.iter_episodes(1, 0, 1, 0, 1, 2))
    samples = list(S.iter_samples(max_episodes=2))
    assert len(samples) == 2 * S.P * S.K
    pos = np.concatenate([e.pos for e in eps])
    desc = {k: np.concatenate([S.describe(e)[k] for e in eps])
            for k in FIELDS}
    for i, d in enumerate(samples):
        assert tuple(d) == sm.SAMPLE_FIELDS
        key = t.index.key(int(t.rows[pos[i]]))
        assert d['key'] == key and isinstance(d['key'], str)
        assert d['spk'] == str(t.spk_names[desc['spk'][i]])
        assert type(d['spk']) is str and d['sample_rate'] == dp.SR
        wav, sr = decode_wav(toy['blobs'][key])
        assert d['wav'].dtype == torch.float32 and d['wav'].shape[0] == 1
        assert torch.equal(d['wav'], wav) and sr == dp.SR
        for k in INT_SAMPLE_FIELDS[1:]:
            assert type(d[k]) is int and d[k] == desc[k][i], k
    # __iter__ is the stream of this process at the current epoch
    first = next(iter(S))
    assert first['key'] == samples[0]['key']
    assert torch.equal(first['wav'], samples[0]['wav'])


def test_read_error_is_replaced_by_the_same_speaker(toy, caplog):
    S = make_sampler(toy)
    t = S.table
    good = list(S.iter_samples(1, 0, 1, 0, 1, 2))
    bad_pos = [i for i, d in enumerate(good) if d['arm'] == 0][:3]
    bad_keys = {good[i]['key'] for i in bad_pos}
    real = S.reader.load

    def flaky(row):
        if t.index.key(row) in bad_keys:
            raise IOError('broken member')
        return real(row)
    S.reader.load = flaky
    with caplog.at_level(logging.WARNING, logger='ovsb.data.sampler'):
        got = list(S.iter_samples(1, 0, 1, 0, 1, 2))
    assert sum('read error' in r.getMessage() for r in caplog.records) >= 3
    assert len(got) == len(good)
    # the replacements come from the generator keyed (seed, e, r, w, 1)
    rng = np.random.default_rng((3, 1, 0, 0, 1))
    eps = list(S.iter_episodes(1, 0, 1, 0, 1, 2))
    pos = np.concatenate([e.pos for e in eps])
    for i, (g, d) in enumerate(zip(good, got)):
        assert {k: v for k, v in g.items() if k not in ('key', 'wav')} == \
            {k: v for k, v in d.items() if k not in ('key', 'wav')}
        if g['key'] not in bad_keys:
            assert d['key'] == g['key'] and torch.equal(d['wav'], g['wav'])
            continue
        s = int(t.spk_of[pos[i]])
        lo, hi = t.spk_ptr[s], t.spk_ptr[s + 1]
        while True:
            u = int(lo + rng.integers(hi - lo))
            if t.index.key(int(t.rows[u])) not in bad_keys:
                break
        assert d['key'] == t.index.key(int(t.rows[u])) and d['spk'] == g['spk']
    # nothing readable: 5 attempts, then an error naming the speaker
    S.reader.load = lambda row: (_ for _ in ()).throw(IOError('gone'))
    with pytest.raises(IOError, match='5 consecutive read errors'):
        next(S.iter_samples(1, 0, 1, 0, 1, 1))


# ----------------------------------------------------------------------------
# stat lines


def _lines(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == 'ovsb.data.sampler' and 'episodic rank' in
            r.getMessage()]


def test_stat_lines_cadence_and_closing_line(toy, caplog):
    caplog.set_level(logging.WARNING, logger='ovsb.data.sampler')
    S = make_sampler(toy, log_every=3)
    n = S.P * S.K
    it = S.iter_samples(1, 0, 1, 0, 1)
    for _ in range(7 * n + 1):  # 7 full episodes and one row of the 8th
        next(it)
    heads = [x for x in _lines(caplog) if ' episodes, ' in x]
    assert [x.split(': ')[1].split(' ')[0] for x in heads] == ['3', '6']
    assert not any('(end)' in x for x in heads)
    it.close()  # the consumer stops: one closing line
    heads = [x for x in _lines(caplog) if ' episodes, ' in x]
    assert len(heads) == 3 and 'epoch 1 (end): 8 episodes' in heads[-1]
    kinds = [x for x in _lines(caplog) if '(end)' in x]
    assert len(kinds) == 4  # head, repeats / empty slots, types, cells
    assert 'speakers repeated inside an episode {}' in kinds[1]
    assert 'positive types' in kinds[2] and "'vb2_zh'" in kinds[3]

    # log_every beyond the stream (the reference configuration: 500 with
    # about 270 episodes per worker): the closing line is the only one,
    # and it is emitted for every worker, not only worker 0
    caplog.clear()
    S = make_sampler(toy, log_every=500)
    for wid in range(2):
        list(S.iter_samples(1, 0, 1, wid, 2, 4))
    heads = [x for x in _lines(caplog) if ' episodes, ' in x]
    assert len(heads) == 2 and all('(end): 4 episodes' in x for x in heads)
    assert 'worker 0' in heads[0] and 'worker 1' in heads[1]

    # a line that the cadence just emitted is not repeated at the end
    caplog.clear()
    S = make_sampler(toy, log_every=2)
    list(S.iter_samples(1, 0, 1, 0, 1, 4))
    heads = [x for x in _lines(caplog) if ' episodes, ' in x]
    assert len(heads) == 2 and not any('(end)' in x for x in heads)

    # off switches
    caplog.clear()
    list(make_sampler(toy, log_every=0).iter_samples(1, 0, 1, 0, 1, 2))
    list(make_sampler(toy, log_every=500, log_at_end=False).iter_samples(
        1, 0, 1, 0, 1, 2))
    assert not _lines(caplog)


def test_stat_lines_report_repeats_and_empty_slots(toy):
    S = make_sampler(toy, variant(toy, P=40))
    for r in range(2):
        for w in range(3):
            list(S.iter_episodes(1, r, 2, w, 3, 20))
            st = S.stats
            lines = S.stat_lines(st, (r, w, 1), 1.0, end=True)
            for name, x in zip(S.table.arm_names, st.repeat_speakers):
                assert ("'%s': %d" % (name, x) in lines[1]) == bool(x)
            for name, e in zip(S.slot_names, st.empty_slot_episodes):
                assert ("'%s': {'episodes': %d" % (name, e) in lines[1]) == \
                    bool(e)


def test_logging_does_not_change_the_draws(toy):
    quiet = make_sampler(toy, log_every=0)
    loud = make_sampler(toy, log_every=1)
    a = list(quiet.iter_samples(2, 1, 2, 1, 3, 6))
    b = list(loud.iter_samples(2, 1, 2, 1, 3, 6))
    assert [d['key'] for d in a] == [d['key'] for d in b]
    assert [d['ptype'] for d in a] == [d['ptype'] for d in b]


def _take_first_wavs(batch):
    return [d['key'] for d in batch]


def test_closing_line_from_dataloader_workers(toy, tmp_path):
    """The training loop leaves an endless loader with ``break``; every
    worker must still emit its closing stat line."""
    S = make_sampler(toy, log_every=500)
    S.set_epoch(4)
    log = logging.getLogger('ovsb.data.sampler')
    handler = logging.FileHandler(str(tmp_path / 'workers.log'))
    log.addHandler(handler)  # inherited by the forked workers

    class Chain(torch.utils.data.IterableDataset):
        # a downstream generator stage, as the processor chain
        def __iter__(self):
            for d in S:
                yield d

    loader = torch.utils.data.DataLoader(
        Chain(), batch_size=S.P * S.K, num_workers=2, prefetch_factor=2,
        collate_fn=_take_first_wavs,
        multiprocessing_context='fork')
    try:
        it = iter(loader)
        keys = [next(it) for _ in range(6)]
        del it  # as leaving the loop: the workers are shut down
        gc.collect()
    finally:
        log.removeHandler(handler)
        handler.close()
    with open(str(tmp_path / 'workers.log')) as f:
        err = f.read()
    ends = [x for x in err.splitlines() if 'episodes, ' in x]
    assert len(ends) == 2, err
    for wid in range(2):
        line = [x for x in ends if 'worker %d epoch 4 (end)' % wid in x]
        assert len(line) == 1, err
    # workers got their own cells: batch i comes from worker i % 2
    ref = [[d['key'] for d in itertools.islice(
        S.iter_samples(4, 0, 1, w, 2), 3 * S.P * S.K)] for w in range(2)]
    n = S.P * S.K
    for i, batch in enumerate(keys):
        assert batch == ref[i % 2][(i // 2) * n:(i // 2 + 1) * n]


# ----------------------------------------------------------------------------
# parity with the SOURCE sampler

SOURCE_STREAM = r'''
import hashlib, itertools, json, sys
import numpy as np
job = json.load(open(sys.argv[1]))
out = sys.argv[2]
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.utils import spk2id
from wespeaker.dataset.episodic import build_episodic_sampler, PTYPES

FIELDS = job['fields']
utt_spk = read_table(job['train_label'])
classes = spk2id(utt_spk)
arr, variants = {}, []
for vi, v in enumerate(job['variants']):
    ep = v['episodic']
    S = build_episodic_sampler(ep, utt_spk, ep['P'] * ep['K'],
                               v['sample_num_per_epoch'], v['min_len'],
                               v['seed'])
    cls = np.array([classes[str(n)] for n in S.spk_names], np.int64)
    cons = S.cons is not None
    none = np.full(S.n_utts, -1, np.int64)
    cond = {n: getattr(S, n + '_of') if cons else none
            for n in ('seg', 'dev', 'dcls', 'dial', 'genre')}
    cells = []
    for ci, cell in enumerate(v['cells']):
        epoch, rank, world, wid, nw, n_ep = cell
        eps = []
        for e in S.iter_episodes(epoch=epoch, rank=rank, world=world,
                                 wid=wid, nw=nw, max_episodes=n_ep):
            u = np.array([x for x, _ in e], np.int64)
            a = np.array([x for _, x in e], np.int64)
            meta = np.array(S.ep_meta, np.int64) if cons else \
                np.full((len(u), 2), -1, np.int64)
            s = S.spk_of[u]
            d = dict(label=cls[s], row=S.rows[u], spk=s, arm=a,
                     lang=S.spk_lang[s], vid=S.vid_of[u], kpos=meta[:, 0],
                     ptype=meta[:, 1], seg=cond['seg'][u], dev=cond['dev'][u],
                     dcls=cond['dcls'][u], dial=cond['dial'][u],
                     genre=cond['genre'][u])
            eps.append(np.stack([np.asarray(d[f], np.int64) for f in FIELDS],
                                1))
        x = np.ascontiguousarray(np.stack(eps), '<i8')
        name = 'v%d_c%d' % (vi, ci)
        arr[name] = x
        st = S.stats
        cells.append(dict(
            name=name, cell=cell, sha256=hashlib.sha256(
                x.tobytes()).hexdigest(),
            arm_draws=st['arm_draws'].tolist(),
            ptype=st['ptype'].tolist() if cons else None,
            same_video_fill=int(st['same_video_fill']),
            repeat_fill=int(st['repeat_fill']),
            arm_resets=int(st['arm_resets']),
            cell_draws=st['cell_draws'].tolist() if 'cell_draws' in st
            else []))
    samples = []
    for epoch, rank, world, wid, nw, n_ep in v.get('samples', []):
        S.set_epoch(epoch)
        S.dist_info = lambda r=rank, w=world, i=wid, n=nw: (r, w, i, n)
        rows = []
        for d in itertools.islice(iter(S), n_ep * ep['P'] * ep['K']):
            w = d['wav']
            d = dict(d)
            d['wav'] = [hashlib.sha256(w.numpy().tobytes()).hexdigest(),
                        list(w.shape), str(w.dtype)]
            rows.append([[k, d[k]] for k in d])
        del S.dist_info
        samples.append(rows)
    variants.append(dict(cons=cons, cells=cells, samples=samples,
                         arm_names=list(S.arm_names)))
np.savez(out + '.npz', **arr)
json.dump(dict(ptypes=list(PTYPES), variants=variants, numpy=np.__version__),
          open(out + '.json', 'w'))
'''


def source_stream(train_label, variants, cwd, tag):
    """Run the SOURCE sampler on ``variants`` and return (arrays, info).

    variants: [dict(episodic, min_len, seed, sample_num_per_epoch, cells,
    samples)] with cells / samples = [(epoch, rank, world, worker, workers,
    episodes)]. Dumps are cached under a hash of the job.
    """
    path = train_label if os.path.isabs(train_label) else os.path.join(
        cwd, train_label)
    st = os.stat(path)
    job = dict(train_label=train_label, variants=variants, fields=FIELDS)
    blob = json.dumps([job, SOURCE_STREAM, cwd, dp.SOURCE_TREE, st.st_size,
                       st.st_mtime_ns, sys.version, np.__version__])
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, '%s_%s' % (
        tag, hashlib.sha1(blob.encode()).hexdigest()[:16]))
    if not (os.path.isfile(out + '.npz') and os.path.isfile(out + '.json')):
        with open(out + '.job', 'w') as f:
            json.dump(job, f)
        with open(out + '.py', 'w') as f:
            f.write(SOURCE_STREAM)
        env = dict(os.environ, PYTHONPATH=dp.SOURCE_TREE,
                   PYTHONDONTWRITEBYTECODE='1')
        tmp = out + '.tmp%d' % os.getpid()
        subprocess.run(['nice', sys.executable, out + '.py', out + '.job',
                        tmp], cwd=cwd, env=env, check=True)
        os.replace(tmp + '.npz', out + '.npz')
        os.replace(tmp + '.json', out + '.json')
    with open(out + '.json') as f:
        info = json.load(f)
    return np.load(out + '.npz'), info


def first_difference(mine, ref, fields):
    """Text naming the first episode / row / field where two streams differ"""
    if mine.shape != ref.shape:
        return 'shapes %s vs %s' % (mine.shape, ref.shape)
    e, r, f = np.argwhere(mine != ref)[0]
    return 'episode %d row %d field %s: port %d, source %d (%d of %d ' \
        'episodes differ)' % (e, r, fields[f], mine[e, r, f], ref[e, r, f],
                              (mine != ref).any((1, 2)).sum(), len(ref))


def compare_variant(S, src, vinfo):
    """Every cell of one variant: digest and counters equal the source's.

    Returns (cells, episodes) compared; raises on the first mismatch with
    the first differing episode and field.
    """
    cols = list(range(len(FIELDS))) if vinfo['cons'] else \
        [FIELDS.index(f) for f in PLAIN_FIELDS]
    fields = [FIELDS[c] for c in cols]
    assert list(S.table.arm_names) == vinfo['arm_names']
    n_ep = 0
    for c in vinfo['cells']:
        mine = stream(S, tuple(c['cell']))
        ref = src[c['name']]
        if vinfo['cons']:
            same = digest(mine) == c['sha256'] == digest(ref)
        else:
            same = digest(mine[:, :, cols]) == digest(ref[:, :, cols])
        if not same:
            raise AssertionError('cell %s: %s' % (c['cell'], first_difference(
                mine[:, :, cols], ref[:, :, cols], fields)))
        st = S.stats
        assert st.arm_draws.tolist() == c['arm_draws'], c['cell']
        assert st.cell_draws.tolist() == c['cell_draws'], c['cell']
        assert st.fill.same_video_fill == c['same_video_fill'], c['cell']
        assert st.fill.repeat_fill == c['repeat_fill'], c['cell']
        assert c['arm_resets'] == 0
        if vinfo['cons']:
            assert st.ptype.tolist() == c['ptype'], c['cell']
        n_ep += len(ref)
    return len(vinfo['cells']), n_ep


def compare_samples(S, rows, cell):
    """The samples of ``cell`` equal the source's dicts (key order, values,
    waveform sha256 / shape / dtype)"""
    n = 0
    for d, ref in zip(S.iter_samples(*cell), rows):
        assert list(d) == [k for k, _ in ref], (n, list(d))
        for k, v in ref:
            if k == 'wav':
                w = d['wav']
                assert isinstance(w, torch.Tensor)
                got = [hashlib.sha256(w.numpy().tobytes()).hexdigest(),
                       list(w.shape), str(w.dtype)]
                assert got == v, (n, d['key'])
            else:
                assert type(d[k]) is type(v) and d[k] == v, (n, k, d[k], v)
        n += 1
    assert n == len(rows) == cell[5] * S.P * S.K
    return n


TOY_CELLS = [(1, 0, 1, 0, 1, 250), (2, 0, 1, 0, 1, 120), (1, 0, 2, 0, 3, 60),
             (1, 0, 2, 1, 3, 60), (1, 0, 2, 2, 3, 60), (1, 1, 2, 0, 3, 60),
             (1, 1, 2, 1, 3, 60), (1, 1, 2, 2, 3, 60), (3, 1, 2, 1, 3, 60),
             (9, 0, 1, 1, 2, 60)]


def toy_variants(toy):
    """name -> toy sampler block; together they exercise every policy and
    branch of the active path"""
    xs3 = {'pair_policy': 'xsession'}
    return {
        'base': variant(toy),
        # K 3: alternation over 2 groups, more rows than genres, fill-ups
        'k3': variant(toy, K=3, P=8),
        # cross-session draws on 3D-Speaker speakers: segment repair
        'segfix': variant(toy, construction__arms__3dspeaker=xs3),
        'segfix_k3': variant(toy, K=3, P=8,
                             construction__arms__3dspeaker=xs3),
        'twins': variant(toy, construction__arms__3dspeaker=xs3,
                         construction__forbid_same_segment=False),
        'file_first': variant(
            toy, construction__arms__3dspeaker__segment_first=False,
            construction__arms__3dspeaker__p_dialect=1.0,
            construction__arms__kespeech__p_dialect=1.0),
        'no_dialect': variant(
            toy, construction__arms__3dspeaker__p_dialect=0.0,
            construction__arms__kespeech__p_dialect=0.0,
            construction__forbid_same_segment=False),
        # quotas above the speakers of small slots: repeats, deferred
        # entries of weighted arms, 8 retries
        'big_p': variant(toy, P=60),
        'no_cells': variant(toy, cells=None),
        'no_policies': variant(toy, construction=None, cells=None),
        'unweighted': variant(
            toy, construction__arms__cn2__w_bridge=None,
            construction__arms__cn1__w_bridge=None,
            construction__arms__3dspeaker__dialect_mass=None,
            construction__arms__mandarin_t1=None),
    }


@dp.needs_source
def test_parity_toy_streams_equal_source(toy):
    names, blocks = zip(*toy_variants(toy).items())
    jobs = [dict(episodic=ep, min_len=dp.TOY_MIN_LEN, seed=3 + i,
                 sample_num_per_epoch=0, cells=TOY_CELLS,
                 samples=[(1, 0, 1, 0, 1, 3), (2, 1, 2, 2, 3, 2)])
            for i, ep in enumerate(blocks)]
    src, info = source_stream('utt2spk', jobs, toy['root'], 'toy')
    assert tuple(info['ptypes']) == PTYPES
    seen = np.zeros(len(PTYPES), np.int64)
    repeats = 0
    for i, (name, ep) in enumerate(zip(names, blocks)):
        S = make_sampler(toy, ep, seed=3 + i)
        v = info['variants'][i]
        assert v['cons'] == (name != 'no_policies')
        n_cells, n_ep = compare_variant(S, src, v)
        assert n_cells == len(TOY_CELLS) and n_ep == 850
        for c in v['cells']:
            if v['cons']:
                seen += np.array(c['ptype']).sum(0)
        for rows, cell in zip(v['samples'], jobs[i]['samples']):
            if v['cons']:
                compare_samples(S, rows, cell)
        list(S.iter_episodes(1, 0, 2, 1, 3, 60))
        repeats += int(S.stats.repeat_speakers.sum())
    # every positive type of the active path occurred
    active = ['xsession', 'xsession_segfix', '3ds_contrast', 'dialect',
              'cn2_xgenre', 'same_segment', 'ks_xphase', 'ks_dialect']
    assert all(seen[PT[n]] > 0 for n in active), dict(zip(PTYPES, seen))
    assert repeats > 0  # the repeated-speaker path was part of the streams


@dp.needs_source
def test_parity_toy_read_retries_equal_source(toy, tmp_path):
    """Half of one shard is cut off: reads behind the cut fail in both
    samplers and must be replaced by the same utterances"""
    broken = str(tmp_path / 'broken')
    shutil.copytree(os.path.join(toy['root'], 'shards'),
                    os.path.join(broken, 'shards'))
    path = os.path.join(broken, toy['shards'][1])
    os.truncate(path, int(os.path.getsize(path) * 0.7))
    ep = variant(toy, root=broken)
    cell = (1, 0, 1, 0, 1, 6)
    jobs = [dict(episodic=ep, min_len=dp.TOY_MIN_LEN, seed=11,
                 sample_num_per_epoch=0, cells=[], samples=[cell])]
    _, info = source_stream('utt2spk', jobs, toy['root'], 'toy_retry')
    ep.pop('root')
    S = make_sampler(toy, ep, seed=11, index_root=broken)
    n = compare_samples(S, info['variants'][0]['samples'][0], cell)
    # the cut was hit: some keys differ from the rows that were drawn
    drawn = np.concatenate([e.pos for e in S.iter_episodes(*cell)])
    keys = [d['key'] for d in S.iter_samples(*cell)]
    t = S.table
    replaced = sum(k != t.index.key(int(t.rows[u]))
                   for k, u in zip(keys, drawn))
    assert n == len(keys) and replaced > 0


# -- the real training list ---------------------------------------------------

WORLD, WORKERS = 8, 12
FULL_EPOCHS, FULL_RANKS, FULL_WORKERS = (1, 2, 9, 14), (0, 3, 7), (0, 5, 11)
GRID_EPOCHS, GRID_EPISODES = (1, 2), 60
SAMPLE_CELLS = [(1, 0, WORLD, 0, WORKERS, 3), (1, 0, 1, 0, 1, 3)]


def real_cells():
    """(full cells, grid cells, episodes of a worker's epoch)"""
    _label, ep, _min_len, _seed, n_epoch = dp.real_job()
    per_worker = int(math.ceil(n_epoch // WORLD // (ep['P'] * ep['K']) /
                               float(WORKERS)))
    # a few more than a worker can deliver (prefetched batches)
    n_full = per_worker + 16
    full = [(e, r, WORLD, w, WORKERS, n_full) for e in FULL_EPOCHS
            for r in FULL_RANKS for w in FULL_WORKERS]
    grid = [(e, r, WORLD, w, WORKERS, GRID_EPISODES) for e in GRID_EPOCHS
            for r in range(WORLD) for w in range(WORKERS)]
    return full, grid, per_worker


def real_sampler():
    _label, ep, _min_len, seed, n_epoch = dp.real_job()
    table, weights, _ = dp.real_port()
    return sm.EpisodicSampler(table, weights, ep['P'], ep['K'], n_epoch,
                              seed, stratify=ep['stratify'], log_every=0)


def real_stream():
    label, ep, min_len, seed, n_epoch = dp.real_job()
    full, grid, _ = real_cells()
    jobs = [dict(episodic=ep, min_len=min_len, seed=seed,
                 sample_num_per_epoch=n_epoch, cells=full + grid,
                 samples=SAMPLE_CELLS)]
    return source_stream(label, jobs, dp.SOURCE_RECIPE, 'real')


@dp.needs_source
def test_parity_real_streams_equal_source():
    full, grid, per_worker = real_cells()
    src, info = real_stream()
    assert tuple(info['ptypes']) == PTYPES
    S = real_sampler()
    assert S.episodes_per_worker(WORLD, WORKERS) == per_worker
    v = info['variants'][0]
    assert [tuple(c['cell']) for c in v['cells']] == full + grid
    n_cells, n_ep = compare_variant(S, src, v)
    assert n_cells == len(full) + len(grid) == 36 + 192
    assert n_ep == 36 * (per_worker + 16) + 192 * GRID_EPISODES
    print('\n[real list] draw streams equal the source: %d cells, %d '
          'episodes, %d rows x %d fields' % (
              n_cells, n_ep, n_ep * S.P * S.K, len(FIELDS)))


@dp.needs_source
def test_parity_real_samples_equal_source():
    _src, info = real_stream()
    S = real_sampler()
    for rows, cell in zip(info['variants'][0]['samples'], SAMPLE_CELLS):
        assert compare_samples(S, rows, cell) == 3 * S.P * S.K


@dp.needs_source
def test_real_repeat_rates_are_counted():
    """The counters on the real list: epoch 1, a few cells. The small arm
    with the large weight repeats speakers inside episodes (documented
    limitation); no other arm does."""
    S = real_sampler()
    t = S.table
    rep = np.zeros(len(t.arm_names), np.int64)
    draws = np.zeros(len(t.arm_names), np.int64)
    for rank, wid in ((0, 0), (3, 5), (7, 11)):
        n = S.episodes_per_worker(WORLD, WORKERS)
        for ep in S.iter_episodes(1, rank, WORLD, wid, WORKERS, n):
            spk = t.spk_of[ep.pos[::S.K]]
            assert len(spk) - len(set(spk.tolist())) >= 0
        rep += S.stats.repeat_speakers
        draws += S.stats.arm_draws
        assert not S.stats.empty_slot_episodes.any()
    cn1 = arm_id(t, 'cn1')
    assert rep[cn1] > 0 and rep.sum() == rep[cn1]
    assert draws[cn1] / draws.sum() == pytest.approx(0.25, abs=0.002)
    # what-if arm weights go through the same code
    w = t.arm_weights.copy()
    w[cn1] = 0.40
    w[[arm_id(t, n) for n in ('vox2', 'vb2', '3dspeaker')]] -= 0.05
    S2 = sm.EpisodicSampler(dataclasses.replace(t, arm_weights=w), S.weights,
                            S.P, S.K, S.sample_num_per_epoch, S.seed,
                            log_every=0)
    list(S2.iter_episodes(1, 0, WORLD, 0, WORKERS, 100))
    list(S.iter_episodes(1, 0, WORLD, 0, WORKERS, 100))
    assert S2.stats.repeat_speakers[cn1] > S.stats.repeat_speakers[cn1]
