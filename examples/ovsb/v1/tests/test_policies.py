"""Pair policies: the K rows drawn for one speaker (toy corpus).

Parity of the draws with the SOURCE sampler is covered by the stream
comparison in ``test_sampler.py``; here every rule is checked on its own.
"""

import collections
import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_parity as dp  # noqa: E402

from ovsb.data import policies as po  # noqa: E402
from ovsb.data.index import build_index  # noqa: E402
from ovsb.data.policies import PT, PTYPES  # noqa: E402
from ovsb.data.speakers import PolicySpec, SpeakerTable  # noqa: E402


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    t = dp.make_toy(tmp_path_factory.mktemp('toy_policies'))
    build_index(t['files']['shard.list'], os.path.join(t['root'], 'index'),
                nj=1, root=t['root'])
    t['table'], _ = dp.toy_table(t)
    t['cache'] = {}
    return t


def table_for(toy, K=2, forbid=True, **policies):
    """Toy table with another K / forbid_same_segment / arm policies
    (``arm=PolicySpec`` replaces, ``arm=None`` removes the arm's policy)"""
    key = (K, forbid, tuple(sorted((k, repr(v)) for k, v in policies.items())))
    if key not in toy['cache']:
        ep = copy.deepcopy(toy['episodic'])
        ep['K'] = K
        ep['construction']['forbid_same_segment'] = forbid
        spec = dp.specs_from_episodic(ep, toy['root'])
        for arm, pol in policies.items():
            if pol is None:
                spec['policies'].pop(arm, None)
            else:
                spec['policies'][arm] = pol
        toy['cache'][key] = SpeakerTable.from_files(
            toy['files']['utt2spk'], min_len=dp.TOY_MIN_LEN, **spec)
    return toy['cache'][key]


def sid(table, name):
    return int(np.flatnonzero(table.spk_names == name)[0])


def arm_of(table, s):
    return int(table.spk_arm[s])


def draws(table, s, n=400, seed=0, counters=None):
    """n draws of speaker s under its arm's policy: [(rows, ptype name)]"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        rows, pt = po.draw_rows(table, s, arm_of(table, s), rng, counters)
        assert len(rows) == table.K and all(type(u) is int for u in rows)
        lo, hi = table.spk_ptr[s], table.spk_ptr[s + 1]
        assert all(lo <= u < hi for u in rows)
        out.append((rows, PTYPES[pt]))
    return out


def ks_dialect_spec(toy, p):
    return PolicySpec('ks_contrast', p_dialect=p,
                      subdialect_file=toy['files']['utt2subdialect'])


# ----------------------------------------------------------------------------


def test_ptypes_codes_are_fixed():
    assert PTYPES == (
        'xsession', 'xsession_segfix', '3ds_contrast', '3ds_xdev',
        '3ds_xseg', 'dialect', 'cn2_xgenre', 'same_segment', 'cn2_hardpair',
        '3ds_farfar', 'ks_xphase', 'ks_dialect', 'same_session')
    assert PT == {n: i for i, n in enumerate(PTYPES)}
    assert po.PTYPES is PTYPES


def test_cross_session_k_sessions(toy):
    t = toy['table']
    s = sid(t, 'id00003')  # 3 videos x 2 files
    c = po.DrawCounters()
    vids = collections.Counter()
    files = collections.Counter()
    for rows, pt in draws(t, s, 3000, counters=c):
        assert pt == 'xsession'
        v = t.vid_of[rows]
        assert v[0] != v[1]
        vids.update(v.tolist())
        files.update(rows)
    assert (c.same_video_fill, c.repeat_fill) == (0, 0)
    # uniform over the sessions, then over the files of the session
    assert all(abs(n / 6000 - 1 / 3) < 0.03 for n in vids.values())
    assert len(files) == 6 and all(abs(n / 6000 - 1 / 6) < 0.03
                                   for n in files.values())
    # an arm without a policy draws exactly the cross-session rows
    a, b = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(50):
        rows, pt = po.draw_rows(t, s, arm_of(t, s), a)
        assert t.policy[arm_of(t, s)] is None
        assert rows == po.draw_cross_session(t, s, b) and pt == PT['xsession']


def test_cross_session_fill_ups(toy):
    t3 = table_for(toy, K=3)
    # 2 videos x 2 files, K 3: one row per video and another distinct file
    s = sid(t3, 'vb2_id0001')
    c = po.DrawCounters()
    for rows, pt in draws(t3, s, 300, counters=c):
        assert pt == 'xsession' and len(set(rows)) == 3
        assert len(set(t3.vid_of[rows[:2]].tolist())) == 2
    assert (c.same_video_fill, c.repeat_fill) == (300, 0)
    # 2 files in one video (the third is too short), K 3: a row repeats
    s = sid(t3, 'id00900')
    assert t3.spk_ptr[s + 1] - t3.spk_ptr[s] == 2
    c = po.DrawCounters()
    for rows, _ in draws(t3, s, 300, counters=c):
        assert len(set(rows)) == 2
    assert (c.same_video_fill, c.repeat_fill) == (300, 300)
    # counters are optional
    po.draw_cross_session(t3, s, np.random.default_rng(0))


def test_pick_segment_first_exposes_segments_equally(toy):
    t = toy['table']
    idx = po.speaker_rows(t, sid(t, '3ds_3D_SPK_00000'))
    seg = t.seg_of[idx]
    seg_b = seg[seg != seg[0]][0]
    # 5 files of segment A, 2 of segment B
    cand = np.concatenate([idx[seg == seg[0]], idx[seg == seg_b][:2]])
    rng = np.random.default_rng(0)
    n = 4000
    got = collections.Counter(int(t.seg_of[po.pick(t, cand, rng, True)])
                              for _ in range(n))
    assert 0.45 < got[int(seg[0])] / n < 0.55
    got = collections.Counter(int(t.seg_of[po.pick(t, cand, rng, False)])
                              for _ in range(n))
    assert 0.65 < got[int(seg[0])] / n < 0.78
    # rows without a segment: always uniform over the files
    v = po.speaker_rows(t, sid(t, 'id00003'))
    got = collections.Counter(po.pick(t, v, rng, True) for _ in range(n))
    assert len(got) == len(v)


def test_pick_session_is_session_uniform(toy):
    t = toy['table']
    idx = po.speaker_rows(t, sid(t, 'id00002'))
    vid = t.vid_of[idx]
    # 2 files of one video, 1 file of another
    cand = np.concatenate([idx[vid == vid[0]], idx[vid != vid[0]][:1]])
    rng = np.random.default_rng(1)
    got = collections.Counter(int(t.vid_of[po.pick_session(t, cand, rng)])
                              for _ in range(4000))
    assert 0.45 < got[int(vid[0])] / 4000 < 0.55
    one = idx[vid == vid[0]]
    assert all(po.pick_session(t, one, rng) in one for _ in range(20))


def test_next_contrast_levels(toy):
    t = toy['table']
    idx = po.speaker_rows(t, sid(t, '3ds_3D_SPK_00001'))
    seg, dev, dcls = t.seg_of[idx], t.dev_of[idx], t.dcls_of[idx]
    rng = np.random.default_rng(2)
    u0 = int(idx[0])
    other_seg = seg != t.seg_of[u0]
    same_dev = dev == t.dev_of[u0]
    same_dcls = dcls == t.dcls_of[u0]
    assert (other_seg & ~same_dev & same_dcls).any()  # toy has such rows

    def level(cand, restrict=None):
        out = set()
        for _ in range(60):
            u, lev = po.next_contrast(t, [u0], cand, rng, True, restrict)
            assert u in cand
            out.add(lev)
        assert len(out) == 1
        return out.pop(), u

    lev, u = level(idx)  # everything differs
    assert lev == 0 and t.seg_of[u] != t.seg_of[u0]
    assert t.dev_of[u] != t.dev_of[u0] and t.dcls_of[u] != t.dcls_of[u0]
    lev, u = level(idx[other_seg & ~same_dev & same_dcls])
    assert lev == 1  # another device in the same distance class
    lev, u = level(idx[other_seg & same_dev])
    assert lev == 2  # another segment only
    lev, u = level(idx[~other_seg])
    assert lev == 3 and t.seg_of[u] == t.seg_of[u0]  # nothing else left
    # restrict limits the candidates of every level
    keep = other_seg & same_dev
    for _ in range(40):
        u, lev = po.next_contrast(t, [u0], idx, rng, True, keep)
        assert lev == 2 and t.dev_of[u] == t.dev_of[u0]
    u, lev = po.next_contrast(t, [u0], idx, rng, True, ~other_seg)
    assert lev == 3 and t.seg_of[u] == t.seg_of[u0]


def test_3ds_contrast_pairs(toy):
    t = toy['table']
    s = sid(t, '3ds_3D_SPK_00002')  # Standard only
    for rows, pt in draws(t, s, 500):
        assert pt == '3ds_contrast'  # toy: a full contrast always exists
        assert t.seg_of[rows[0]] != t.seg_of[rows[1]]
        assert t.dev_of[rows[0]] != t.dev_of[rows[1]]
        assert t.dcls_of[rows[0]] != t.dcls_of[rows[1]]
    # first rows cover every segment about equally
    first = collections.Counter(int(t.seg_of[r[0]]) for r, _ in
                                draws(t, s, 2000))
    assert len(first) == 4 and all(abs(n / 2000 - 0.25) < 0.04
                                   for n in first.values())


def test_3ds_dialect_pairs_follow_p_dialect(toy):
    def spec(p):
        return PolicySpec('contrast', p_dialect=p, segment_first=True)

    name = '3ds_3D_SPK_00101'  # 3 Standard and 3 dialect segments
    for p, lo, hi in ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.42, 0.58)):
        t = table_for(toy, **{'3dspeaker': spec(p)})
        s = sid(t, name)
        got = draws(t, s, 1500)
        rate = np.mean([pt == 'dialect' for _, pt in got])
        assert lo <= rate <= hi, (p, rate)
        for rows, pt in got:
            dl = t.dial_of[rows]
            assert t.seg_of[rows[0]] != t.seg_of[rows[1]]
            if pt == 'dialect':  # Standard first, then the dialect row
                assert dl[0] == 0 and dl[1] > 0
                assert t.dev_of[rows[0]] != t.dev_of[rows[1]]
            else:
                assert pt == '3ds_contrast'
    # a Standard-only speaker never draws the probability: its rows do not
    # depend on p_dialect
    rows = []
    for p in (0.0, 1.0):
        t = table_for(toy, **{'3dspeaker': spec(p)})
        rows.append(draws(t, sid(t, '3ds_3D_SPK_00002'), 100, seed=5))
    assert rows[0] == rows[1]


def test_3ds_dialect_k3_third_row_is_free(toy):
    t = table_for(toy, K=3, **{'3dspeaker': PolicySpec('contrast',
                                                       p_dialect=1.0)})
    s = sid(t, '3ds_3D_SPK_00100')
    kinds = collections.Counter()
    for rows, pt in draws(t, s, 400):
        dl = t.dial_of[rows]
        assert pt == 'dialect' and dl[0] == 0 and dl[1] > 0
        assert len(set(t.seg_of[rows].tolist())) == 3
        kinds[bool(dl[2] > 0)] += 1
    assert kinds[True] > 50 and kinds[False] > 50


def test_genre_contrast(toy):
    t = toy['table']
    s = sid(t, 'cn2_id10001')  # interview, singing, vlog; 2 sessions each
    pairs = collections.Counter()
    for rows, pt in draws(t, s, 3000):
        g = t.genre_of[rows]
        assert pt == 'cn2_xgenre' and g[0] != g[1] and (g >= 0).all()
        pairs[tuple(sorted(g.tolist()))] += 1
    assert len(pairs) == 3 and all(abs(n / 3000 - 1 / 3) < 0.04
                                   for n in pairs.values())
    # one genre: the cross-session draw
    s = sid(t, 'cn2_id10006')
    for rows, pt in draws(t, s, 200):
        assert pt == 'xsession' and t.vid_of[rows[0]] != t.vid_of[rows[1]]
    assert po.draw_genre_contrast(t, s, np.random.default_rng(0)) is None
    # the same rule serves the second genre-keyed arm
    s = sid(t, 'cn2_id00000')
    assert t.arm_names[arm_of(t, s)] == 'cn1'
    assert all(pt == 'cn2_xgenre' for _, pt in draws(t, s, 50))


def test_genre_contrast_more_rows_than_genres(toy):
    t3 = table_for(toy, K=3)
    s = sid(t3, 'cn2_id00001')  # speech, drama; 2 sessions each
    for rows, pt in draws(t3, s, 300):
        assert pt == 'cn2_xgenre'
        assert len(set(t3.genre_of[rows[:2]].tolist())) == 2
        assert len(set(t3.vid_of[rows].tolist())) == 3  # a new session


def test_ks_contrast_types(toy):
    t = toy['table']
    by = {}
    for i in range(7):
        s = sid(t, 'ks_%07d' % (1000000 + i))
        c = collections.Counter()
        for rows, pt in draws(t, s, 600):
            ph, dl = t.ks_phase[rows], t.ks_dial[rows]
            assert len(set(rows)) == 2
            if pt == 'ks_xphase':
                assert ph.tolist() == [1, 2]
            elif pt == 'ks_dialect':  # Standard first, then the sub-dialect
                assert ph.tolist() == [1, 1] and dl[0] == 0 and dl[1] > 0
            else:
                assert pt == 'xsession'
            c[pt] += 1
        by[i] = c
    for i in (0, 1):  # both phases, Standard and sub-dialect in phase 1
        assert set(by[i]) == {'ks_dialect', 'ks_xphase'}
        assert abs(by[i]['ks_dialect'] / 600 - 0.5) < 0.08
    for i in (2, 3):  # both phases, Standard only
        assert set(by[i]) == {'ks_xphase'}
    for i in (4, 5):  # phase 1 only, with a sub-dialect
        assert set(by[i]) == {'ks_dialect', 'xsession'}
        assert abs(by[i]['ks_dialect'] / 600 - 0.5) < 0.08
    assert set(by[6]) == {'xsession'}  # phase 1, Standard only


def test_ks_contrast_probability_and_fallthrough(toy):
    for p in (0.0, 1.0):
        t = table_for(toy, kespeech=ks_dialect_spec(toy, p))
        s = sid(t, 'ks_1000000')
        assert {pt for _, pt in draws(t, s, 200)} == {
            'ks_dialect' if p else 'ks_xphase'}
    t = toy['table']
    rng = np.random.default_rng(0)
    assert po.draw_ks_contrast(t, sid(t, 'ks_1000006'), 0.5, rng) is None
    # without the policy the arm draws across sessions (the phases)
    t0 = table_for(toy, kespeech=None)
    for name, n_phase in (('ks_1000002', 2), ('ks_1000006', 1)):
        s = sid(t0, name)
        for rows, pt in draws(t0, s, 100):
            assert pt == 'xsession'
            assert len(set(t0.vid_of[rows].tolist())) == n_phase


def test_alternate_groups_and_distinct_rows(toy):
    rng = np.random.default_rng(3)
    idx = np.arange(100, 108)
    a = idx < 102           # 2 rows
    b = idx >= 102          # 6 rows
    for _ in range(100):
        out = po.alternate(idx, (a, b), 5, rng)
        assert [bool(a[u - 100]) for u in out] == [True, False, True, False,
                                                   True]
        # the two rows of group a are used before one repeats
        assert {out[0], out[2]} == {100, 101} and out[4] in (100, 101)
        assert out[1] != out[3]
    t3 = table_for(toy, K=3)
    s = sid(t3, 'ks_1000002')
    for rows, pt in draws(t3, s, 100):
        assert pt == 'ks_xphase' and len(set(rows)) == 3
        assert t3.ks_phase[rows].tolist() == [1, 2, 1]


def test_forbid_same_segment_repairs_cross_session_rows(toy):
    xs = PolicySpec('xsession')
    t = table_for(toy, forbid=True, **{'3dspeaker': xs})
    s = sid(t, '3ds_3D_SPK_00003')
    kinds = collections.Counter()
    for rows, pt in draws(t, s, 1500):
        assert pt in ('xsession', 'xsession_segfix')
        assert t.seg_of[rows[0]] != t.seg_of[rows[1]]
        # the repair keeps the rows on different sessions (conditions)
        assert t.vid_of[rows[0]] != t.vid_of[rows[1]]
        kinds[pt] += 1
    # 4 segments on every device: 1 cross-session pair in 4 is a twin
    assert abs(kinds['xsession_segfix'] / 1500 - 0.25) < 0.04
    # without the flag the twins stay and are labelled
    t = table_for(toy, forbid=False, **{'3dspeaker': xs})
    s = sid(t, '3ds_3D_SPK_00003')
    kinds = collections.Counter()
    for rows, pt in draws(t, s, 1500):
        same = t.seg_of[rows[0]] == t.seg_of[rows[1]]
        assert pt == ('same_segment' if same else 'xsession')
        kinds[pt] += 1
    assert abs(kinds['same_segment'] / 1500 - 0.25) < 0.04
    # the single-segment speaker is drawable only without the flag
    one = sid(t, '3ds_3D_SPK_00200')
    assert t.spk_arm[one] >= 0
    assert {pt for _, pt in draws(t, one, 50)} == {'same_segment'}


def test_forbid_same_segment_k3(toy):
    t3 = table_for(toy, K=3, forbid=True,
                   **{'3dspeaker': PolicySpec('xsession')})
    s = sid(t3, '3ds_3D_SPK_00004')
    kinds = collections.Counter()
    for rows, pt in draws(t3, s, 600):
        assert len(set(t3.seg_of[rows].tolist())) == 3
        kinds[pt] += 1
    assert set(kinds) == {'xsession', 'xsession_segfix'}


def test_draws_are_reproducible_and_read_only(toy):
    t = toy['table']
    before = {n: getattr(t, n).copy() for n in
              ('rows', 'spk_of', 'vid_of', 'seg_of', 'dev_of', 'dcls_of',
               'dial_of', 'genre_of', 'ks_phase', 'ks_dial', 'spk_arm')}
    for name in ('id00001', 'vb2_id0002', 'cn2_id10002', 'cn2_id10005',
                 'cn2_id00000', '3ds_3D_SPK_00101', 'ks_1000001',
                 'mgd_14_0002', 'a1_S0001'):
        s = sid(t, name)
        assert draws(t, s, 30, seed=9) == draws(t, s, 30, seed=9)
        assert draws(t, s, 30, seed=9) != draws(t, s, 30, seed=10)
    for n, v in before.items():
        assert np.array_equal(getattr(t, n), v), n
