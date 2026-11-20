"""SpeakerTable: join, order, sessions, arms, conditions, share cells."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_parity as dp  # noqa: E402

from ovsb.data import speakers as sp  # noqa: E402
from ovsb.data.index import TarIndex, build_index  # noqa: E402

# counts logged by the reference run on the real training list
REAL_SPEAKERS = 152367
REAL_UTTS = 7824864
REAL_ARMS = ['vox2', 'vb2', 'cn2', 'mandarin_t1', '3dspeaker', 'kespeech',
             'tvx', 'cn1']
REAL_ARM_SPEAKERS = [5994, 104297, 1555, 2357, 9447, 24768, 3161, 787]
REAL_CELL = dict(arm='vb2', share=0.04, speakers=2422, arm_speakers=104297)


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    t = dp.make_toy(tmp_path_factory.mktemp('toy_speakers'))
    build_index(t['files']['shard.list'], os.path.join(t['root'], 'index'),
                nj=1, root=t['root'])
    t['table'], t['weights'] = dp.toy_table(t)
    return t


def spk_id(table, name):
    return int(np.flatnonzero(table.spk_names == name)[0])


def keys_of(table, s):
    lo, hi = table.spk_ptr[s], table.spk_ptr[s + 1]
    return [table.index.key(r) for r in table.rows[lo:hi]]


# ----------------------------------------------------------------------------
# rules on single ids / keys


def test_corpus_of():
    assert sp.corpus_of('id00012') == 'vox2'
    assert sp.corpus_of('vb2_id0001') == 'vb2'
    assert sp.corpus_of('cn2_id10001') == 'cn2'
    assert sp.corpus_of('plainname') == 'vox2'
    assert sp.corpus_of('cn2_id00001', {'cn2_id00001': 'cn1'}) == 'cn1'
    assert sp.corpus_of('cn2_id10001', {'cn2_id00001': 'cn1'}) == 'cn2'


def test_session_of():
    cases = {
        b'id00012/abcDEF/00001.wav': b'id00012/abcDEF',
        b'ks_1000001/phase1/1000001_0b1a': b'ks_1000001/phase1',
        b'cn2_id10005/recitation-18-014': b'cn2_id10005/recitation-18',
        b'cn2_id10005/vlog-02-001-comb2.wav': b'cn2_id10005/vlog-02',
        b'a4_S01/M_R001_0012_0345': b'a4_S01/M_R001',
        b'a4_S01/rec_12_345.wav': b'a4_S01/rec',
        b'x_1/single': b'x_1',
        b'x_1/a-b': b'x_1',
        b'x_1/rec_12_abc': b'x_1',
    }
    for key, want in cases.items():
        assert sp.session_of(key) == want, key


def test_assign_arms_first_match_and_languages():
    arms = [sp.ArmSpec('a', 0.5, 'vox2'), sp.ArmSpec('b', 0.3, ['vb2', 'x']),
            sp.ArmSpec('c', 0.2, ('x', 'y'))]
    assert arms[0].corpus == ('vox2', ) and arms[1].corpus == ('vb2', 'x')
    spks = ['id1', 'vb2_1', 'x_1', 'y_1', 'z_1', 'vb2_2']
    arm, langs = sp.assign_arms(spks, arms, {'vb2_1': 'zh', 'id1': 'fr'},
                                vox2_lang='en')
    assert arm.tolist() == [0, 1, 1, 2, -1, 1]
    # VoxCeleb2 speakers get vox2_lang whatever the file says
    assert langs == ['en', 'zh', 'unk', 'unk', 'unk', 'unk']
    assert sp.arm_matches(arms[2], 'y') and not sp.arm_matches(arms[2], 'z')


def test_spec_validation():
    with pytest.raises(ValueError, match='pair_policy'):
        sp.PolicySpec(pair_policy='random')
    with pytest.raises(ValueError, match='spk_weight_file'):
        sp.PolicySpec(spk_weight_share=0.5)
    with pytest.raises(ValueError, match=r'\(0, 1\)'):
        sp.PolicySpec(spk_weight_file='f', spk_weight_share=1.0)
    pol = sp.PolicySpec(dialect_mass={'boost': 2.0, 'cap_x_median': 4})
    assert pol.dialect_mass == sp.DialectMass(2.0, 4)


def test_read_label(tmp_path):
    p = tmp_path / 'utt2spk'
    p.write_text('k1 s1\nk0 s0\n')
    assert sp.read_label(str(p)) == [('k1', 's1'), ('k0', 's0')]
    p.write_text('k1 s1\nk0\n')
    with pytest.raises(ValueError, match=':2: expected 2 columns'):
        sp.read_label(str(p))


# ----------------------------------------------------------------------------
# the table on the synthetic corpus


def test_join_counts_and_structure(toy):
    t = toy['table']
    n_label = len(toy['label'])
    assert (t.n_missing, t.n_short) == (1, 1)
    assert t.n_utts == n_label - 2 == len(t.spk_of) == len(t.vid_of)
    spks = sorted({s for _, s in toy['utts']})
    assert t.spk_names.tolist() == spks and t.n_spk == len(spks)
    assert t.class_names.tolist() == spks and t.n_class == len(spks)
    assert t.spk_class.tolist() == list(range(len(spks)))
    assert (np.diff(t.spk_of) >= 0).all()
    assert t.spk_ptr[0] == 0 and t.spk_ptr[-1] == t.n_utts
    assert np.array_equal(np.diff(t.spk_ptr), np.bincount(t.spk_of))
    label = dict(toy['label'])
    for s in range(t.n_spk):
        ks = keys_of(t, s)
        assert all(label[k] == t.spk_names[s] for k in ks)
        v = t.vid_of[t.spk_ptr[s]:t.spk_ptr[s + 1]]
        assert (np.diff(v) >= 0).all()  # sessions are contiguous
        assert t.spk_nvid[s] == len(set(v.tolist())) == \
            len({sp.session_of(k.encode()) for k in ks})
    assert toy['short_key'] not in keys_of(t, spk_id(t, 'id00900'))
    assert t.spk_nvid[spk_id(t, 'id00900')] == 1
    # one-session speakers: id00900, zzz_0001, three phase-1-only KeSpeech
    assert int((t.spk_nvid < 2).sum()) == 5
    assert 'speakers with 1 video 5' in t.summary()


def test_order_inside_a_session_is_the_label_order(toy):
    """stable sort: utterances of one (speaker, session) keep the order of
    the label file, so the line order is part of the sampler's input"""
    t = toy['table']
    line = {k: i for i, (k, _) in enumerate(toy['label'])}
    n_checked = 0
    for s in range(t.n_spk):
        lo, hi = t.spk_ptr[s], t.spk_ptr[s + 1]
        ks = keys_of(t, s)
        v = t.vid_of[lo:hi]
        for u in set(v.tolist()):
            pos = [line[k] for k, x in zip(ks, v) if x == u]
            assert pos == sorted(pos)
            n_checked += len(pos) > 1
    assert n_checked > 50
    # the same lines sorted by key: same sets per speaker, another order
    kw = dp.specs_from_episodic(toy['episodic'], toy['root'])
    kw.pop('index_root')
    index = kw.pop('index')
    t2 = sp.SpeakerTable.build(
        sorted(toy['label']), TarIndex(index, root=toy['root']),
        sp.read_two_col(kw.pop('spk2corpus')),
        sp.read_two_col(kw.pop('spk2lang')), min_len=dp.TOY_MIN_LEN, **kw)
    assert np.array_equal(t2.spk_ptr, t.spk_ptr)
    assert np.array_equal(t2.vid_of, t.vid_of)
    assert np.array_equal(np.sort(t2.rows), np.sort(t.rows))
    assert not np.array_equal(t2.rows, t.rows)


def test_arms_languages_and_exclusion(toy):
    t = toy['table']
    assert t.arm_names == tuple(dp.TOY_ARMS)
    assert np.allclose(t.arm_weights, [0.2, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1])
    # the single-segment 3D-Speaker speaker cannot give K different
    # segments; 'zzz' matches no arm
    assert t.arm_speaker_counts().tolist() == [7, 6, 8, 8, 9, 7, 4]
    assert t.spk_arm[spk_id(t, '3ds_3D_SPK_00200')] == -1
    assert t.spk_arm[spk_id(t, 'zzz_0001')] == -1
    assert t.info['excluded_lt_K_segments'] == 1
    assert int((t.spk_arm < 0).sum()) == 2
    arm = {n: i for i, n in enumerate(t.arm_names)}
    assert t.spk_arm[spk_id(t, 'cn2_id00001')] == arm['cn1']  # from the file
    assert t.spk_arm[spk_id(t, 'cn2_id10001')] == arm['cn2']
    assert t.spk_arm[spk_id(t, 'a1_S0002')] == arm['mandarin_t1']
    assert t.arm_speakers(arm['cn1']).tolist() == \
        [spk_id(t, 'cn2_id0%04d' % i) for i in range(4)]
    assert t.lang_names.tolist() == ['en', 'ja', 'unk', 'zh']
    lang = dict(zip(t.spk_names.tolist(), t.lang_names[t.spk_lang].tolist()))
    assert lang['id00003'] == 'en' and lang['vb2_id0001'] == 'zh'
    assert lang['vb2_id0004'] == 'ja' and lang['vb2_id0005'] == 'unk'
    assert lang['cn2_id10001'] == 'unk'
    assert t.policy[arm['vox2']] is None and t.policy[arm['vb2']] is None
    assert t.policy[arm['kespeech']].pair_policy == 'ks_contrast'
    assert t.K == 2 and t.forbid_same_segment


def test_condition_tables_and_speaker_summaries(toy):
    t = toy['table']
    n3 = sum(s.startswith('3ds_') for _, s in toy['utts'])
    assert t.info['n_3ds_utts'] == n3 == int((t.seg_of >= 0).sum())
    assert t.info['n_3ds_segments'] == t.n_segments == n3 // len(dp.DEVS)
    assert t.info['n_cn2_genre_utts'] == int((t.genre_of >= 0).sum()) == 96
    for name, nseg, dial, std in (('3ds_3D_SPK_00000', 4, 0, True),
                                  ('3ds_3D_SPK_00100', 6, 12, True),
                                  ('3ds_3D_SPK_00102', 6, 9, True),
                                  ('3ds_3D_SPK_00200', 1, 0, True)):
        s = spk_id(t, name)
        assert (t.spk_nseg[s], t.spk_dial[s], t.spk_has_std[s]) == \
            (nseg, dial, std), name
    for name, n in (('cn2_id10000', 3), ('cn2_id10005', 1),
                    ('cn2_id00001', 2), ('cn2_id00003', 1), ('id00000', 0)):
        assert t.spk_ngenre[spk_id(t, name)] == n, name
    # conditions are aligned with the sorted positions
    for u in np.flatnonzero(t.seg_of >= 0)[::17]:
        key = t.index.key(t.rows[u])
        cond = key.split('/')[1]
        assert t.dev_of[u] == int(cond[6:8]) and t.dial_of[u] == int(cond[-2:])
    for u in np.flatnonzero(t.genre_of >= 0)[::7]:
        g = t.index.key(t.rows[u]).split('/')[1].split('-')[0]
        assert t.genre_names[t.genre_of[u]] == g.encode()
    s = spk_id(t, 'id00000')
    assert t.spk_nseg[s] == 0 and not t.spk_has_std[s]


def test_kespeech_tables(toy):
    t = toy['table']
    assert t.ks_dial_names[0] == b'Mandarin'
    assert set(t.ks_dial_names) == {b'Mandarin', b'Beijing', b'Ji-Lu'}
    assert t.info['ks'] == dict(
        utts=44, utts_without_subdialect=0,
        subdialects=[x.decode() for x in t.ks_dial_names], speakers=7,
        speakers_both_phases=4, speakers_std_and_dialect=4)
    ks = [spk_id(t, 'ks_%07d' % (1000000 + i)) for i in range(7)]
    assert t.spk_ks_both[ks].tolist() == [1, 1, 1, 1, 0, 0, 0]
    assert t.spk_ks_stddial[ks].tolist() == [1, 1, 0, 0, 1, 1, 0]
    sub = dp.toy_subdialect(toy['utts'])
    for u in np.flatnonzero(t.ks_phase >= 0):
        key = t.index.key(t.rows[u])
        assert t.ks_phase[u] == int(key.split('/')[1][5:])
        assert t.ks_dial_names[t.ks_dial[u]].decode() == sub[key]
    assert int((t.ks_phase >= 0).sum()) == 44
    # without a ks_contrast arm the arrays say "not applicable"
    kw = dp.specs_from_episodic(toy['episodic'], toy['root'])
    kw['policies'].pop('kespeech')
    t2 = sp.SpeakerTable.from_files(toy['files']['utt2spk'],
                                    min_len=dp.TOY_MIN_LEN, **kw)
    assert (t2.ks_phase == -1).all() and (t2.ks_dial == -1).all()
    assert not t2.spk_ks_both.any() and 'ks' not in t2.info
    assert np.array_equal(t2.rows, t.rows)


def test_share_cells(toy):
    t = toy['table']
    (cell, ) = t.share_cells
    assert (cell.name, cell.arm, cell.share) == ('vb2_zh', 1, 0.04)
    assert t.spk_names[cell.mask].tolist() == \
        ['vb2_id0000', 'vb2_id0001', 'vb2_id0002']
    assert t.cells_info == {'vb2_zh': dict(arm='vb2', share=0.04, speakers=3,
                                           arm_speakers=6)}
    kw = dp.specs_from_episodic(toy['episodic'], toy['root'])
    label = toy['files']['utt2spk']

    def build(*cells):
        kw['cells'] = list(cells)
        return sp.SpeakerTable.from_files(label, **kw)
    assert build().share_cells == ()
    with pytest.raises(ValueError, match='must lie in'):
        build(sp.CellSpec('c', 'vb2', 0.2, 'zh'))  # = the arm's weight
    with pytest.raises(ValueError, match='selects no speaker'):
        build(sp.CellSpec('c', 'vb2', 0.04, 'fr'))
    with pytest.raises(ValueError, match='is not one of'):
        build(sp.CellSpec('c', 'vb3', 0.04, 'zh'))
    with pytest.raises(ValueError, match='overlaps'):
        build(sp.CellSpec('c', 'vb2', 0.04, 'zh'),
              sp.CellSpec('d', 'vb2', 0.04, 'zh'))
    two = build(sp.CellSpec('c', 'vb2', 0.04, 'zh'),
                sp.CellSpec('d', 'vb2', 0.02, 'ja'))
    assert [c.name for c in two.share_cells] == ['c', 'd']
    assert int(two.share_cells[1].mask.sum()) == 2


def test_build_rejects_bad_arms(toy):
    kw = dp.specs_from_episodic(toy['episodic'], toy['root'])
    label = toy['files']['utt2spk']
    with pytest.raises(ValueError, match='unknown arms'):
        sp.SpeakerTable.from_files(label, **dict(
            kw, policies={'nope': sp.PolicySpec()}))
    with pytest.raises(ValueError, match='duplicate arm'):
        sp.SpeakerTable.from_files(label, **dict(
            kw, arms=kw['arms'] + [sp.ArmSpec('vox2', 0.1, 'q')]))
    with pytest.raises(ValueError, match='arm weights'):
        sp.SpeakerTable.from_files(label, **dict(
            kw, arms=[sp.ArmSpec('vox2', -0.1, 'vox2')], cells=[],
            policies={}))


def test_arm_order_sets_the_arm_codes(toy):
    """the arm index is the position in the given list; a config whose arms
    were re-ordered (e.g. keys sorted on dump) gives other codes"""
    kw = dp.specs_from_episodic(toy['episodic'], toy['root'])
    kw['arms'] = sorted(kw['arms'], key=lambda a: a.name)
    t2 = sp.SpeakerTable.from_files(toy['files']['utt2spk'],
                                    min_len=dp.TOY_MIN_LEN, **kw)
    t = toy['table']
    assert t2.arm_names != t.arm_names
    assert not np.array_equal(t2.spk_arm, t.spk_arm)
    name = np.array(t.arm_names + ('', ))
    name2 = np.array(t2.arm_names + ('', ))
    assert np.array_equal(name[t.spk_arm], name2[t2.spk_arm])


# ----------------------------------------------------------------------------
# parity with the SOURCE sampler


@dp.needs_source
def test_parity_toy_table_equals_source(toy):
    src, info = dp.toy_source(toy)
    done = dp.compare_table(toy['table'], src, info)
    assert set(dp.TABLE_ARRAYS) <= set(done)


@dp.needs_source
def test_parity_real_table_equals_source():
    """every static table equals the SOURCE sampler's on the real list"""
    src, info = dp.real_source()
    table, _weights, timing = dp.real_port()
    done = dp.compare_table(table, src, info)
    assert set(dp.TABLE_ARRAYS) <= set(done)
    print('[real list] exact-equal: %s' % ', '.join(done))
    print('[real list] source: %s, maxrss %.1f GB; port: %s' % (
        info['seconds'], info['maxrss_gb'], timing))


@dp.needs_source
def test_real_counts_match_the_reference_run():
    table, _weights, _timing = dp.real_port()
    assert table.n_spk == REAL_SPEAKERS == table.n_class
    assert table.n_utts == REAL_UTTS
    assert (table.n_missing, table.n_short) == (0, 0)
    assert list(table.arm_names) == REAL_ARMS
    assert table.arm_speaker_counts().tolist() == REAL_ARM_SPEAKERS
    assert int((table.spk_arm < 0).sum()) == 1
    assert int((table.spk_nvid < 2).sum()) == 16437
    assert table.cells_info == {'vb2_zh': REAL_CELL}
    assert int(table.share_cells[0].mask.sum()) == 2422
    assert table.info == {
        'n_3ds_utts': 472944, 'n_3ds_segments': 102303,
        'n_cn2_genre_utts': 464280,
        'ks': {'utts': 886604, 'utts_without_subdialect': 0,
               'subdialects': ['Mandarin', 'Northeastern', 'Jiang-Huai',
                               'Southwestern', 'Jiao-Liao', 'Beijing',
                               'Zhongyuan', 'Ji-Lu', 'Lan-Yin'],
               'speakers': 24768, 'speakers_both_phases': 15531,
               'speakers_std_and_dialect': 10953},
        'excluded_lt_K_segments': 1}
    # the source subprocess logs the same construction as the reference run
    _src, info = dp.real_source()
    line = [x for x in info['log'] if x.startswith('episodic sampler:')][0]
    assert '7824864 utts / 152367 spk (0 keys not in the index' in line
    assert "arms %s spk %s unassigned 1" % (REAL_ARMS,
                                            REAL_ARM_SPEAKERS) in line
