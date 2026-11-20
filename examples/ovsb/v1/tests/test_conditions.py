"""Key parsers: 3D-Speaker conditions, CN-Celeb genre, KeSpeech phase."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_parity as dp  # noqa: E402

from ovsb.data import conditions as co  # noqa: E402

COND = 'Device01_Distance04_Dialect00'
KEY_3DS = '3ds_3D_SPK_00001/%s/3D_SPK_00001_001_%s' % (COND, COND)


def test_parse_3ds():
    c = co.parse_3ds(KEY_3DS)
    assert c == (b'3ds_3D_SPK_00001/3D_SPK_00001_001', 1, 2, 0)
    assert (c.segment, c.device, c.distance_class, c.dialect) == c
    assert co.parse_3ds(KEY_3DS.encode()) == c
    assert co.parse_3ds(KEY_3DS + '.wav') == c
    c = co.parse_3ds('3ds_x/Device03_Distance00_Dialect12/'
                     'x_007_Device03_Distance00_Dialect12')
    assert c == (b'3ds_x/x_007', 3, 0, 12)
    # an utterance name without the condition suffix is its own segment
    assert co.parse_3ds('3ds_x/%s/free_name' % COND).segment == \
        b'3ds_x/free_name'
    for key in ('id00001/abc/00001.wav', '3ds_x/%s' % COND,
                '3ds_x/NoCondition/utt', 'cn2_id10005/recitation-18-014'):
        assert co.parse_3ds(key) is None, key


def test_distance_class():
    codes = (0, 2, 3, 8, 9, 10, 4, 11, 12, 13, 1, 5, 6, 7)
    assert [co.distance_class(c) for c in codes] == \
        [0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
    assert co.distance_class(99) == 0  # unknown code: close, as close-talk


def test_parse_genre():
    assert co.parse_genre('cn2_id10005/recitation-18-014') == b'recitation'
    assert co.parse_genre(b'cn2_id00012/live_broadcast-01-002-comb3') == \
        b'live_broadcast'
    assert co.parse_genre('cn2_id10005/two-fields') is None
    assert co.parse_genre('id1/a/b') is None
    assert co.parse_genre('cn1_id1/vlog-01-001') is None  # prefix rule


def test_parse_ks_phase():
    assert co.parse_ks_phase('ks_1000001/phase1/1000001_0b1a33a3') == 1
    assert co.parse_ks_phase(b'ks_1000001/phase2/x') == 2
    assert co.parse_ks_phase('ks_1000001/phaseX/x') is None
    assert co.parse_ks_phase('ks_1000001/phase1') is None
    assert co.parse_ks_phase('id1/phase1/x') is None


def test_condition_arrays():
    keys = [KEY_3DS, 'id00001/v/1.wav',
            '3ds_a/Device06_Distance10_Dialect09/a_002_'
            'Device06_Distance10_Dialect09',
            'cn2_id1/vlog-01-001', 'cn2_id1/singing-01-001',
            '3ds_3D_SPK_00001/Device03_Distance00_Dialect00/3D_SPK_00001_001'
            '_Device03_Distance00_Dialect00',
            'cn2_id2/vlog-02-001', 'cn2_id2/bad', '3ds_bad/x/y']
    c = co.condition_arrays(np.array(keys, dtype='S'))
    # ids in order of first appearance; a segment is shared by its devices
    assert c.seg.tolist() == [0, -1, 1, -1, -1, 0, -1, -1, -1]
    assert c.seg_names == [b'3ds_3D_SPK_00001/3D_SPK_00001_001',
                           b'3ds_a/a_002']
    assert c.dev.tolist() == [1, -1, 6, -1, -1, 3, -1, -1, -1]
    assert c.dcls.tolist() == [2, -1, 1, -1, -1, 0, -1, -1, -1]
    assert c.dial.tolist() == [0, -1, 9, -1, -1, 0, -1, -1, -1]
    assert c.genre.tolist() == [-1, -1, -1, 0, 1, -1, 0, -1, -1]
    assert c.genre_names == [b'vlog', b'singing']
    assert all(a.dtype == np.int64 for a in (c.seg, c.dev, c.dcls, c.dial,
                                             c.genre))
    empty = co.condition_arrays(np.array([], dtype='S'))
    assert len(empty.seg) == 0 and empty.seg_names == []


def test_ks_arrays(tmp_path):
    keys = ['ks_1/phase1/a', 'ks_1/phase1/b.wav', 'ks_1/phase2/c',
            'id1/v/1', 'ks_2/phase1/d', 'ks_2/other/e', 'ks_2/phase1/f']
    p1, p2 = tmp_path / 't1', tmp_path / 't2'
    p1.write_text('ks_1/phase1/a Mandarin\nks_1/phase1/b Ji-Lu\n'
                  'ks_1/phase2/c Beijing\nks_2/other/e Ji-Lu\nshort\n')
    p2.write_text('ks_2/phase1/d Beijing\n')
    table = co.read_subdialect([str(p1), str(p2)])
    assert table[b'ks_1/phase1/b'] == b'Ji-Lu' and len(table) == 5
    k = co.ks_arrays(np.array(keys, dtype='S'), table)
    assert k.phase.tolist() == [1, 1, 2, -1, 1, -1, 1]
    # Standard is 0, the others in order of first appearance; '.wav' keys
    # are also looked up without the suffix; no entry -> -1
    assert k.dial.tolist() == [0, 1, 2, -1, 2, 1, -1]
    assert k.dial_names == [b'Mandarin', b'Ji-Lu', b'Beijing']
    assert (k.n_utts, k.n_without_subdialect) == (6, 1)
    none = co.ks_arrays(np.array(keys, dtype='S'), {})
    assert none.dial.tolist() == [-1] * 7 and none.n_without_subdialect == 6
    assert none.phase.tolist() == k.phase.tolist()


def test_toy_corpus_conditions():
    utts = dp.toy_utts()
    keys = np.array([k for k, _ in utts], dtype='S')
    c = co.condition_arrays(keys)
    is3 = np.array([s.startswith('3ds_') for _, s in utts])
    iscn = np.array([s.startswith('cn2_') for _, s in utts])
    assert ((c.seg >= 0) == is3).all() and ((c.genre >= 0) == iscn).all()
    # every segment was recorded by the five devices of the toy corpus
    assert (np.bincount(c.seg[is3]) == len(dp.DEVS)).all()
    assert sorted(set(c.dev[is3].tolist())) == [1, 3, 4, 6]
    assert sorted(set(c.dcls[is3].tolist())) == [0, 1, 2]
    assert sorted(set(c.dial[is3].tolist())) == [0, 7, 9, 12]
    assert sorted(c.genre_names) == [b'drama', b'interview', b'singing',
                                     b'speech', b'vlog']
