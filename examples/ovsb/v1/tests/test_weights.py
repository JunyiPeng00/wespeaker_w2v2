"""Speaker sampling weights: bridge, dialect mass, weight file, cell share."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_parity as dp  # noqa: E402

from ovsb.data import weights as wt  # noqa: E402
from ovsb.data.index import build_index  # noqa: E402
from ovsb.data.speakers import (DialectMass, PolicySpec,  # noqa: E402
                                SpeakerTable)


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    t = dp.make_toy(tmp_path_factory.mktemp('toy_weights'))
    build_index(t['files']['shard.list'], os.path.join(t['root'], 'index'),
                nj=1, root=t['root'])
    t['table'], t['weights'] = dp.toy_table(t)
    return t


def table_with(toy, policies):
    kw = dp.specs_from_episodic(toy['episodic'], toy['root'])
    kw['policies'] = policies
    return SpeakerTable.from_files(toy['files']['utt2spk'],
                                   min_len=dp.TOY_MIN_LEN, **kw)


def arm_of(table, name):
    return table.arm_speakers(table.arm_names.index(name))


def test_bridge_weights(toy):
    t, w = toy['table'], toy['weights']
    for arm, n_bridge in (('cn2', 4), ('cn1', 2)):
        spk = arm_of(t, arm)
        assert sorted(set(w.spk_w[spk].tolist())) == [1.0, 2.0]
        assert np.array_equal(w.spk_w[spk] == 2.0, t.spk_ngenre[spk] >= 2)
        assert w.info['%s_bridge_speakers' % arm] == n_bridge
    # w_bridge 1.0 is "no rule": the arm stays unweighted
    t2 = table_with(toy, {'cn2': PolicySpec('contrast', w_bridge=1.0)})
    w2 = wt.build_weights(t2)
    assert w2.spk_w is None and not w2.arm_weighted.any() and w2.info == {}


def test_dialect_mass(toy):
    t, w = toy['table'], toy['weights']
    spk = arm_of(t, '3dspeaker')
    d = t.spk_dial[spk]
    # 3 dialect speakers, 2 main dialects (12: one speaker, 9: two), boost 3
    # -> 4.5 of mass per dialect; the cap is 8 x median (1.0)
    assert sorted(set(d.tolist())) == [0, 9, 12]
    assert w.spk_w[spk][d == 12].tolist() == [4.5]
    assert w.spk_w[spk][d == 9].tolist() == [2.25, 2.25]
    assert (w.spk_w[spk][d == 0] == 1.0).all()
    assert (w.info['3dspeaker_dialect_speakers'], w.info['3dspeaker_dialects'],
            w.info['3dspeaker_weight_capped']) == (3, 2, 0)
    # the excluded single-segment speaker keeps weight 1 and is in no arm
    assert t.spk_arm[np.flatnonzero(t.spk_nseg == 1)[0]] == -1


def test_dialect_mass_cap(toy):
    pol = PolicySpec('contrast', dialect_mass=DialectMass(30.0, 8))
    t = table_with(toy, {'3dspeaker': pol})
    w = wt.build_weights(t)
    spk = arm_of(t, '3dspeaker')
    assert w.spk_w[spk].max() == 8.0
    assert w.info['3dspeaker_weight_capped'] == 3
    assert w.arm_weighted.tolist() == [n == '3dspeaker' for n in t.arm_names]


def test_dialect_mass_without_dialect_speakers(toy):
    """an arm with the rule but no dialect speaker stays unweighted"""
    w = np.ones(5)
    assert wt.dialect_mass_weights(w, np.arange(5), np.zeros(5, np.int64),
                                   DialectMass()) is None
    assert (w == 1.0).all()
    t = table_with(toy, {'vox2': PolicySpec(dialect_mass=DialectMass())})
    assert wt.build_weights(t).spk_w is None


def test_weight_file_and_share(toy):
    t, w = toy['table'], toy['weights']
    a = t.arm_names.index('mandarin_t1')
    spk = arm_of(t, 'mandarin_t1')
    by_name = dict(zip(t.spk_names[spk].tolist(), w.spk_w[spk].tolist()))
    assert by_name.pop('mgd_14_0000') == 3.0
    assert by_name.pop('mgd_14_0001') == 1.5
    assert set(by_name.values()) == {1.0} and len(by_name) == 6
    assert w.info['mandarin_t1_spk_weight_file_hits'] == 2
    assert w.info['mandarin_t1_spk_weight_share'] == 0.5
    assert np.isnan(np.delete(w.spk_share, a)).all() and w.spk_share[a] == 0.5
    assert t.spk_names[w.spk_listed].tolist() == ['mgd_14_0000',
                                                  'mgd_14_0001']
    assert w.arm_weighted.tolist() == [False, False, True, True, True, False,
                                       True]
    # without the share the listed speakers are only multiplied
    pol = PolicySpec(spk_weight_file=toy['files']['spk_weights.txt'])
    w2 = wt.build_weights(table_with(toy, {'mandarin_t1': pol}))
    assert w2.spk_share is None and w2.spk_listed is None
    assert np.array_equal(w2.spk_w[spk], w.spk_w[spk])


def test_share_weights():
    w = np.array([3.0, 1.5, 1.0, 1.0, 1.0, 2.0])
    listed = np.array([1, 1, 0, 0, 0, 0], bool)
    for share in (0.5, 0.2, 0.9):
        r = wt.share_weights(w, listed, share)
        assert abs(r[listed].sum() / r.sum() - share) < 1e-12
        assert np.array_equal(r[~listed], w[~listed])  # others untouched
        assert np.isclose(r[0] / r[1], 2.0)  # ratios inside the group kept
    assert w.tolist() == [3.0, 1.5, 1.0, 1.0, 1.0, 2.0]  # input not modified
    # a cell without listed (or without other) speakers is left as it is
    none = np.zeros(6, bool)
    assert wt.share_weights(w, none, 0.5) is w
    assert wt.share_weights(w, ~none, 0.5) is w


def test_rule_order_bridge_then_file(toy):
    """the file multiplies the weight the earlier rules gave"""
    path = os.path.join(toy['root'], 'cn_weights.txt')
    with open(path, 'w') as f:
        f.write('cn2_id10000 0.5\ncn2_id10007 3.0\n')
    pol = PolicySpec('contrast', w_bridge=2.0, spk_weight_file=path)
    t = table_with(toy, {'cn2': pol})
    w = wt.build_weights(t)
    by_name = dict(zip(t.spk_names.tolist(), w.spk_w.tolist()))
    assert by_name['cn2_id10000'] == 1.0  # bridge 2.0 x 0.5
    assert by_name['cn2_id10001'] == 2.0
    assert by_name['cn2_id10007'] == 3.0
    assert wt.read_weight_file(path) == {'cn2_id10000': 0.5,
                                         'cn2_id10007': 3.0}


def test_no_policy_no_weights(toy):
    w = wt.build_weights(table_with(toy, {}))
    assert w.spk_w is None and w.spk_share is None and w.spk_listed is None
    assert w.arm_weighted.tolist() == [False] * 7 and w.info == {}


# ----------------------------------------------------------------------------
# parity with the SOURCE sampler


@dp.needs_source
def test_parity_toy_weights_equal_source(toy):
    src, info = dp.toy_source(toy)
    assert dp.compare_weights(toy['table'], toy['weights'], src, info) == 4


@dp.needs_source
def test_parity_real_weights_equal_source():
    """spk_w, arm_weighted, spk_share, spk_listed and the per-cell share
    rescale equal the SOURCE sampler's on the real list; the counts are
    those of the reference run"""
    src, info = dp.real_source()
    table, weights, _timing = dp.real_port()
    assert dp.compare_weights(table, weights, src, info) == 4
    assert weights.info == {
        'cn2_bridge_speakers': 375, 'mandarin_t1_spk_weight_file_hits': 619,
        'mandarin_t1_spk_weight_share': 0.5,
        '3dspeaker_dialect_speakers': 1008, '3dspeaker_dialects': 13,
        '3dspeaker_weight_capped': 98, 'cn1_bridge_speakers': 567}
    assert list(weights.info) == list(info['cons'])[-len(weights.info):]
    assert weights.arm_weighted.tolist() == [
        n in ('cn2', 'mandarin_t1', '3dspeaker', 'cn1')
        for n in table.arm_names]
    assert int(weights.spk_listed.sum()) == 619
