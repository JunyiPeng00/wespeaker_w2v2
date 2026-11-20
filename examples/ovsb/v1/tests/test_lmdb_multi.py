"""Weighted multi-source LMDB reader: draws, key lookup, source parity."""

import collections
import random

import pytest

import proc_parity as pp
from ovsb.config import schema
from ovsb.data.lmdb_multi import (MultiLmdbData, open_noise_source,
                                  open_reverb_source)
from wespeaker.dataset.lmdb_data import LmdbData
from wespeaker.dataset.processor import _rir_same_room

needs_source = pytest.mark.skipif(
    not pp.have_source(), reason='set OVSB_SOURCE_TREE for parity tests')


def _pair():
    a = pp.FakeSource([('a/r1/x', b'ax'), ('a/r1/y', b'ay'), ('a/r2/z', b'az')])
    b = pp.FakeSource([('b/r9/u', b'bu'), ('a/r1/x', b'shadowed')])
    return a, b


def test_weights_are_normalised_and_drawn():
    a, b = _pair()
    multi = MultiLmdbData([a, b], [3.0, 1.0])
    assert multi.weights == [0.75, 0.25] and multi.cum == [0.75, 1.0]
    random.seed(0)
    picks = collections.Counter(multi.pick_source() for _ in range(20000))
    assert abs(picks[0] / 20000 - 0.75) < 0.01
    random.seed(1)
    owners = collections.Counter(multi.random_one()[1][:1]
                                 for _ in range(20000))
    assert abs(owners[b'a'] / 20000 - 0.75) < 0.02
    assert MultiLmdbData([a, b], [0.0, 2.0]).cum == [0.0, 1.0]


def test_draw_is_one_uniform_then_the_source_draw():
    a, b = _pair()
    multi = MultiLmdbData([a, b], [0.5, 0.5])
    random.seed(5)
    got = [multi.random_one() for _ in range(50)]
    random.seed(5)
    want = []
    for _ in range(50):
        src = a if random.random() < 0.5 else b
        want.append(src.random_one())
    assert got == want


def test_keys_are_the_union_and_lookups_reach_the_owner():
    a, b = _pair()
    multi = MultiLmdbData([a, b], [0.5, 0.5])
    assert multi.keys == ['a/r1/x', 'a/r1/y', 'a/r2/z', 'b/r9/u']
    with multi.db.begin(write=False) as txn:
        assert txn.get(b'a/r1/x') == b'ax'  # the first source owns it
        assert txn.get(b'b/r9/u') == b'bu'
        assert txn.get('a/r2/z') == b'az'
        assert txn.get(b'no/such/key') is None
    assert multi.num_excluded == 0


def test_same_room_lookup_works_across_sources():
    a, b = _pair()
    multi = MultiLmdbData([a, b], [0.5, 0.5])
    random.seed(2)
    seen = {_rir_same_room(multi, 'a/r1/x') for _ in range(40)}
    assert seen == {('a/r1/x', b'ax'), ('a/r1/y', b'ay')}
    assert _rir_same_room(multi, 'b/r9/other') == ('b/r9/u', b'bu')
    # an unknown room falls back to a plain draw
    assert _rir_same_room(multi, 'nowhere/x')[0] in multi.keys


def test_invalid_arguments():
    a, b = _pair()
    for sources, weights in (([a, b], [1.0]), ([], []), ([a], [0.0]),
                             ([a, b], [1.0, -1.0])):
        with pytest.raises(AssertionError):
            MultiLmdbData(sources, weights)


def test_sources_of_an_augment_config(tmp_path):
    lmdbs = pp.make_aug_lmdbs(str(tmp_path))
    cfg = schema.build_section(schema.AugmentCfg, pp.augment_cfg(lmdbs), 'a')
    reverb = open_reverb_source(cfg)
    assert isinstance(reverb, MultiLmdbData) and len(reverb.sources) == 2
    assert reverb.cum == [0.5, 1.0]
    assert len(reverb.keys) == 8 + 6  # the shared key is counted once
    assert reverb.owner[reverb.sources[0].keys[1]] is reverb.sources[0]
    noise = open_noise_source(cfg)
    assert isinstance(noise, LmdbData)
    assert noise.num_excluded == 2 and len(noise.keys) == 10
    random.seed(3)
    for _ in range(100):
        key, value = reverb.random_one()
        with reverb.db.begin(write=False) as txn:
            if key != reverb.sources[0].keys[1]:
                assert txn.get(key.encode()) == value
    excluded = set(open(lmdbs['noise_exclude_list']).read().split())
    assert not excluded & {noise.random_one()[0] for _ in range(300)}


# ------------------------------------------------------------------ parity
@needs_source
def test_parity_with_source(tmp_path):
    lmdbs = pp.make_aug_lmdbs(str(tmp_path / 'lmdb'))
    # an LMDB cannot be opened twice in a process: one copy per layout
    other = pp.make_aug_lmdbs(str(tmp_path / 'lmdb2'), seed=1)
    specs = [
        dict(kind='multi_lmdb', seed=3, n=400,
             augment=pp.augment_cfg(lmdbs)),
        dict(kind='multi_lmdb', seed=4, n=400,
             augment=pp.augment_cfg(
                 other, reverb_sources=[other['reverb_real'], 'simulated'],
                 reverb_weights=[0.2, 0.7], noise_exclude_list=None)),
    ]
    source, new = pp.run_both(specs, str(tmp_path))
    for a, b in zip(source, new):
        assert len(b['draws']) == 800 and len(b['lookups']) == 15
        assert a == b
    assert new[0]['num_excluded'] == 2 and new[1]['num_excluded'] == 0
