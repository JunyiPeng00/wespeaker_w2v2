"""Training pipeline on the toy corpus: stage order, epochs, batch rows."""

import copy
import os
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_parity as dp  # noqa: E402
import pipeline_helpers as ph  # noqa: E402

from ovsb.data import channel, pipeline  # noqa: E402
from ovsb.data.collate import BATCH_FIELDS  # noqa: E402
from ovsb.data.lmdb_multi import (open_noise_source,  # noqa: E402
                                  open_reverb_source)
from ovsb.data.sampler import EpisodicSampler  # noqa: E402
from wespeaker.dataset.dataset import EpochProcessor  # noqa: E402

BATCH = ph.P * ph.K


@pytest.fixture(scope='module')
def recipe(tmp_path_factory):
    r = ph.make_toy_recipe(tmp_path_factory.mktemp('toy_pipeline'))
    r.table = pipeline.table_from_config(r.cfg)
    # an LMDB environment is opened once per process
    r.reverb = open_reverb_source(r.cfg.data.augment)
    r.noise = open_noise_source(r.cfg.data.augment)
    return r


def build(recipe, cfg=None):
    cfg = cfg or recipe.cfg
    sampler = pipeline.sampler_from_config(cfg, recipe.table)
    return pipeline.build_train_dataset(cfg, sampler, recipe.reverb,
                                        recipe.noise)


def stages(dataset):
    out, node = [], dataset.source
    while not isinstance(node, EpisodicSampler):
        out.append(node)
        node = node.source
    return out[::-1]


def batches(dataset, n, seed=11):
    """``n`` collated batches of BATCH consecutive rows, in this process."""
    ph.seed_all(seed)
    rows = ph.take(dataset, n * BATCH)
    return [default_collate(rows[i:i + BATCH])
            for i in range(0, len(rows), BATCH)]


# ------------------------------------------------------------ config glue
def test_table_from_config_is_the_table_of_the_typed_specs(recipe):
    table = recipe.table
    want, _ = dp.toy_table(recipe.toy)
    for name in dp.TABLE_ARRAYS:
        np.testing.assert_array_equal(getattr(table, name),
                                      getattr(want, name), err_msg=name)
    assert table.arm_names == tuple(recipe.cfg.data.sampler.arm_order)
    assert [c.name for c in table.share_cells] == ['vb2_zh']
    assert table.K == ph.K and table.forbid_same_segment
    assert [p.pair_policy if p else None for p in table.policy] == \
        [p.pair_policy if p else None for p in want.policy]
    assert table.n_short == 1 and table.n_missing == 1


def test_filter_and_min_len_follow_the_frame_grid(recipe):
    cfg = ph.variant(recipe, **{'data.filter': {'min_sec': 1.0,
                                                'max_sec': 8.0}})
    assert pipeline.filter_frames(cfg.data.filter) == (50, 400)
    assert pipeline.table_min_len(cfg) == 16000
    assert pipeline.filter_frames(recipe.cfg.data.filter) == (3, 6)
    assert pipeline.table_min_len(recipe.cfg) == 960
    assert pipeline.arm_index(recipe.cfg) == {
        n: i for i, n in enumerate(dp.TOY_ARMS)}


def test_sampler_from_config(recipe):
    cfg = recipe.cfg
    sampler = pipeline.sampler_from_config(cfg, recipe.table)
    assert (sampler.P, sampler.K, sampler.seed) == (ph.P, ph.K, ph.SEED)
    assert sampler.sample_num_per_epoch == \
        cfg.data.sampler.sample_num_per_epoch
    assert sampler.log_every == cfg.data.sampler.log_every
    assert sampler.weights.arm_weighted.any()
    bad = copy.deepcopy(cfg)  # a config that skipped the schema check
    bad.data.loader.batch_size = BATCH + 2
    with pytest.raises(ValueError, match='P x K'):
        pipeline.sampler_from_config(bad, recipe.table)


# ------------------------------------------------------------- stage chain
def test_stage_order(recipe):
    dataset = build(recipe)
    chain = stages(dataset)
    assert [s.f.__name__ for s in chain] == [
        'filter', 'spk_to_id', 'resample', 'random_chunk_var',
        'zero_run_guard', 'draw_aug_type', 'add_reverb_noise', 'pad_batch']
    assert [isinstance(s, EpochProcessor) for s in chain] == [
        False, False, False, True, True, False, False, False]
    assert pipeline.find_sampler(dataset) is dataset.sampler
    by_name = {s.f.__name__: s for s in chain}
    assert by_name['filter'].kw == dict(
        frame_shift=20, data_type='shard', min_num_frames=3,
        max_num_frames=6)
    crop_kw = by_name['random_chunk_var'].kw
    assert crop_kw['batch_size'] == BATCH and crop_kw['seed'] == ph.SEED
    assert crop_kw['max_lens'] == [800, 1200]
    assert sorted(crop_kw['arm_caps']) == sorted(
        pipeline.arm_index(recipe.cfg)[n] for n in ('cn2', 'cn1'))
    dither_kw = by_name['zero_run_guard'].kw
    assert dither_kw['seed'] == ph.SEED and 'rank' not in dither_kw
    assert dither_kw['dither'] and dither_kw['stats']
    assert by_name['add_reverb_noise'].args[:2] == (recipe.reverb,
                                                    recipe.noise)
    assert by_name['add_reverb_noise'].kw == {'reverberant_noise': True}
    # no stage may receive a keyword called 'source' (Processor owns it)
    assert not any('source' in s.kw for s in chain)
    spk2id = by_name['spk_to_id'].args[0]
    assert list(spk2id) == [str(n) for n in recipe.table.class_names]
    assert list(spk2id.values()) == list(range(recipe.table.n_class))


def test_channel_stage_only_when_its_probability_is_positive(recipe,
                                                             monkeypatch):
    probes = []
    monkeypatch.setattr(channel, '_ffmpeg_ok',
                        lambda: probes.append(1) or True)
    assert 'add_channel_asym' not in [s.f.__name__
                                      for s in stages(build(recipe))]
    assert probes == []  # chain off: FFmpeg is not even probed
    cfg = ph.variant(recipe, **{'data.channel.prob': 1.0})
    names = [s.f.__name__ for s in stages(build(recipe, cfg))]
    assert names[-3:] == ['add_reverb_noise', 'add_channel_asym', 'pad_batch']
    assert probes == [1]
    kw = stages(build(recipe, cfg))[-2].kw
    assert kw['arms'] == [pipeline.arm_index(cfg)['mandarin_t1']]
    assert kw['kpos'] == (1, ) and kw['prob'] == 1.0
    monkeypatch.setattr(channel, '_ffmpeg_ok', lambda: False)
    with pytest.raises(RuntimeError, match='FFmpeg'):
        build(recipe, cfg)


def test_sources_and_arm_order_are_checked(recipe):
    sampler = pipeline.sampler_from_config(recipe.cfg, recipe.table)
    with pytest.raises(ValueError, match='reverb and a noise source'):
        pipeline.build_train_dataset(recipe.cfg, sampler, None, recipe.noise)
    order = list(recipe.cfg.data.sampler.arm_order)
    cfg = ph.variant(recipe, **{'data.sampler.arm_order': order[::-1]})
    with pytest.raises(ValueError, match='arm_order'):
        pipeline.build_train_dataset(cfg, sampler, recipe.reverb,
                                     recipe.noise)


# ------------------------------------------------------------------ epochs
def test_set_epoch_reaches_sampler_and_epoch_stages(recipe):
    dataset = build(recipe)
    crop, dither = [s for s in stages(dataset)
                    if isinstance(s, EpochProcessor)]
    dataset.set_epoch(4)
    assert (dataset.epoch, dataset.stream) == (4, 0)
    assert (dataset.sampler.epoch, dataset.sampler.stream) == (4, 0)
    assert crop.kw['epoch'] == dither.kw['epoch'] == 4
    # stream 0: the keyword is absent, as in a run without restarts
    assert 'stream' not in crop.kw and 'stream' not in dither.kw
    dataset.set_epoch(4, stream=2)
    assert (dataset.sampler.epoch, dataset.sampler.stream) == (4, 2)
    assert crop.kw['stream'] == dither.kw['stream'] == 2
    dataset.set_epoch(5)
    assert (dataset.sampler.epoch, dataset.sampler.stream) == (5, 0)
    assert crop.kw['epoch'] == 5
    assert 'stream' not in crop.kw and 'stream' not in dither.kw


# -------------------------------------------------------------------- rows
def test_rows_and_batches(recipe):
    dataset = build(recipe)
    dataset.set_epoch(1)
    got = batches(dataset, 8)
    table = recipe.table
    spk_of_key = dict(recipe.toy['utts'])
    classes = [str(n) for n in table.class_names]
    lens = set(c[0] for c in recipe.cfg.data.crop.max_secs)
    pair_batches = 0
    for batch in got:
        assert tuple(batch) == BATCH_FIELDS
        assert batch['wav'].shape[:2] == (BATCH, 1)
        assert batch['wav'].dtype == torch.float32
        width = batch['wav'].shape[2]
        assert (batch['wav_len'] <= width).all()
        assert (batch['wav_len'] >= 0.03 * 16000).all()
        for i, n in enumerate(batch['wav_len'].tolist()):
            assert not batch['wav'][i, 0, n:].any()
            assert batch['wav'][i, 0, :n].abs().max() > 0
        assert batch['label'].tolist() == [
            classes.index(spk_of_key[k]) for k in batch['key']]
        assert (batch['sample_rate'] == 16000).all()
        roles = batch['pair_role'].tolist()
        if any(roles):
            pair_batches += 1
            assert roles == [1, 2] * (BATCH // 2)
            assert width == 1440  # pair_anchor_sec
            # anchors stay clean
            assert (batch['aug_code'][0::2] == 0).all()
        else:
            assert width / 16000 in lens
            # one episode: K consecutive rows per speaker
            assert batch['kpos'].tolist() == [0, 1] * ph.P
            assert batch['label'][0::2].tolist() == \
                batch['label'][1::2].tolist()
        assert set(batch['aug_code'].tolist()) <= {0, 1, 2, 3}
        assert (batch['chain_code'] == -1).all()
        assert (batch['chain_fail'] == -1).all()
    assert 0 < pair_batches < len(got)
    # the keys are the sampler's stream; a pair batch uses half an episode
    stream = ph.cell_keys(dataset.sampler, 1, 1, 0, 1, len(got))
    used = ph.utterance_keys(got)
    assert used == stream[:len(used)]
    assert len(used) == BATCH * len(got) - (BATCH // 2) * pair_batches
    codes = torch.cat([b['aug_code'] for b in got])
    assert len(set(codes.tolist())) == 4  # every augmentation type occurs


def test_long_utterances_are_cut_by_the_filter(recipe):
    # filter.max_sec is 0.12 s: no crop source is longer than 1920 samples,
    # so with a cap above it rows are at most 1920 long
    cfg = ph.variant(recipe, **{
        'data.crop.max_secs': [[0.2, 1.0]], 'data.crop.full_prob': 1.0,
        'data.crop.pair_prob': 0.0, 'data.crop.arm_caps': {}})
    dataset = build(recipe, cfg)
    dataset.set_epoch(1)
    got = batches(dataset, 3)
    lens = torch.cat([b['wav_len'] for b in got])
    assert lens.max() == 1920 and lens.min() >= 1200
    assert all(b['wav'].shape[2] == 3200 for b in got)


def test_same_epoch_and_stream_repeat_and_others_differ(recipe):
    def run(epoch, stream=0, seed=11):
        dataset = build(recipe)
        dataset.set_epoch(epoch, stream)
        return [ph.row_digest(b) for b in batches(dataset, 4, seed)]

    first = run(1)
    assert run(1) == first  # same seeds, same (epoch, stream)
    assert run(1, seed=12) != first  # the process seed matters (crops, aug)
    second = run(2)
    assert not set(second) & set(first)
    restart = run(1, stream=1)
    assert not set(restart) & set(first)
    assert run(1, stream=1) == restart
    # one dataset object, epochs one after the other
    dataset = build(recipe)
    seq = []
    for epoch in (1, 2, 1):
        dataset.set_epoch(epoch)
        seq.append([ph.row_digest(b) for b in batches(dataset, 4)])
    assert seq[0] == first and seq[1] == second and seq[2] == first


def test_epoch_changes_the_sampler_stream_and_the_crop_caps(recipe):
    dataset = build(recipe)
    keys, widths = [], []
    for epoch in (1, 2):
        dataset.set_epoch(epoch)
        got = batches(dataset, 6)
        keys.append(ph.utterance_keys(got))
        widths.append([b['wav'].shape[2] for b in got])
    assert keys[0] != keys[1]
    assert widths[0] != widths[1]
