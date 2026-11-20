"""pad_batch: the fixed field set, padding, integer codes, collation."""

import os
import re

import pytest
import torch
from torch.utils.data import default_collate

from conftest import RECIPE_DIR

from ovsb.config.schema import AUG_TYPES, CODECS
from ovsb.data import channel
from ovsb.data.collate import (BATCH_FIELDS, CHAIN_CODECS, REQUIRED_FIELDS,
                               pad_batch)
from ovsb.data.policies import PTYPES
from ovsb.data.sampler import SAMPLE_FIELDS


def sample(n=1000, **extra):
    """A row as the stages before the collate leave it."""
    row = dict(key='spk/utt', spk='spk', label=7, sample_rate=16000,
               wav=torch.arange(n, dtype=torch.float32).unsqueeze(0) + 1.0,
               arm=3, lang=1, vid=11, kpos=1, ptype=2, seg=-1, dev=-1,
               dcls=-1, dial=-1, genre=4)
    row.update(extra)
    return row


def one(**extra):
    return next(pad_batch([sample(**extra)]))


def test_field_set_and_order():
    assert BATCH_FIELDS == (
        'key', 'label', 'sample_rate', 'wav', 'wav_len', 'pair_role', 'arm',
        'kpos', 'ptype', 'aug_code', 'chain_code', 'chain_fail', 'zr_lsb',
        'zr_run', 'zr_dith', 'zr_bypass', 'eg_redraw', 'eg_fail')
    row = one(pad_to=1600, pair_role=2, aug_type='reverb', chain='opus',
              zr_lsb=0.25, zr_run=1, zr_dith=1, zr_silent=0, eg_redraw=3,
              eg_fail=1)
    assert tuple(row) == BATCH_FIELDS
    # conditions, language and session of the sampler are not batch fields
    assert not set(row) & {'spk', 'lang', 'vid', 'seg', 'dev', 'dcls', 'dial',
                           'genre', 'pad_to', 'aug_type', 'chain',
                           'zr_silent', 'label_b', 'lam'}
    assert (row['label'], row['arm'], row['kpos'], row['ptype']) == \
        (7, 3, 1, 2)
    assert (row['pair_role'], row['zr_lsb'], row['zr_run'], row['zr_dith'],
            row['eg_redraw'], row['eg_fail']) == (2, 0.25, 1, 1, 3, 1)


def test_padding_keeps_the_samples_and_reports_the_length():
    src = sample(n=1000, pad_to=1600)
    row = next(pad_batch([src]))
    assert row['wav'].shape == (1, 1600) and row['wav_len'] == 1000
    assert torch.equal(row['wav'][:, :1000], src['wav'])
    assert not row['wav'][:, 1000:].any()
    same = one(n=1600, pad_to=1600)
    assert same['wav'].shape == (1, 1600) and same['wav_len'] == 1600
    free = one(n=700)  # no pad_to: the row keeps its length
    assert free['wav'].shape == (1, 700) and free['wav_len'] == 700


def test_defaults_of_fields_no_stage_has_set():
    row = one()
    assert row['pair_role'] == 0
    assert (row['aug_code'], row['chain_code'], row['chain_fail']) == \
        (-1, -1, -1)
    assert (row['zr_lsb'], row['zr_run'], row['zr_dith'], row['zr_bypass'],
            row['eg_redraw'], row['eg_fail']) == (0.0, 0, 0, 0, -1, 0)
    assert isinstance(row['zr_lsb'], float)


def test_codes_follow_the_code_tables():
    # the index orders are part of the data format (logs, parity dumps)
    assert AUG_TYPES == ('clean', 'noise', 'reverb', 'reverb_noise')
    assert CHAIN_CODECS == CODECS == channel.CHAIN_CODECS == (
        'mp3', 'opus', 'vorbis', 'mulaw8k', 'g722')
    assert PTYPES[:8] == ('xsession', 'xsession_segfix', '3ds_contrast',
                          '3ds_xdev', '3ds_xseg', 'dialect', 'cn2_xgenre',
                          'same_segment')
    for code, name in enumerate(AUG_TYPES):
        assert one(aug_type=name)['aug_code'] == code
    for code, name in enumerate(CHAIN_CODECS):
        row = one(chain=name)
        assert (row['chain_code'], row['chain_fail']) == (code, -1)
        # a failed codec call is tagged chain_fail, never chain
        row = one(chain_fail=name)
        assert (row['chain_code'], row['chain_fail']) == (-1, code)
    assert one(aug_type='unknown')['aug_code'] == -1


def test_bypass_flag_is_forwarded():
    assert one(zr_bypass=1, zr_silent=1)['zr_bypass'] == 1


def test_bad_rows_are_refused():
    with pytest.raises(ValueError, match='padded to 800'):
        one(n=1000, pad_to=800)
    for name in REQUIRED_FIELDS:
        row = sample()
        row.pop(name)
        with pytest.raises(KeyError, match=name):
            next(pad_batch([row]))


def test_stage_is_one_to_one_and_lazy():
    seen = []

    def stream():
        for i in range(5):
            seen.append(i)
            yield sample(n=100 + i, pad_to=200, key='k%d' % i)

    out = pad_batch(stream())
    assert seen == []
    first = next(out)
    assert seen == [0] and first['key'] == 'k0'
    assert [r['wav_len'] for r in out] == [101, 102, 103, 104]


def test_default_collate_of_a_batch():
    rows = [sample(n=900 + 100 * i, pad_to=1600, key='k%d' % i, label=i,
                   pair_role=i % 3, zr_lsb=0.1 * i,
                   chain='g722' if i == 2 else None,
                   chain_fail='mp3' if i == 3 else None) for i in range(4)]
    batch = default_collate(list(pad_batch(rows)))
    assert tuple(batch) == BATCH_FIELDS
    assert batch['key'] == ['k0', 'k1', 'k2', 'k3']
    assert batch['wav'].shape == (4, 1, 1600)
    assert batch['wav'].dtype == torch.float32
    assert batch['wav_len'].tolist() == [900, 1000, 1100, 1200]
    assert batch['zr_lsb'].dtype == torch.float64
    for name in BATCH_FIELDS:
        if name not in ('key', 'wav', 'zr_lsb'):
            assert batch[name].dtype == torch.int64, name
            assert batch[name].shape == (4, ), name
    assert batch['chain_code'].tolist() == [-1, -1, 4, -1]
    assert batch['chain_fail'].tolist() == [-1, -1, -1, 0]
    assert batch['sample_rate'].tolist() == [16000] * 4


def test_required_fields_come_from_the_sampler_or_spk_to_id():
    assert set(REQUIRED_FIELDS) - {'label'} <= set(SAMPLE_FIELDS)
    assert set(REQUIRED_FIELDS) <= set(BATCH_FIELDS)


def test_every_field_the_training_step_reads_is_emitted():
    read = set()
    for name in ('step.py', 'arm_log.py', 'loop.py'):
        path = os.path.join(RECIPE_DIR, 'ovsb', 'engine', name)
        if not os.path.exists(path):
            continue
        with open(path) as fin:
            read |= set(re.findall(r"""batch\[['"](\w+)['"]\]""",
                                   fin.read()))
    assert {'label', 'wav', 'wav_len', 'pair_role', 'arm'} <= read
    assert read <= set(BATCH_FIELDS), read - set(BATCH_FIELDS)
