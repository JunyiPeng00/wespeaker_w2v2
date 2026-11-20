"""Head record: sidecar round trip, fail-closed check, existing format."""
import json
import os

import pytest

from ovsb.model.head_record import (HeadMismatchError, check_run_head,
                                    head_sidecar_path, speaker_list_digest,
                                    verify_head, write_head_sidecar)

SPKS = ['id%03d' % i for i in range(7)]
ROWS = 21


def read(path):
    with open(path) as fin:
        return json.load(fin)


def test_digest_is_order_and_repeat_invariant():
    digest = speaker_list_digest(SPKS)
    assert digest == speaker_list_digest(reversed(SPKS + SPKS[:3]))
    assert digest != speaker_list_digest(SPKS[:-1])
    assert digest != speaker_list_digest(SPKS[:-1] + ['id006 '])
    assert len(digest) == 64


def test_sidecar_round_trip(tmp_path):
    ckpt = str(tmp_path / 'init.pt')
    record = write_head_sidecar(ckpt, SPKS[::-1] + SPKS, ROWS, source='a.pt',
                                extra={'tool': 'unit test'})
    assert read(head_sidecar_path(ckpt)) == record
    assert record == {'spk_sha256': speaker_list_digest(SPKS),
                      'num_classes': 7, 'rows': 21, 'rows_per_class': 3,
                      'source': 'a.pt', 'tool': 'unit test'}
    verify_head(record, SPKS, ROWS, 'record')
    assert not os.path.exists(head_sidecar_path(ckpt) + '.tmp')


def test_sidecar_rows_must_be_a_multiple(tmp_path):
    with pytest.raises(ValueError, match='not a multiple'):
        write_head_sidecar(str(tmp_path / 'x.pt'), SPKS, 20)
    with pytest.raises(ValueError):
        write_head_sidecar(str(tmp_path / 'x.pt'), [], 0)


@pytest.mark.parametrize('change, message', [
    ({'spk_sha256': 'f' * 64}, 'speaker-list sha256'),
    ({'num_classes': 8}, 'classes 8 != 7'),
    ({'num_classes': '7'}, 'classes'),
    ({'rows': 14}, 'rows 14 != 21'),
    ({'rows': None}, 'rows None'),
    ({'virtual_classes': 4}, 'virtual_classes'),
    ({'k_per_class_sha256': 'abc'}, 'k_per_class_sha256'),
])
def test_verify_refuses_any_mismatch(tmp_path, change, message):
    record = write_head_sidecar(str(tmp_path / 'x.pt'), SPKS, ROWS)
    with pytest.raises(HeadMismatchError, match=message):
        verify_head(dict(record, **change), SPKS, ROWS, 'record')


def test_verify_refuses_another_list_or_row_count(tmp_path):
    record = write_head_sidecar(str(tmp_path / 'x.pt'), SPKS, ROWS)
    with pytest.raises(HeadMismatchError):
        verify_head(record, SPKS[:-1] + ['other'], ROWS, 'record')
    with pytest.raises(HeadMismatchError):
        verify_head(record, SPKS, 2 * len(SPKS), 'record')
    verify_head(dict(record, virtual_classes=0), SPKS, ROWS, 'record')


def test_existing_sidecar_format_is_accepted():
    """Keys of a sidecar written by the original tools (extra provenance)."""
    record = {
        'spk_sha256': speaker_list_digest(SPKS), 'num_classes': 7, 'rows': 21,
        'rows_per_class': 3, 'source': '/somewhere/init.classes.pt',
        'train_label': 'data/list/utt2spk', 'frontend_num_layers': 22,
        'imprinted': {'classes': 2, 'first_class_id': 3, 'last_class_id': 4,
                      'corpus': 'x', 'tool': 'imprint', 'embeddings': 'e.scp',
                      'emb_utt2spk': 'utt2spk'}}
    verify_head(record, SPKS, ROWS, 'record')


def test_check_fails_closed_without_any_record(tmp_path):
    with pytest.raises(HeadMismatchError, match='cannot be verified'):
        check_run_head(str(tmp_path / 'model_0.pt'), str(tmp_path), SPKS, ROWS)
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize('text', ['', '{not json', '[1, 2]', '"text"'])
def test_check_fails_closed_on_unreadable_record(tmp_path, text):
    ckpt = str(tmp_path / 'model_0.pt')
    with open(head_sidecar_path(ckpt), 'w') as fout:
        fout.write(text)
    with pytest.raises(HeadMismatchError):
        check_run_head(ckpt, str(tmp_path), SPKS, ROWS)
    assert not os.path.exists(tmp_path / 'head_spk.json')


def test_check_with_sidecar_writes_the_run_record(tmp_path):
    ckpt = str(tmp_path / 'init' / 'model_0.pt')
    models = tmp_path / 'models'
    os.makedirs(tmp_path / 'init')
    os.makedirs(models)
    write_head_sidecar(ckpt, SPKS, ROWS)
    assert check_run_head(ckpt, str(models), SPKS, ROWS, write=False) \
        == 'sidecar'
    assert os.listdir(models) == []
    assert check_run_head(ckpt, str(models), SPKS, ROWS) == 'sidecar'
    assert read(models / 'head_spk.json') == {
        'spk_sha256': speaker_list_digest(SPKS), 'num_classes': 7,
        'rows': 21, 'verified_from': 'sidecar', 'source': ckpt}
    # resume from a checkpoint of the run: no sidecar, the run record counts
    resumed = str(models / 'model_4.pt')
    assert check_run_head(resumed, str(models), SPKS, ROWS) == 'run'
    with pytest.raises(HeadMismatchError, match='head_spk.json'):
        check_run_head(resumed, str(models), SPKS + ['new'], ROWS + 3)


def test_bad_sidecar_wins_over_good_run_record_and_writes_nothing(tmp_path):
    good = str(tmp_path / 'good.pt')
    write_head_sidecar(good, SPKS, ROWS)
    check_run_head(good, str(tmp_path), SPKS, ROWS)
    before = read(tmp_path / 'head_spk.json')
    bad = str(tmp_path / 'bad.pt')
    write_head_sidecar(bad, SPKS[:-1] + ['zz'], ROWS)
    with pytest.raises(HeadMismatchError, match='bad.pt.head.json'):
        check_run_head(bad, str(tmp_path), SPKS, ROWS)
    assert read(tmp_path / 'head_spk.json') == before


def test_mismatch_does_not_write_the_run_record(tmp_path):
    ckpt = str(tmp_path / 'model_0.pt')
    write_head_sidecar(ckpt, SPKS, ROWS)
    with pytest.raises(HeadMismatchError):
        check_run_head(ckpt, str(tmp_path), SPKS, ROWS + 3)
    assert not os.path.exists(tmp_path / 'head_spk.json')
