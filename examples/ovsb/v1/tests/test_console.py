"""Progress table, epoch meter and the loss lines; run provenance."""
import json
import math
import os

import pytest
import torch
import torch.distributed as dist

from test_build import make_tiny_assets, tiny_config

from ovsb.engine import console as elog
from ovsb.engine.console import (EpochMeter, Totals, all_reduce_sum,
                                 bottom_line, detail_lines, header_lines,
                                 row_line)
from ovsb.engine.run_meta import collect_run_meta, sha256_file, write_run_meta
from ovsb.model.head_record import write_head_sidecar

CPU = torch.device('cpu')


def cells(line):
    return [c.strip() for c in line.strip('|').split('|')]


def test_header_has_both_learning_rates():
    lines = header_lines()
    assert len(lines) == 3
    assert cells(lines[1]) == ['Epoch', 'Batch', 'Lr', 'LrFE', 'Margin',
                               'Loss', 'Acc']
    assert lines[0] == lines[2] == bottom_line()
    assert len(lines[1]) == len(lines[0]) == 7 * 11 + 1


def test_row_format():
    line = row_line(1, 100, 1.92883e-05, 5.7865e-06, 0.2, 4.5454, 41.208)
    assert line == ('|         1|       100|1.9288e-05|5.7865e-06|       0.2|'
                    '    4.5454|    41.208|')


def logits_for(labels, predicted, classes=6):
    out = torch.zeros(len(labels), classes)
    out[torch.arange(len(labels)), torch.tensor(predicted)] = 5.0
    return out


def test_meter_means_over_steps_and_rows():
    meter = EpochMeter(CPU)
    labels = torch.tensor([0, 1, 2, 3])
    meter.add(torch.tensor(2.0), logits_for(labels, [0, 1, 2, 0]), labels,
              {'ce': torch.tensor(1.5), 'cons': None,
               'pauc': torch.tensor(3.0)}, {'pauc_pos': 4, 'pauc_neg': 2})
    labels2 = torch.tensor([1, 1])
    meter.add(torch.tensor(4.0), logits_for(labels2, [0, 0]), labels2,
              {'ce': torch.tensor(3.0), 'cons': torch.tensor(0.5),
               'pauc': None}, {'pauc_pos': 0, 'pauc_neg': 1})
    totals = meter.reduce()
    assert totals == Totals(steps=2, loss=3.0, acc=50.0, cons=0.5, cons_n=1,
                            pauc=3.0, pauc_n=1)
    # the pair statistics are those of the last batch with the term
    assert meter.last_stats == {'pauc_pos': 4, 'pauc_neg': 2}
    assert meter.reduce() == totals  # reading does not reset
    meter.reset()
    empty = meter.reduce()
    assert empty.steps == 0 and math.isnan(empty.loss) \
        and math.isnan(empty.acc) and math.isnan(empty.cons)


def test_meter_is_reduced_over_processes(monkeypatch):
    """Two processes: the logged loss / accuracy are the global means."""
    other = torch.tensor([3.0, 9.0, 12.0, 3.0, 0.0, 0.0, 4.0, 2.0],
                         dtype=torch.float64)

    def fake_all_reduce(tensor, op=None):
        tensor.add_(other)

    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'all_reduce', fake_all_reduce)
    meter = EpochMeter(CPU)
    labels = torch.tensor([0, 1, 2, 3])
    meter.add(torch.tensor(1.0), logits_for(labels, [0, 1, 2, 3]), labels,
              {'ce': torch.tensor(1.0), 'cons': None, 'pauc': None}, {})
    totals = meter.reduce()
    assert totals.steps == 4 and totals.loss == pytest.approx(2.5)
    assert totals.acc == pytest.approx(100.0 * 7 / 16)
    assert (totals.pauc, totals.pauc_n) == (2.0, 2)
    # the local sums are untouched by the reduction
    assert meter.sums[:2].tolist() == [1.0, 1.0]


def test_all_reduce_sum_single_process_copies():
    values = torch.arange(4, dtype=torch.float64)
    out = all_reduce_sum(values, CPU)
    out += 1
    assert values.tolist() == [0, 1, 2, 3]


class FakeCenter:
    def describe(self):
        return 'centre line'


def test_detail_lines():
    totals = Totals(steps=100, loss=4.5, acc=40.0, cons=0.4126, cons_n=27,
                    pauc=3.1652, pauc_n=100)
    stats = {'pauc_pos': 64, 'pauc_neg': 1040, 'pauc_cos_pos': 0.385,
             'pauc_cos_neg': 0.082}
    lines = detail_lines(totals, 1, 100, stats, FakeCenter(), [33421, 674760])
    assert lines == [
        'consistency loss 0.4126 (27 pair batches) at epoch 1 batch 100',
        'pair_auc loss 3.1652 (100 batches; last batch 64 pos 1040 neg, cos '
        'pos 0.385 neg 0.082) at epoch 1 batch 100',
        'centre line',
        'time_mask: 0.0495 of valid frames masked (33421 / 674760)']


def test_detail_lines_without_terms():
    nan = float('nan')
    totals = Totals(steps=3, loss=1.0, acc=0.0, cons=nan, cons_n=0, pauc=nan,
                    pauc_n=0)
    assert detail_lines(totals, 1, 3, {}, FakeCenter(), [5, 10]) == []
    totals.pauc, totals.pauc_n = 2.0, 1
    lines = detail_lines(totals, 1, 3, None, None, [0, 0])
    assert len(lines) == 1 and lines[0].startswith('pair_auc loss 2.0000 (1 ')


def test_module_does_not_shadow_stdlib_logging():
    import logging
    assert logging is not elog and hasattr(logging, 'getLogger')


# ------------------------------------------------------------ run provenance
@pytest.fixture(scope='module')
def cfg(tmp_path_factory):
    assets = make_tiny_assets(tmp_path_factory.mktemp('tiny_w2vbert'))
    return tiny_config(assets)


def test_run_meta(cfg, tmp_path, monkeypatch):
    label = tmp_path / 'utt2spk'
    label.write_text('u1 s1\nu2 s2\n')
    config_file = tmp_path / 'conf.yaml'
    config_file.write_text('run: {}\n')
    ckpt = str(tmp_path / 'model_0.pt')
    write_head_sidecar(ckpt, ['s1', 's2'], 6)
    cfg.data.lists.train_label = str(label)
    cfg.run.exp_dir = str(tmp_path / 'exp')
    monkeypatch.setenv('SLURM_JOB_ID', '4242')
    path = write_run_meta(cfg, str(config_file), ckpt, world_size=8,
                          cwd=str(tmp_path))
    assert path == os.path.join(cfg.run.exp_dir, 'run_meta_4242.json')
    with open(path) as fin:
        meta = json.load(fin)
    assert meta['slurm_job_id'] == '4242' and meta['world_size'] == 8
    assert meta['seed'] == cfg.run.seed
    assert meta['config'] == {'path': str(config_file),
                              'sha256': sha256_file(str(config_file))}
    assert meta['data']['train_label'] == {
        'path': str(label), 'sha256': sha256_file(str(label))}
    assert meta['data']['spk2corpus']['sha256'] is None  # file not there
    assert meta['checkpoint']['head_json']['num_classes'] == 2
    assert meta['checkpoint']['sha256_head_json'] == sha256_file(
        ckpt + '.head.json')
    assert set(meta['git']) == {'commit', 'dirty', 'branch'}


def test_run_meta_never_raises(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    cfg.run.exp_dir = str(tmp_path / 'exp2')
    meta = collect_run_meta(cfg)
    assert meta['config'] == {'path': None, 'sha256': None}
    assert meta['checkpoint'] == {'path': None, 'sha256_head_json': None,
                                  'head_json': None}
    path = write_run_meta(cfg)
    assert os.path.basename(path) == 'run_meta_pid%d.json' % os.getpid()
    blocker = tmp_path / 'file'
    blocker.write_text('x')
    cfg.run.exp_dir = str(blocker / 'below_a_file')
    assert write_run_meta(cfg) is None
