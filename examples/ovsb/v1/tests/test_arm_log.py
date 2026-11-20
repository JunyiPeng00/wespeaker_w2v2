"""Per-arm log: sums, line format, reduction, parity with the source."""
import json
import logging
import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ovsb.config.schema import AUG_TYPES, CODECS
from ovsb.engine.arm_log import ArmLogger

SOURCE_TREE = os.environ.get('OVSB_SOURCE_TREE')
needs_source = pytest.mark.skipif(
    not SOURCE_TREE, reason='OVSB_SOURCE_TREE is not set')
CPU = torch.device('cpu')
ARMS = ['vox2', 'cn1', 'mandarin_t1']
# code order of the positive-pair types in the source tree
PTYPES = ('xsession', 'xsession_segfix', '3ds_contrast', '3ds_xdev',
          '3ds_xseg', 'dialect', 'cn2_xgenre', 'same_segment',
          'cn2_hardpair', '3ds_farfar', 'ks_xphase', 'ks_dialect',
          'same_session')


def make_batch(seed, rows=24, classes=30, pair=False):
    gen = torch.Generator().manual_seed(seed)

    def ints(high, low=0):
        return torch.randint(low, high, (rows,), generator=gen)

    role = torch.tensor([1, 2] * (rows // 2)) if pair \
        else torch.zeros(rows, dtype=torch.long)
    batch = {
        'arm': ints(len(ARMS)), 'pair_role': role,
        'kpos': torch.arange(rows) % 2, 'ptype': ints(len(PTYPES)),
        'aug_code': ints(len(AUG_TYPES)),
        'chain_code': ints(len(CODECS), low=-1),
        'zr_lsb': torch.rand(rows, generator=gen), 'zr_run': ints(2),
        'zr_dith': ints(2), 'zr_bypass': ints(2),
        'eg_redraw': ints(4, low=-1), 'eg_fail': ints(2)}
    logits = 3.0 * torch.randn(rows, classes, generator=gen)
    labels = ints(classes)
    logits[torch.arange(0, rows, 3), labels[::3]] += 20.0  # some correct
    return batch, logits, labels


class Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def collecting_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = [Collect()]
    logger.propagate = False
    return logger, logger.handlers[0].lines


def test_sums_of_one_batch():
    batch, logits, labels = make_batch(0)
    arm_log = ArmLogger(ARMS, PTYPES, CPU)
    arm_log.add(batch, logits, labels)
    res = arm_log.flush(logging.getLogger('arm-test'), 1, 100)
    ce = F.cross_entropy(logits, labels, reduction='none')
    assert sum(d['rows'] for d in res.values()) == 24
    for k, name in enumerate(ARMS):
        sel = batch['arm'] == k
        d = res[name]
        assert d['rows'] == int(sel.sum())
        assert d['share'] == round(d['rows'] / 24, 4)
        assert d['ce'] == round(float(ce[sel].double().mean()), 4)
        assert d['acc'] == round(float(
            (logits.argmax(1) == labels)[sel].double().mean()), 4)
        assert d['view'] == 0.0
        assert d['zr_lsb50'] == round(float(
            (batch['zr_lsb'][sel] > 0.5).double().mean()), 4)
        chain = batch['chain_code'][sel]
        assert d['chain'] == round(float((chain >= 0).double().mean()), 4)
        if (chain >= 0).any():
            assert sum(d['codec'].values()) == pytest.approx(1.0, abs=2e-4)
            assert set(d['codec']) == {CODECS[c] for c in chain[chain >= 0]}
        gate = batch['eg_redraw'][sel]
        if (gate >= 0).any():
            n = float((gate >= 0).sum())
            assert d['eg_gated'] == round(n / d['rows'], 4)
            assert d['eg_redraw'] == round(float(gate.clamp(min=0).sum()) / n,
                                           4)
        assert set(d['aug']) == {AUG_TYPES[c] for c in batch['aug_code'][sel]}
        first = sel & (batch['kpos'] == 0)
        if first.any():
            assert set(d['ptype']) == {PTYPES[c] for c in
                                       batch['ptype'][first]}
        else:
            assert 'ptype' not in d


def test_line_format_and_reset():
    logger, lines = collecting_logger('arm-format')
    arm_log = ArmLogger(['a', 'unused', 'b'], PTYPES, CPU)
    batch, logits, labels = make_batch(1)
    batch['arm'] = torch.tensor([0, 2] * 12)
    arm_log.add(batch, logits, labels)
    res = arm_log.flush(logger, 3, 200)
    assert list(res) == ['a', 'b'] and len(lines) == 2
    head, _, body = lines[0].partition(' {')
    assert head == 'ARMLOG epoch 3 batch 200 arm a'
    parsed = json.loads('{' + body)
    assert parsed == res['a'] and list(parsed) == sorted(parsed)
    assert arm_log.flush(logger, 3, 300) == {}  # reset by the flush
    assert arm_log.flush(None) is None


def test_views_and_groups_of_a_pair_batch():
    batch, logits, labels = make_batch(2, pair=True)
    batch['arm'] = torch.zeros(24, dtype=torch.long)
    batch['kpos'] = torch.tensor([0, 0, 1, 1] * 6)
    batch['ptype'] = torch.tensor([6, 0, 3, 3] * 6)
    arm_log = ArmLogger(ARMS, PTYPES, CPU)
    arm_log.add(batch, logits, labels)
    res = arm_log.flush(logging.getLogger('arm-test'))
    assert res['vox2']['view'] == 0.5
    # groups are counted on their first row that is not a view
    assert res['vox2']['ptype'] == {'cn2_xgenre': 1.0}


def test_code_outside_its_table_is_refused():
    batch, logits, labels = make_batch(3)
    batch['aug_code'][5] = len(AUG_TYPES)
    with pytest.raises(ValueError, match='outside the table'):
        ArmLogger(ARMS, PTYPES, CPU).add(batch, logits, labels)


def test_flush_sums_over_processes(monkeypatch):
    def fake_all_reduce(tensor, op=None):
        tensor.mul_(2)  # a second process with identical sums

    batch, logits, labels = make_batch(4)
    single = ArmLogger(ARMS, PTYPES, CPU)
    single.add(batch, logits, labels)
    want = single.flush(logging.getLogger('arm-test'))
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'all_reduce', fake_all_reduce)
    both = ArmLogger(ARMS, PTYPES, CPU)
    both.add(batch, logits, labels)
    got = both.flush(logging.getLogger('arm-test'))
    for name in want:
        assert got[name]['rows'] == 2 * want[name]['rows']
        assert {k: v for k, v in got[name].items() if k != 'rows'} == \
            {k: v for k, v in want[name].items() if k != 'rows'}


SOURCE_DRIVER = r'''
import logging, sys
import torch
import wespeaker
from wespeaker.utils.arm_log import ArmLogger
from wespeaker.dataset.episodic import PTYPES

data = torch.load(sys.argv[1])
lines = []
handler = logging.Handler()
handler.emit = lambda record: lines.append(record.getMessage())
logger = logging.getLogger('src')
logger.setLevel(logging.INFO)
logger.addHandler(handler)
arm_log = ArmLogger(data['arms'])
for batch, logits, labels in data['batches']:
    batch = dict(batch, label_b=torch.full_like(labels, -1),
                 lam=torch.ones(len(labels)))
    arm_log.add(batch, logits, labels)
arm_log.flush(logger, 2, 300)
torch.save({'tree': wespeaker.__file__, 'lines': lines,
            'ptypes': list(PTYPES)}, sys.argv[2])
'''


@needs_source
def test_lines_equal_the_source_logger(tmp_path):
    """Same batches -> character-identical ARMLOG lines."""
    batches = [make_batch(10), make_batch(11, pair=True), make_batch(12)]
    inp, out, script = (str(tmp_path / n) for n in
                        ('in.pt', 'out.pt', 'driver.py'))
    torch.save({'arms': ARMS, 'batches': batches}, inp)
    with open(script, 'w') as fout:
        fout.write(SOURCE_DRIVER)
    env = dict(os.environ, PYTHONPATH=SOURCE_TREE, PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, script, inp, out], check=True, env=env,
                   cwd=str(tmp_path))
    src = torch.load(out)
    assert os.path.abspath(src['tree']).startswith(
        os.path.abspath(SOURCE_TREE))
    assert tuple(src['ptypes']) == PTYPES
    logger, lines = collecting_logger('arm-parity')
    arm_log = ArmLogger(ARMS, PTYPES, CPU)
    for batch, logits, labels in batches:
        arm_log.add(batch, logits, labels)
    arm_log.flush(logger, 2, 300)
    assert len(lines) == len(ARMS) and lines == src['lines']
