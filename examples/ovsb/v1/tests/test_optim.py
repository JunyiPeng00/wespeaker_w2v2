"""Optimiser param groups and the learning-rate / margin schedule.

Parity tests (see tests/test_build.py for the environment variables)
compare every step of the reference run's 14 x 3,645 step schedule with
the scheduler objects of the source tree, with the values printed in the
reference run's train.log and, with ``OVSB_PARITY_REF``, with the hash the
source tree's parity harness recorded.
"""
import hashlib
import math
import os
import re
import struct

import pytest
import torch

from test_build import (PARITY_REF, RUN_DIR, SOURCE_RECIPE, TINY_CLASSES,
                        load_reference, make_tiny_assets, needs_source,
                        run_source, tiny_config, write_tiny_legacy)

from ovsb.config import schema
from ovsb.engine.logging import row_line
from ovsb.engine.optim import (build_optimizer, build_schedulers,
                               learning_rates)
from ovsb.model.build import build_model

REF_EPOCHS, REF_EPOCH_ITER = 14, 3645


@pytest.fixture(scope='module')
def tiny_assets(tmp_path_factory):
    return make_tiny_assets(tmp_path_factory.mktemp('tiny_w2vbert'))


def build_all(cfg):
    model = build_model(cfg, TINY_CLASSES)
    optimizer = build_optimizer(model, cfg)
    scheduler, margin = build_schedulers(optimizer, model, cfg)
    return model, optimizer, scheduler, margin


def group_names(model, optimizer):
    name_of = {id(p): n for n, p in model.named_parameters()}
    return [[name_of[id(p)] for p in g['params']]
            for g in optimizer.param_groups]


def trajectory(optimizer, scheduler, margin, steps):
    """[(lr of every group..., margin)] for steps 0 .. steps - 1."""
    rows = []
    for cur in range(steps):
        scheduler.step(cur)
        margin.step(cur)
        rows.append([float(g['lr']) for g in optimizer.param_groups]
                    + [float(margin.get_margin())])
    return rows


# ------------------------------------------------------------------- unit
def test_param_groups(tiny_assets):
    cfg = tiny_config(tiny_assets)
    model, optimizer, _, _ = build_all(cfg)
    assert isinstance(optimizer, torch.optim.AdamW)
    names = group_names(model, optimizer)
    order = [n for n, _ in model.named_parameters()]
    assert len(names) == 2
    # group 0: the encoder; group 1: backend + head; model order inside
    assert names[0] == [n for n in order if n.startswith('frontend.')]
    assert names[1] == [n for n in order if not n.startswith('frontend.')]
    assert names[1][-1] == 'projection.weight' and len(names[1]) == 15
    assert [g['_is_frontend'] for g in optimizer.param_groups] == [True, False]
    for group in optimizer.param_groups:
        assert group['weight_decay'] == cfg.optim.weight_decay == 1e-4
        assert group['fused'] is True
        assert group['betas'] == (0.9, 0.999) and group['eps'] == 1e-8
    assert sum(len(g['params']) for g in optimizer.param_groups) == len(order)


def test_schedule_formula(tiny_assets):
    cfg = tiny_config(tiny_assets, 'run.num_epochs=3',
                      'data.sampler.sample_num_per_epoch=38400')
    epoch_iter = schema.epoch_iter(cfg)
    assert epoch_iter == 100 and schema.scale_ratio(cfg) == 6.0
    _, optimizer, scheduler, margin = build_all(cfg)
    opt = cfg.optim
    total, warm = 3 * epoch_iter, opt.warm_up_epoch * epoch_iter
    rows = trajectory(optimizer, scheduler, margin, total)
    assert rows[0][:2] == [0.0, 0.0]  # warm-up starts at zero
    for cur in (1, 7, 13, 14, 15, 99, 100, 250, 299):
        coeff = 6.0 * min(1.0, cur / warm)
        base = coeff * opt.initial_lr * math.exp(
            cur / total * math.log(opt.final_lr / opt.initial_lr))
        assert rows[cur][1] == pytest.approx(base, rel=1e-12)
        assert rows[cur][0] == rows[cur][1] * opt.frontend_lr_ratio == \
            0.3 * rows[cur][1]
    peak = max(range(total), key=lambda c: rows[c][1])
    assert peak == round(warm) == 14  # the last warm-up step
    assert all(r[2] == cfg.model.head.margin == 0.2 for r in rows)
    scheduler.step(250)
    assert learning_rates(optimizer) == (rows[250][1], rows[250][0])


def test_step_index_alone_decides_the_rates(tiny_assets):
    """A rebuilt scheduler (resume) continues the same trajectory."""
    cfg = tiny_config(tiny_assets)
    _, optimizer, scheduler, margin = build_all(cfg)
    first = trajectory(optimizer, scheduler, margin, 40)
    _, optimizer, scheduler, margin = build_all(cfg)
    for cur in (39, 5, 20):
        scheduler.step(cur)
        assert [g['lr'] for g in optimizer.param_groups] == first[cur][:2]


def test_margin_scheduler_sets_the_head_margin(tiny_assets):
    cfg = tiny_config(tiny_assets, 'model.head.margin=0.35')
    model, _, _, margin = build_all(cfg)
    model.projection.update(0.0)
    assert margin.get_margin() == 0.0
    cfg2 = tiny_config(tiny_assets, 'model.head.margin=0.35')
    model, optimizer, _, margin = build_all(cfg2)
    for cur in (0, 1, 5000):
        margin.step(cur)
        assert margin.get_margin() == model.projection.m == 0.35


def test_frontend_modules_must_split_the_model(tiny_assets):
    cfg = tiny_config(tiny_assets, 'optim.frontend_modules=[encoder]')
    model = build_model(cfg, TINY_CLASSES)
    optimizer = build_optimizer(model, cfg)
    with pytest.raises(ValueError, match='optim.frontend_modules'):
        build_schedulers(optimizer, model, cfg)


# ----------------------------------------------------------- source parity
SCHEDULE_DRIVER = r'''
torch.manual_seed(3)
model = build_model()
ddp, optimizer, scheduler, margin, epoch_iter = build_optim(model)
name_of = {id(p): n for n, p in model.named_parameters()}
groups = [[name_of[id(p)] for p in g['params']]
          for g in optimizer.param_groups]
hyper = [{k: v for k, v in g.items() if k not in ('params', 'lr')}
         for g in optimizer.param_groups]
rows = []
for cur in range(configs['num_epochs'] * epoch_iter):
    scheduler.step(cur)
    margin.step(cur)
    rows.append([float(g['lr']) for g in optimizer.param_groups]
                + [float(margin.get_margin())])
torch.save({'tree': wespeaker.__file__, 'groups': groups, 'hyper': hyper,
            'rows': torch.tensor(rows, dtype=torch.float64),
            'epoch_iter': epoch_iter, 'get_lr': scheduler.get_lr()},
           sys.argv[2])
'''


@pytest.fixture(scope='module')
def reference_schedule(tiny_assets, tmp_path_factory):
    """(recipe-side rows, optimizer, model, source dump) of the full run."""
    tmp_path = tmp_path_factory.mktemp('schedule')
    legacy, cfg = write_tiny_legacy(tmp_path, tiny_assets)
    src = run_source(SCHEDULE_DRIVER, tmp_path, legacy)
    assert schema.epoch_iter(cfg) == src['epoch_iter'] == REF_EPOCH_ITER
    model, optimizer, scheduler, margin = build_all(cfg)
    rows = trajectory(optimizer, scheduler, margin,
                      REF_EPOCHS * REF_EPOCH_ITER)
    return rows, optimizer, model, src


@needs_source
def test_param_groups_equal_source(reference_schedule):
    _, optimizer, model, src = reference_schedule
    names = group_names(model, optimizer)
    assert names == src['groups']  # membership and order inside each group
    assert [sorted(n) for n in names] == [sorted(n) for n in src['groups']]
    hyper = [{k: v for k, v in g.items() if k not in ('params', 'lr')}
             for g in optimizer.param_groups]
    assert hyper == src['hyper']


@needs_source
def test_every_step_equals_source_scheduler(reference_schedule):
    rows, optimizer, _, src = reference_schedule
    assert len(rows) == 51030 == src['rows'].shape[0]
    mine = torch.tensor(rows, dtype=torch.float64)
    assert torch.equal(mine, src['rows'])  # lr of both groups and the margin
    # what the source printed as 'Lr' is the encoder group
    assert src['get_lr'] == rows[-1][0] == learning_rates(optimizer)[1]


@needs_source
def test_schedule_matches_the_train_log(reference_schedule):
    """Every 'Lr' / 'Margin' cell of the reference run's log is reproduced.

    The source table printed the encoder learning rate as 'Lr'; here that
    value is the 'LrFE' column.
    """
    rows = reference_schedule[0]
    cell = re.compile(r'\|\s*(\d+)\|\s*(\d+)\|\s*([-+.\de]+)\|\s*([-+.\de]+)\|'
                      r'\s*[-+.\de]+\|\s*[-+.\de]+\|\s*$')
    seen = {}
    with open(os.path.join(SOURCE_RECIPE, RUN_DIR, 'train.log')) as fin:
        for line in fin:
            match = cell.search(line)
            if match:
                seen.setdefault((int(match[1]), int(match[2])), set()).add(
                    (match[3], match[4]))
    assert len(seen) == REF_EPOCHS * (REF_EPOCH_ITER // 100 + 1)
    for (epoch, batch), printed in seen.items():
        cur = (epoch - 1) * REF_EPOCH_ITER + batch - 1
        lr_fe, lr, margin = rows[cur]
        mine = [c.strip() for c in row_line(
            epoch, batch, lr, lr_fe, margin, 0.0, 0.0).split('|')]
        assert printed == {(mine[4], mine[5])}, (epoch, batch)
        assert float(mine[3]) == pytest.approx(float(mine[4]) / 0.3, rel=1e-4)


@needs_source
@pytest.mark.skipif(not PARITY_REF, reason='OVSB_PARITY_REF is not set')
def test_schedule_hash_equals_reference_dump(reference_schedule):
    rows = reference_schedule[0]
    ref = load_reference()['schedule']
    assert ref['total_steps'] == len(rows) and ref['resume_invariant']
    digest = hashlib.sha256()
    for row in rows:
        digest.update(struct.pack('<3d', *row))
    assert digest.hexdigest() == ref['uninterrupted']['sha256']
    for cur, values in ref['uninterrupted']['samples'].items():
        assert rows[int(cur)] == values
