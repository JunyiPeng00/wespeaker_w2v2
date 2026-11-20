"""Epoch loop and run loop on the tiny model (CPU, one process).

The step itself is compared with the original trainer in tests/test_step.py;
here: the loop adds nothing to the steps, its log, its stop and save
decisions, and (``OVSB_SOURCE_TREE``) the table rows against the rows the
original epoch function prints for the same batches.
"""
import copy
import logging
import os

import pytest
import torch
import torch.distributed as dist
import yaml

from loop_helpers import FakeData, fake_batch, make_run
from test_build import (TINY_CLASSES, make_tiny_assets, needs_source,
                        run_source, tiny_config, write_tiny_legacy)
from test_step import make_bundle, reseed, tiny_batches

from ovsb.config import load_legacy, schema
from ovsb.config.schema import CODECS
from ovsb.engine import checkpoint as ckpt
from ovsb.engine import loop as loop_mod
from ovsb.engine.arm_log import make_arm_logger
from ovsb.engine.checkpoint import FRESH, PARTIAL, ResumePoint
from ovsb.engine.console import bottom_line
from ovsb.engine.guards import NonFiniteGuard
from ovsb.engine.loop import FINISHED, SEGMENT, SIGNAL, fit, run_epoch
from ovsb.engine.optim import build_optimizer, build_schedulers
from ovsb.engine.signals import StopFlag
from ovsb.engine.step import SKIP_LOSS, train_step
from ovsb.data.policies import PTYPES
from ovsb.model.build import build_model

CPU = torch.device('cpu')
STEPS = 6


@pytest.fixture(scope='module')
def tiny_assets(tmp_path_factory):
    return make_tiny_assets(tmp_path_factory.mktemp('tiny_w2vbert'))


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def capture_logger(name):
    logger = logging.Logger(name)
    handler = Capture()
    logger.addHandler(handler)
    return logger, handler.lines


class Setup:
    """Model, objective, optimiser and schedulers of a tiny run."""

    def __init__(self, tmp_path, assets, *overrides, seed=21):
        self.cfg = tiny_config(assets, *make_run(tmp_path, assets, 1, 3,
                                                 STEPS, *overrides))
        torch.manual_seed(seed)
        self.model = build_model(self.cfg, TINY_CLASSES)
        self.bundle = make_bundle(self.cfg)
        self.optimizer = build_optimizer(self.model, self.cfg)
        self.scheduler, self.margin = build_schedulers(
            self.optimizer, self.model, self.cfg)
        self.model_dir = os.path.join(self.cfg.run.exp_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)

    def args(self):
        return (self.model, self.bundle, self.cfg, CPU, self.optimizer,
                self.scheduler, self.margin)

    def weights(self):
        return {k: v.clone() for k, v in self.model.state_dict().items()}


def seeded(batches, seed=50):
    """Hand out ``batches``, reseeding the frame mask before each step."""
    for i, batch in enumerate(batches):
        reseed(seed + i)
        yield batch


def rows_of(lines):
    return [[c.strip() for c in line.strip('|').split('|')]
            for line in lines if line.startswith('|')]


# ------------------------------------------------------------- run_epoch
def test_loop_adds_nothing_to_the_steps(tiny_assets, tmp_path):
    """N steps through run_epoch == the same batches fed step by step."""
    a = Setup(tmp_path / 'a', tiny_assets)
    b = Setup(tmp_path / 'b', tiny_assets)
    order = a.cfg.data.sampler.arm_order
    batches = [fake_batch(100 + i, order) for i in range(STEPS)]
    epoch = 2
    res = run_epoch(seeded(batches), *a.args(), epoch,
                    arm_log=make_arm_logger(a.cfg, PTYPES, CPU))
    assert (res.steps_done, res.stopped, res.skipped) == (STEPS, False, 0)
    b.model.train()
    losses = []
    for i, batch in enumerate(batches):
        reseed(50 + i)
        step = train_step(b.model, b.bundle, batch, b.cfg, CPU, b.optimizer,
                          b.scheduler, b.margin, (epoch - 1) * STEPS + i)
        losses.append(float(step.loss))
    wa, wb = a.weights(), b.weights()
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    assert a.bundle.center.n_upd == b.bundle.center.n_upd > 0
    assert torch.equal(a.bundle.center.mean, b.bundle.center.mean)
    assert a.scheduler.current_iter == b.scheduler.current_iter == 2 * STEPS
    assert res.totals.steps == STEPS
    assert res.totals.loss == pytest.approx(sum(losses) / STEPS, rel=1e-12)


def test_log_lines_and_rows(tiny_assets, tmp_path):
    s = Setup(tmp_path, tiny_assets)
    logger, lines = capture_logger('rows')
    data = FakeData(s.cfg)
    run_epoch(data.loader(1, 0, None), *s.args(), 1, logger,
              arm_log=make_arm_logger(s.cfg, PTYPES, CPU))
    rows = rows_of(lines)
    # every 2 steps, plus the closing row of the epoch
    assert [r[:2] for r in rows] == [['1', '2'], ['1', '4'], ['1', '6'],
                                     ['1', '6']]
    assert all(len(r) == 7 for r in rows) and rows[-1] == rows[-2]
    assert float(rows[0][2]) > float(rows[0][3]) > 0  # Lr > LrFE
    assert all(r[4] == '0.2' for r in rows)
    armlog = [i for i, line in enumerate(lines) if line.startswith('ARMLOG')]
    table = [i for i, line in enumerate(lines) if line.startswith('|')]
    pauc = [i for i, line in enumerate(lines)
            if line.startswith('pair_auc loss')]
    assert len(pauc) == 3 and armlog and pauc[0] < armlog[0] < table[0]
    assert any('at epoch 1 batch 2' in line for line in lines)
    assert sum(line.startswith('pair-auc centre') for line in lines) == 3
    assert sum(line.startswith('time_mask') for line in lines) == 3
    # a process that does not print runs the same steps silently
    quiet = Setup(tmp_path / 'quiet', tiny_assets)
    run_epoch(FakeData(quiet.cfg).loader(1, 0, None), *quiet.args(), 1, None,
              arm_log=make_arm_logger(quiet.cfg, PTYPES, CPU))


def test_start_step_and_global_index(tiny_assets, tmp_path, monkeypatch):
    s = Setup(tmp_path, tiny_assets)
    seen = []

    def spy(model, bundle, batch, cfg, device, optimizer, scheduler, margin,
            cur_iter, amp=None):
        seen.append(cur_iter)
        return train_step(model, bundle, batch, cfg, device, optimizer,
                          scheduler, margin, cur_iter, amp)

    monkeypatch.setattr(loop_mod, 'train_step', spy)
    logger, lines = capture_logger('start')
    data = FakeData(s.cfg)
    res = run_epoch(data.loader(3, 1, None), *s.args(), 3, logger,
                    start_step=4)
    assert seen == [2 * STEPS + 4, 2 * STEPS + 5]
    assert res.steps_done == STEPS and res.totals.steps == 2
    assert len(data.seen) == 2  # only the remaining batches were drawn
    assert [r[:2] for r in rows_of(lines)] == [['3', '6'], ['3', '6']]
    with pytest.raises(ValueError, match='outside the epoch'):
        run_epoch(iter(()), *s.args(), 1, start_step=STEPS)


def test_loader_that_ends_early_is_an_error(tiny_assets, tmp_path):
    s = Setup(tmp_path, tiny_assets)
    order = s.cfg.data.sampler.arm_order
    with pytest.raises(RuntimeError, match='ended after step 2 of 6'):
        run_epoch([fake_batch(i, order) for i in range(2)], *s.args(), 1)


def test_stop_flag_ends_the_epoch_after_the_current_step(tiny_assets,
                                                         tmp_path):
    s = Setup(tmp_path, tiny_assets)
    stop = StopFlag()
    order = s.cfg.data.sampler.arm_order

    def batches():
        for i in range(STEPS):
            if i == 2:
                stop.request()  # while step index 2 is being prepared
            yield fake_batch(i, order)

    logger, lines = capture_logger('stop')
    res = run_epoch(batches(), *s.args(), 1, logger, stop)
    assert (res.steps_done, res.stopped, res.totals.steps) == (3, True, 3)
    assert rows_of(lines)[-1][:2] == ['1', '3']  # the closing row
    assert s.scheduler.current_iter == 3


def test_non_finite_steps_are_counted_not_logged(tiny_assets, tmp_path,
                                                 monkeypatch):
    s = Setup(tmp_path, tiny_assets)

    def step(model, bundle, batch, cfg, device, optimizer, scheduler, margin,
             cur_iter, amp=None):
        res = train_step(model, bundle, batch, cfg, device, optimizer,
                         scheduler, margin, cur_iter, amp)
        if cur_iter in (1, 3):  # the 2nd and 4th step: loss not finite
            res.skipped = SKIP_LOSS
        return res

    monkeypatch.setattr(loop_mod, 'train_step', step)
    logger, lines = capture_logger('skip')
    res = run_epoch(FakeData(s.cfg).loader(1, 0, None), *s.args(), 1, logger)
    assert (res.steps_done, res.skipped, res.totals.steps) == (STEPS, 2, 4)
    # the rows of the skipped steps 2 and 4 are not printed
    assert [r[1] for r in rows_of(lines)] == ['6', '6']
    assert sum('non-finite loss' in line for line in lines) == 2

    def always(*args, **kwargs):
        res = train_step(*args, **kwargs)
        res.skipped = SKIP_LOSS
        return res

    monkeypatch.setattr(loop_mod, 'train_step', always)
    monkeypatch.setattr(loop_mod, 'NonFiniteGuard',
                        lambda: NonFiniteGuard(3))
    with pytest.raises(RuntimeError, match='3 consecutive non-finite'):
        run_epoch(FakeData(s.cfg).loader(1, 0, None), *s.args(), 1)


def test_failed_codec_calls_are_counted_and_reported(tiny_assets, tmp_path,
                                                     monkeypatch):
    """``batch['chain_fail']`` (codec index, -1 = none): per interval and
    per epoch, summed over the processes; nothing is printed without one."""
    s = Setup(tmp_path, tiny_assets)
    order = s.cfg.data.sampler.arm_order
    batches = [fake_batch(i, order) for i in range(STEPS)]
    for batch in batches:
        batch['chain_fail'] = torch.full((8, ), -1)
    batches[0]['chain_fail'][1] = CODECS.index('opus')
    batches[1]['chain_fail'][[0, 5]] = CODECS.index('g722')
    batches[5]['chain_fail'][7] = CODECS.index('mp3')
    logger, lines = capture_logger('fails')
    res = run_epoch(batches, *s.args(), 1, logger)
    assert res.chain_fails == 4
    found = [x for x in lines if x.startswith('codec chain')]
    assert found == [
        "codec chain: 3 failed call(s) since the last report, rows kept "
        "without the codec {'opus': 1, 'g722': 2} at epoch 1 batch 2",
        "codec chain: 1 failed call(s) since the last report, rows kept "
        "without the codec {'mp3': 1} at epoch 1 batch 6"]
    # the count is a sum over all processes
    meter = loop_mod.ChainFailMeter(CPU)
    meter.add(batches[1])
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'all_reduce', lambda t: t.mul_(3))
    assert meter.flush() == {'g722': 6} and meter.epoch_total == 6
    assert meter.flush() == {}
    meter.add({'key': ['no such field']})
    assert meter.flush() == {}


# ------------------------------------------------------------------- fit
def run_fit(s, data, start=None, **kwargs):
    logger, lines = capture_logger('fit')
    res = fit(*s.args(), data, start or ResumePoint(FRESH), 0, 1, logger,
              **kwargs)
    return res, lines


def test_fit_runs_saves_and_links(tiny_assets, tmp_path):
    s = Setup(tmp_path, tiny_assets, 'run.save_epoch_interval=2')
    data = FakeData(s.cfg)
    res, lines = run_fit(s, data)
    assert (res.reason, res.finished, res.epoch, res.step) == \
        (FINISHED, True, 3, STEPS)
    # epoch 2 by the interval, epoch 3 as one of the last num_avg epochs
    assert sorted(os.listdir(s.model_dir)) == [
        'aux_state_3.pt', 'final_model.pt', 'model_2.pt', 'model_3.pt',
        'train_state_3.pt']
    assert os.readlink(os.path.join(s.model_dir, 'final_model.pt')) == \
        'model_3.pt'
    # stream 0 everywhere; a loader generator from the second epoch on
    assert data.calls == [(1, 0, False), (2, 0, True), (3, 0, True)]
    assert lines[-1] == bottom_line()
    assert sum(line.startswith('fake data:') for line in lines) == 3
    assert sum(line.startswith('epoch ') for line in lines) == 3
    assert s.scheduler.current_iter == 3 * STEPS


def test_fit_segment_saves_at_its_boundary(tiny_assets, tmp_path):
    s = Setup(tmp_path, tiny_assets, 'run.save_epoch_interval=2')
    res, lines = run_fit(s, FakeData(s.cfg), epochs_this_run=1)
    assert (res.reason, res.epoch, res.step) == (SEGMENT, 2, 0)
    # epoch 1 is saved although the interval alone would not save it
    assert sorted(os.listdir(s.model_dir)) == [
        'aux_state_1.pt', 'model_1.pt', 'train_state_1.pt']
    assert '1 epoch(s) of this process done: stopping after epoch 1 of 3' \
        in lines
    with pytest.raises(ValueError, match='>= 1'):
        run_fit(s, FakeData(s.cfg), epochs_this_run=0)


def test_fit_stop_inside_an_epoch_saves_a_partial_state(tiny_assets,
                                                        tmp_path):
    s = Setup(tmp_path, tiny_assets)
    stop, data = StopFlag(), FakeData(s.cfg)
    real = data._batches

    def batches(epoch, stream, base):
        for i, batch in enumerate(real(epoch, stream, base)):
            if (epoch, i) == (2, 3):
                stop.request()
            yield batch

    data._batches = batches
    res, lines = run_fit(s, data, stop=stop)
    assert (res.reason, res.epoch, res.step) == (SIGNAL, 2, 4)
    assert sorted(os.listdir(s.model_dir)) == [
        'aux_state_1.pt', 'model_1.pt', 'partial_model.pt',
        'partial_state.pt', 'train_state_1.pt']
    point = ckpt.find_resume_point(s.model_dir)
    assert (point.kind, point.start_epoch, point.start_step,
            point.start_stream) == (PARTIAL, 2, 4, 1)
    state = torch.load(os.path.join(s.model_dir, 'partial_state.pt'),
                       weights_only=False)
    assert (state['world'], state['epoch_iter'], state['stream']) == \
        (1, STEPS, 0)
    assert len(state['center']) == 1
    assert state['center'][0]['n_upd'] == s.bundle.center.n_upd
    assert 'stop request: epoch 2 interrupted after step 4 of 6, partial ' \
        'state saved' in lines

    # the next process: same epoch and step, stream 1, then stream 0 again
    data2 = FakeData(s.cfg)
    res2, _ = run_fit(s, data2, start=point)
    assert res2.finished
    assert data2.calls == [(2, 1, True), (3, 0, True)]
    assert [x[:3] for x in data2.seen[:2]] == [(2, 1, 0), (2, 1, 1)]
    assert len(data2.seen) == 2 + STEPS
    assert not os.path.exists(os.path.join(s.model_dir, 'partial_state.pt'))
    assert s.scheduler.current_iter == 3 * STEPS


def test_fit_stop_on_the_last_step_is_an_epoch_boundary(tiny_assets,
                                                        tmp_path):
    s = Setup(tmp_path, tiny_assets, 'run.save_epoch_interval=2')
    stop, data = StopFlag(), FakeData(s.cfg)
    real = data._batches

    def batches(epoch, stream, base):
        for i, batch in enumerate(real(epoch, stream, base)):
            if (epoch, i) == (1, STEPS - 1):
                stop.request()
            yield batch

    data._batches = batches
    res, lines = run_fit(s, data, stop=stop)
    assert (res.reason, res.epoch, res.step) == (SIGNAL, 2, 0)
    assert sorted(os.listdir(s.model_dir)) == [
        'aux_state_1.pt', 'model_1.pt', 'train_state_1.pt']
    assert 'stop request: stopping after epoch 1 of 3' in lines


def test_fit_stop_during_the_save_is_seen_before_the_next_epoch(
        tiny_assets, tmp_path, monkeypatch):
    s = Setup(tmp_path, tiny_assets, 'run.save_epoch_interval=2')
    stop, data = StopFlag(), FakeData(s.cfg)
    real = loop_mod._epoch_summary

    def summary(*args, **kwargs):  # runs between the last step and the save
        real(*args, **kwargs)
        stop.request()

    monkeypatch.setattr(loop_mod, '_epoch_summary', summary)
    res, _ = run_fit(s, data, stop=stop)
    assert (res.reason, res.epoch, res.step) == (SIGNAL, 2, 0)
    assert data.calls == [(1, 0, False)]  # epoch 2 was not started
    assert os.path.exists(os.path.join(s.model_dir, 'train_state_1.pt'))


def test_fit_stop_in_the_last_epoch_end_finishes(tiny_assets, tmp_path):
    s = Setup(tmp_path, tiny_assets)
    stop, data = StopFlag(), FakeData(s.cfg)
    real = data._batches

    def batches(epoch, stream, base):
        for i, batch in enumerate(real(epoch, stream, base)):
            if (epoch, i) == (3, STEPS - 1):
                stop.request()
            yield batch

    data._batches = batches
    res, _ = run_fit(s, data, stop=stop)
    assert res.finished
    assert os.path.exists(os.path.join(s.model_dir, 'final_model.pt'))


# ------------------------------------------------ rows vs the original loop
ROW_DRIVER = r'''
import logging, random
from wespeaker.utils import executor

model = build_model()
model.load_state_dict(torch.load(sys.argv[3]), strict=True)
batches = torch.load(sys.argv[4])
seed, epoch = int(sys.argv[5]), int(sys.argv[6])
ddp, optimizer, scheduler, margin, epoch_iter = build_optim(model)
configs['enable_amp'] = False  # CPU, fp32
lines = []


class Capture(logging.Handler):
    def emit(self, record):
        lines.append(record.getMessage())


logger = logging.Logger('source')
logger.addHandler(Capture())


def loader():
    for i, batch in enumerate(batches):
        random.seed(seed + i)
        np.random.seed(seed + i)
        torch.manual_seed(seed + i)
        yield batch


executor.run_epoch(loader(), epoch_iter, ddp, torch.nn.CrossEntropyLoss(),
                   optimizer, scheduler, margin, epoch, logger,
                   torch.amp.GradScaler('cuda', enabled=False),
                   device=torch.device('cpu'), configs=configs)
torch.save({'tree': wespeaker.__file__, 'lines': lines,
            'epoch_iter': epoch_iter}, sys.argv[2])
'''


@needs_source
def test_table_rows_equal_the_source_rows(tiny_assets, tmp_path):
    """Same weights, batches and frame masks through both epoch loops.

    Equal, as printed: epoch, batch, margin, loss, accuracy of every row
    (two intervals and the closing row) and the loss lines above them. The
    original prints one learning rate, that of the encoder group (its
    ``Lr``); here that value is the ``LrFE`` column and ``Lr`` is the rate
    of the other group.
    """
    legacy, _ = write_tiny_legacy(tmp_path, tiny_assets, center_warmup=1)
    with open(legacy) as fin:
        tree = yaml.safe_load(fin)
    steps = 4
    world, batch = len(tree['gpus']), tree['dataloader_args']['batch_size']
    tree['dataset_args']['sample_num_per_epoch'] = steps * world * batch
    tree['log_batch_interval'] = 2
    tree['scheduler_args']['epoch_iter'] = steps  # derived values the
    tree['margin_update']['epoch_iter'] = steps   # saved config carries
    with open(legacy, 'w') as fout:
        yaml.safe_dump(tree, fout, sort_keys=False)
    from test_build import AUTHORED, SOURCE_RECIPE
    cfg = load_legacy(legacy, os.path.join(SOURCE_RECIPE, AUTHORED))
    assert schema.epoch_iter(cfg) == steps
    order = cfg.data.sampler.arm_order
    batches = tiny_batches(order) + [fake_batch(9, order)]
    torch.manual_seed(31)
    model = build_model(cfg, TINY_CLASSES)
    state_file, batch_file = tmp_path / 'state.pt', tmp_path / 'batches.pt'
    torch.save(model.state_dict(), state_file)
    torch.save(batches, batch_file)
    epoch = 2
    src = run_source(ROW_DRIVER, tmp_path, legacy, state_file, batch_file,
                     50, epoch)
    assert src['epoch_iter'] == steps

    optimizer = build_optimizer(model, cfg)
    scheduler, margin = build_schedulers(optimizer, model, cfg)
    logger, lines = capture_logger('mine')
    run_epoch(seeded(copy.deepcopy(batches)), model, make_bundle(cfg), cfg,
              CPU, optimizer, scheduler, margin, epoch, logger)
    mine, theirs = rows_of(lines), rows_of(src['lines'])
    assert [r[1] for r in theirs] == ['2', '4', '4'] and len(theirs[0]) == 6
    assert len(mine) == len(theirs) == 3 and len(mine[0]) == 7
    for got, want in zip(mine, theirs):
        assert got[:2] + got[4:] == want[:2] + want[3:]  # all but the rates
        assert got[3] == want[2]                          # LrFE == their Lr
        assert float(got[2]) > float(got[3])
    for prefix in ('consistency loss', 'pair_auc loss', 'time_mask'):
        assert [x for x in lines if x.startswith(prefix)] == \
            [x for x in src['lines'] if x.startswith(prefix)], prefix
    print('\nrows (recipe | source):')
    for got, want in zip(mine, theirs):
        print('  %s | %s' % (got, want))
