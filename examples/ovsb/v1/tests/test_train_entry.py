"""The training entry point: start, stop signals, resume, segments.

Tiny model on the CPU with the in-memory data of tests/loop_helpers.py.
Single-process runs call ``ovsb.bin.train.main`` in this process (signals
are real: the data object sends them to the process at a given batch).
The distributed tests start two processes with the gloo backend.

What "equal" means below is bit-identical tensors. The fake data is a pure
function of (seed, rank, epoch, stream, batch index, DataLoader base
seed), and the frame masks of the encoder come from the torch generator,
so equal weights after a resume also show that the generators were in the
state of the uninterrupted run.
"""
import copy
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time

import pytest
import torch
import yaml

from conftest import RECIPE_DIR
from loop_helpers import (DELAY_ENV, SIGNAL_ENV, TRACE_ENV, FakeData, argv_of,
                          fake_training, make_run)
from test_build import make_tiny_assets

from ovsb.bin import train as train_mod
from ovsb.config import load_config
from ovsb.engine import loop as loop_mod
from ovsb.engine.optim import learning_rates

STEPS, EPOCHS = 6, 3
DRIVER = os.path.join(RECIPE_DIR, 'tests', 'train_driver.py')
CONSTRUCTION = os.path.join(RECIPE_DIR, 'tests', 'construction_driver.py')


@pytest.fixture(scope='module')
def tiny_assets(tmp_path_factory):
    return make_tiny_assets(tmp_path_factory.mktemp('tiny_w2vbert'))


class Run:
    """One call of the entry point and what it left behind."""

    def __init__(self, sets, *flags, signals='', monkeypatch):
        self.data, self.steps, self.restored = [], [], None
        self.sets = list(sets)
        self.exp = [s for s in sets if s.startswith('run.exp_dir=')][0][12:]
        self.models = os.path.join(self.exp, 'models')
        real_step, real_fit = loop_mod.train_step, train_mod.fit

        def factory(cfg, rank, world, objects):
            self.data.append(FakeData(cfg, rank, world, objects))
            return self.data[-1]

        def step(model, bundle, batch, cfg, device, optimizer, scheduler,
                 margin, cur_iter, amp=None):
            res = real_step(model, bundle, batch, cfg, device, optimizer,
                            scheduler, margin, cur_iter, amp)
            self.steps.append((cur_iter, ) + learning_rates(optimizer)
                              + (margin.get_margin(), ))
            return res

        def fit(model, bundle, cfg, device, optimizer, scheduler, margin,
                data, start, *args, **kwargs):
            self.restored = dict(
                optimizer=copy.deepcopy(optimizer.state_dict()),
                center=bundle.center.state_dict(),
                scheduler_iter=scheduler.current_iter,
                weights={k: v.clone() for k, v in model.state_dict().items()},
                start=(start.start_epoch, start.start_step,
                       start.start_stream))
            return real_fit(model, bundle, cfg, device, optimizer, scheduler,
                            margin, data, start, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setenv(SIGNAL_ENV, signals)
            patch.delenv('WORLD_SIZE', raising=False)
            patch.setattr(loop_mod, 'train_step', step)
            patch.setattr(train_mod, 'fit', fit)
            torch.set_num_threads(4)
            self.rc = train_mod.main(argv_of(sets, *flags),
                                     build=fake_training,
                                     data_factory=factory)
        self.seen = self.data[0].seen if self.data else []
        self.calls = self.data[0].calls if self.data else []

    def files(self):
        return sorted(os.listdir(self.models))

    def load(self, name):
        return torch.load(os.path.join(self.models, name),
                          weights_only=False)

    def log(self):
        with open(os.path.join(self.exp, 'train.log')) as fin:
            return fin.read()


def equal_tensors(a, b):
    return list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)


def equal_optimizer(a, b):
    if a['param_groups'] != b['param_groups'] or \
            list(a['state']) != list(b['state']):
        return False
    return all(torch.equal(torch.as_tensor(v),
                           torch.as_tensor(b['state'][k][name]))
               for k in a['state'] for name, v in a['state'][k].items())


def equal_center(a, b):
    return all(torch.equal(a[k], b[k]) if k == 'acc' and a[k] is not None
               else a[k] == b[k] for k in a) and set(a) == set(b)


def optimizer_steps(state):
    return sorted({int(s['step']) for s in state['state'].values()})


@pytest.fixture(scope='module')
def reference(tiny_assets, tmp_path_factory):
    """The uninterrupted run of 3 epochs x 6 steps."""
    sets = make_run(tmp_path_factory.mktemp('reference'), tiny_assets)
    with pytest.MonkeyPatch.context() as patch:
        run = Run(sets, monkeypatch=patch)
    assert run.rc == 0
    return run


# ------------------------------------------------------------ a plain run
def test_uninterrupted_run(reference):
    run = reference
    assert run.files() == [
        'aux_state_3.pt', 'final_model.pt', 'head_spk.json', 'model_1.pt',
        'model_2.pt', 'model_3.pt', 'train_state_3.pt']
    assert [s[0] for s in run.steps] == list(range(EPOCHS * STEPS))
    assert run.calls == [(1, 0, False), (2, 0, True), (3, 0, True)]
    assert run.restored['start'] == (1, 0, 0)
    assert run.restored['scheduler_iter'] == 0
    assert run.restored['center']['n_upd'] == 0
    assert optimizer_steps(run.load('train_state_3.pt')['optimizer']) == [18]
    assert not equal_tensors(run.load('model_1.pt'), run.load('model_2.pt'))
    log = run.log()
    for text in ('<== Passed Arguments ==>', 'fresh start at epoch 1',
                 'head class-list check: sidecar (40 classes, 120 rows)',
                 'epoch iteration number: 6', 'start_epoch: 1',
                 'training process ends: finished'):
        assert text in log, text


def test_run_directory_files(reference):
    exp = reference.exp
    names = sorted(os.listdir(exp))
    assert names[:3] == ['config.yaml', 'models', 'recipe.yaml']
    assert names[3].startswith('run_meta_') and names[4] == 'train.log'
    recipe = load_config(os.path.join(exp, 'recipe.yaml'))  # reads back
    assert recipe.derived.epoch_iter == STEPS
    assert recipe.derived.num_class == 40
    assert recipe.data.sampler.arm_order[0] == 'vox2'
    assert len(recipe.data.sampler.arm_order) == 8
    with open(os.path.join(exp, 'config.yaml')) as fin:
        core = yaml.safe_load(fin)
    assert core['model'] == 'SSL_BACKEND_MHFA_LayerNorm'
    assert core['dataset_args']['frontend_num_layers'] == 2
    assert 'time_mask' not in core['dataset_args']['w2vbert_sv_args']


def test_existing_state_needs_resume(reference, monkeypatch, capsys):
    again = Run(reference.sets, monkeypatch=monkeypatch)
    assert again.rc == 1 and again.data == []
    assert 'pass --resume' in capsys.readouterr().err
    done = Run(reference.sets, '--resume', monkeypatch=monkeypatch)
    assert done.rc == 0 and done.steps == []
    assert 'all 3 epochs are done, nothing to do' in done.log()


def test_world_size_must_match_the_config(tiny_assets, tmp_path, monkeypatch,
                                          capsys):
    sets = make_run(tmp_path, tiny_assets, 2)  # run.gpus: [0, 1]
    run = Run(sets, monkeypatch=monkeypatch)   # one process
    assert run.rc == 2
    assert '1 process(es) are running' in capsys.readouterr().err
    assert train_mod.main(['--config', 'conf/missing.yaml']) == 2


def test_plan(reference, tiny_assets, tmp_path, capsys):
    def plan(sets):
        assert train_mod.main(argv_of(sets, '--plan')) == 0
        out = capsys.readouterr().out
        return dict(shlex.split(line)[0].split('=', 1)
                    for line in out.splitlines())

    done = plan(reference.sets)
    assert done['PLAN_EPOCH_DONE'] == '3' and done['PLAN_NEXT_EPOCH'] == '4'
    assert done['PLAN_RESUME_KIND'] == 'epoch'
    assert done['PLAN_NUM_EPOCHS'] == '3' and done['PLAN_WORLD_SIZE'] == '1'
    assert done['PLAN_GLOBAL_BATCH'] == done['PLAN_BATCH_SIZE'] == '8'
    assert done['PLAN_SAVE_OPTIMIZER_STATE'] == '1'
    assert done['PLAN_EXP_DIR'] == reference.exp
    assert done['PLAN_NOISE_LMDB'] == 'data/musan/lmdb'
    new = plan(make_run(tmp_path, tiny_assets, 2))
    assert (new['PLAN_RESUME_KIND'], new['PLAN_EPOCH_DONE'],
            new['PLAN_NEXT_EPOCH'], new['PLAN_NEXT_STEP'],
            new['PLAN_GLOBAL_BATCH']) == ('fresh', '0', '1', '0', '16')
    assert new['PLAN_INIT_CHECKPOINT'].endswith('init.pt')


# ------------------------------------------- (ii), (vi): epoch boundaries
def same_as_reference(run, reference, epochs=(1, 2, 3)):
    """Every saved tensor of the run equals the uninterrupted run's."""
    for n in epochs:
        assert equal_tensors(run.load('model_%d.pt' % n),
                             reference.load('model_%d.pt' % n)), n
    mine, ref = run.load('train_state_3.pt'), reference.load(
        'train_state_3.pt')
    assert equal_optimizer(mine['optimizer'], ref['optimizer'])
    assert mine['scheduler'] == ref['scheduler']
    assert mine['margin_scheduler'] == ref['margin_scheduler']
    a, b = run.load('aux_state_3.pt'), reference.load('aux_state_3.pt')
    assert equal_center(a['center'][0], b['center'][0])


@pytest.mark.parametrize('per_run', [1, 2])
def test_segments_equal_the_uninterrupted_run(reference, tiny_assets,
                                              tmp_path, monkeypatch, per_run):
    """(vi) --epochs_this_run N, resumed until done == one long run."""
    sets = make_run(tmp_path, tiny_assets)
    runs = []
    while not os.path.exists(os.path.join(tmp_path, 'exp', 'models',
                                          'final_model.pt')):
        runs.append(Run(sets, '--resume', '--epochs_this_run', str(per_run),
                        monkeypatch=monkeypatch))
        assert runs[-1].rc == 0
        assert len(runs) <= 3
    assert len(runs) == (3 if per_run == 1 else 2)
    # every process ran whole epochs only and no step twice
    assert [len(r.steps) // STEPS for r in runs] == \
        ([1, 1, 1] if per_run == 1 else [2, 1])
    assert sum((r.steps for r in runs), []) == reference.steps
    # data: the same batches (the DataLoader base seed is part of the seed)
    assert sum((r.seen for r in runs), []) == reference.seen
    assert [c for r in runs for c in r.calls] == reference.calls
    same_as_reference(runs[-1], reference)
    # what the second process restored is what the first one ended with
    second = runs[1].restored
    assert second['start'] == (per_run + 1, 0, 0)
    assert second['scheduler_iter'] == per_run * STEPS
    assert optimizer_steps(second['optimizer']) == [per_run * STEPS]
    assert equal_tensors(second['weights'],
                         reference.load('model_%d.pt' % per_run))
    assert second['center']['n_upd'] > 0
    assert 'resume after epoch %d' % per_run in runs[-1].log()
    assert 'epoch(s) of this process done: stopping after epoch' in \
        runs[-1].log()


@pytest.mark.parametrize('signame', ['SIGUSR1', 'SIGTERM'])
def test_signal_on_the_last_step_of_an_epoch(reference, tiny_assets, tmp_path,
                                             monkeypatch, signame):
    """(ii) interrupted at an epoch boundary + resume == uninterrupted."""
    sets = make_run(tmp_path, tiny_assets)
    first = Run(sets, signals='1:0:%d:0:%s' % (STEPS - 1, signame),
                monkeypatch=monkeypatch)
    assert first.rc == 0 and len(first.steps) == STEPS
    assert first.files() == ['aux_state_1.pt', 'head_spk.json', 'model_1.pt',
                             'train_state_1.pt']  # no partial state
    assert 'stop request: stopping after epoch 1 of 3' in first.log()
    second = Run(sets, '--resume', monkeypatch=monkeypatch)
    assert second.rc == 0
    assert second.restored['start'] == (2, 0, 0)
    assert first.steps + second.steps == reference.steps
    assert first.seen + second.seen == reference.seen
    same_as_reference(second, reference)
    assert 'final_model.pt' in second.files()


# ------------------------------------------------ (i), (v): inside an epoch
def interrupted_run(tiny_assets, tmp_path, monkeypatch, signame):
    """Stop while batch index 2 of epoch 2 is handed out, then resume."""
    sets = make_run(tmp_path, tiny_assets)
    first = Run(sets, signals='2:0:2:0:%s' % signame, monkeypatch=monkeypatch)
    partial = dict(state=first.load('partial_state.pt'),
                   weights=first.load('partial_model.pt'),
                   files=first.files())
    second = Run(sets, '--resume', monkeypatch=monkeypatch)
    return first, partial, second


@pytest.fixture(scope='module')
def interrupted(tiny_assets, tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        return interrupted_run(tiny_assets, tmp_path_factory.mktemp('usr1'),
                               patch, 'SIGUSR1')


def test_signal_inside_an_epoch_saves_and_resumes(reference, interrupted):
    """(i) SIGUSR1 in the middle of epoch 2, then --resume."""
    first, partial, second = interrupted
    state = partial['state']
    # --- the interrupted process
    assert first.rc == 0
    assert [s[0] for s in first.steps] == list(range(STEPS + 3))
    assert partial['files'] == [
        'aux_state_1.pt', 'head_spk.json', 'model_1.pt', 'partial_model.pt',
        'partial_state.pt', 'train_state_1.pt']
    assert (state['epoch'], state['step'], state['stream'], state['world'],
            state['epoch_iter']) == (2, 3, 0, 1, STEPS)
    assert optimizer_steps(state['optimizer']) == [STEPS + 3]
    assert 'stop request: epoch 2 interrupted after step 3 of 6' in first.log()
    # --- what the resumed process restored: epoch, step, stream, optimiser,
    # centre, position of the schedule, weights
    got = second.restored
    assert second.rc == 0
    assert got['start'] == (2, 3, 1)
    assert got['scheduler_iter'] == STEPS + 3
    assert equal_optimizer(got['optimizer'], state['optimizer'])
    assert equal_center(got['center'], state['center'][0])
    assert got['center']['n_upd'] > 0
    assert equal_tensors(got['weights'], partial['weights'])
    assert 'resume inside epoch 2 at step 3, stream 1' in second.log()
    # --- it completes; the partial files are gone
    assert second.files() == [
        'aux_state_3.pt', 'final_model.pt', 'head_spk.json', 'model_1.pt',
        'model_2.pt', 'model_3.pt', 'train_state_3.pt']
    assert second.load('aux_state_3.pt')['stream'] == 0

    # === EQUAL to the uninterrupted run
    both = first.steps + second.steps
    # every global step ran exactly once with the same rates and margin
    assert both == reference.steps
    assert second.steps[0][0] == STEPS + 3
    final, ref = second.load('train_state_3.pt'), reference.load(
        'train_state_3.pt')
    assert final['scheduler'] == ref['scheduler']
    assert final['scheduler']['current_iter'] == EPOCHS * STEPS
    assert final['margin_scheduler'] == ref['margin_scheduler']
    assert optimizer_steps(final['optimizer']) == [EPOCHS * STEPS]
    assert final['optimizer']['param_groups'] == \
        ref['optimizer']['param_groups']  # the rates in the optimiser too
    # everything before the interruption
    assert equal_tensors(first.load('model_1.pt'),
                         reference.load('model_1.pt'))
    assert first.seen[:STEPS + 3] == reference.seen[:STEPS + 3]
    # the batches of the epoch after the interrupted one: the generators
    # and the DataLoader base seed at the start of epoch 3 are those of the
    # uninterrupted run
    assert second.seen[3:] == reference.seen[2 * STEPS:]
    assert second.calls == [(2, 1, True), (3, 0, True)]

    # === NOT EQUAL, by design: the rest of the interrupted epoch is drawn
    # from a fresh stream, so the weights differ from there on
    rest = second.seen[:3]
    assert [x[:3] for x in rest] == [(2, 1, 0), (2, 1, 1), (2, 1, 2)]
    ref_rest = reference.seen[STEPS + 3:2 * STEPS]
    assert [x[:3] for x in ref_rest] == [(2, 0, 3), (2, 0, 4), (2, 0, 5)]
    assert {x[3] for x in rest}.isdisjoint(x[3] for x in reference.seen)
    for n in (2, 3):
        assert not equal_tensors(second.load('model_%d.pt' % n),
                                 reference.load('model_%d.pt' % n))
    assert not equal_optimizer(final['optimizer'], ref['optimizer'])


def test_sigterm_is_the_same_path_as_sigusr1(interrupted, tiny_assets,
                                             tmp_path, monkeypatch):
    """(v) SIGTERM: the same states, bit for bit, as SIGUSR1."""
    first, partial, second = interrupted_run(tiny_assets, tmp_path,
                                             monkeypatch, 'SIGTERM')
    usr_first, usr_partial, usr_second = interrupted
    assert first.rc == second.rc == 0
    assert partial['files'] == usr_partial['files']
    assert equal_tensors(partial['weights'], usr_partial['weights'])
    assert equal_optimizer(partial['state']['optimizer'],
                           usr_partial['state']['optimizer'])
    assert equal_center(partial['state']['center'][0],
                        usr_partial['state']['center'][0])
    assert first.steps == usr_first.steps and second.steps == usr_second.steps
    assert second.seen == usr_second.seen
    for n in (1, 2, 3):
        assert equal_tensors(second.load('model_%d.pt' % n),
                             usr_second.load('model_%d.pt' % n))


def test_two_interruptions_of_one_epoch(reference, tiny_assets, tmp_path,
                                        monkeypatch):
    """Every restart of an epoch gets the next stream id."""
    sets = make_run(tmp_path, tiny_assets)
    signals = '2:0:2:0:SIGUSR1,2:1:1:0:SIGTERM'
    first = Run(sets, signals=signals, monkeypatch=monkeypatch)
    second = Run(sets, '--resume', signals=signals, monkeypatch=monkeypatch)
    state = second.load('partial_state.pt')
    assert (state['epoch'], state['step'], state['stream']) == (2, 5, 1)
    third = Run(sets, '--resume', signals=signals, monkeypatch=monkeypatch)
    assert first.rc == second.rc == third.rc == 0
    assert third.restored['start'] == (2, 5, 2)
    assert third.calls == [(2, 2, True), (3, 0, True)]
    assert [x[:3] for x in third.seen[:1]] == [(2, 2, 0)]
    assert first.steps + second.steps + third.steps == reference.steps
    assert third.seen[1:] == reference.seen[2 * STEPS:]
    assert 'final_model.pt' in third.files()


def test_changed_config_is_reported_on_resume(tiny_assets, tmp_path,
                                              monkeypatch):
    sets = make_run(tmp_path, tiny_assets)
    Run(sets, '--epochs_this_run', '1', monkeypatch=monkeypatch)
    run = Run(sets + ['optim.grad_clip=2.0'], '--resume', '--epochs_this_run',
              '1', monkeypatch=monkeypatch)
    assert run.rc == 0
    assert 'config changed since the run was started: optim.grad_clip: ' \
        '1.0 -> 2.0' in run.log()


# ----------------------------------------------- two processes (gloo, CPU)
def launch(sets, *flags, trace, signals='', world=2, timeout=600):
    """Start ``world`` training processes; returns their exit codes."""
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    os.makedirs(trace, exist_ok=True)
    procs = []
    for rank in range(world):
        env = dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank),
                   WORLD_SIZE=str(world), MASTER_ADDR='127.0.0.1',
                   MASTER_PORT=str(port), OMP_NUM_THREADS='2',
                   PYTHONDONTWRITEBYTECODE='1')
        env[SIGNAL_ENV], env[TRACE_ENV] = signals, trace
        out = open(os.path.join(trace, 'stdout_rank%d.txt' % rank), 'w')
        procs.append(subprocess.Popen(
            [sys.executable, DRIVER] + argv_of(sets, *flags), env=env,
            stdout=out, stderr=subprocess.STDOUT, cwd=RECIPE_DIR))
    try:
        return [p.wait(timeout=timeout) for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGKILL)


def read_trace(trace, rank):
    with open(os.path.join(trace, 'trace_rank%d.txt' % rank)) as fin:
        return [tuple(int(x) for x in line.split()) for line in fin]


def load_models(sets):
    exp = [s for s in sets if s.startswith('run.exp_dir=')][0][12:]
    return lambda name: torch.load(os.path.join(exp, 'models', name),
                                   weights_only=False)


@pytest.fixture(scope='module')
def ddp_reference(tiny_assets, tmp_path_factory):
    """Two processes, 2 epochs x 4 steps, uninterrupted."""
    tmp = tmp_path_factory.mktemp('ddp_reference')
    sets = make_run(tmp, tiny_assets, 2, 2, 4)
    trace = str(tmp / 'trace')
    assert launch(sets, trace=trace) == [0, 0]
    return sets, trace


def test_two_processes_stop_at_the_same_step_and_resume(
        ddp_reference, tiny_assets, tmp_path):
    """(iii), (i) on two processes: SIGUSR1 reaches rank 1 only."""
    ref_sets, ref_trace = ddp_reference
    sets = make_run(tmp_path, tiny_assets, 2, 2, 4)
    load = load_models(sets)
    one, two = str(tmp_path / 'trace1'), str(tmp_path / 'trace2')
    assert launch(sets, trace=one, signals='1:0:1:1:SIGUSR1') == [0, 0]
    # both processes ended after the step of batch index 1: they handed out
    # the same number of batches although only one of them got the signal
    assert [x[:3] for x in read_trace(one, 0)] == [(1, 0, 0), (1, 0, 1)]
    assert [x[:3] for x in read_trace(one, 1)] == [(1, 0, 0), (1, 0, 1)]
    state = load('partial_state.pt')
    assert (state['epoch'], state['step'], state['stream'], state['world'],
            state['epoch_iter']) == (1, 2, 0, 2, 4)
    centers = state['center']
    assert len(centers) == 2
    assert not equal_center(centers[0], centers[1])  # one state per process
    partial = load('partial_model.pt')

    assert launch(sets, '--resume', trace=two) == [0, 0]
    for rank in (0, 1):
        got = torch.load(os.path.join(two, 'restored_rank%d.pt' % rank),
                         weights_only=False)
        assert got['start'] == (1, 2, 1)
        assert got['scheduler_iter'] == 2
        assert got['optimizer_steps'] == [2]
        assert equal_center(got['center'], centers[rank]), rank
        mine, ref = read_trace(two, rank), read_trace(ref_trace, rank)
        # the rest of epoch 1 from stream 1, then epoch 2 as uninterrupted
        assert [x[:3] for x in mine[:2]] == [(1, 1, 0), (1, 1, 1)]
        assert mine[2:] == ref[4:] and len(mine) == 6
        assert read_trace(one, rank) == ref[:2]
    ref_load = load_models(ref_sets)
    final = load('train_state_2.pt')
    assert final['scheduler'] == ref_load('train_state_2.pt')['scheduler']
    assert optimizer_steps(final['optimizer']) == [8]
    assert len(load('aux_state_2.pt')['center']) == 2
    assert not equal_tensors(partial, load('model_2.pt'))
    assert not equal_tensors(load('model_2.pt'), ref_load('model_2.pt'))
    assert not os.path.exists(os.path.join(tmp_path, 'exp', 'models',
                                           'partial_state.pt'))


def test_two_processes_segments_equal_the_uninterrupted_run(
        ddp_reference, tiny_assets, tmp_path):
    """(ii), (vi) on two processes: one epoch per launch."""
    ref_sets, ref_trace = ddp_reference
    sets = make_run(tmp_path, tiny_assets, 2, 2, 4)
    one, two = str(tmp_path / 'trace1'), str(tmp_path / 'trace2')
    flags = ('--resume', '--epochs_this_run', '1')
    assert launch(sets, *flags, trace=one) == [0, 0]
    assert launch(sets, *flags, trace=two) == [0, 0]
    load, ref_load = load_models(sets), load_models(ref_sets)
    for name in ('model_1.pt', 'model_2.pt'):
        assert equal_tensors(load(name), ref_load(name)), name
    assert equal_optimizer(load('train_state_2.pt')['optimizer'],
                           ref_load('train_state_2.pt')['optimizer'])
    mine, ref = load('aux_state_2.pt'), ref_load('aux_state_2.pt')
    for rank in (0, 1):
        assert equal_center(mine['center'][rank], ref['center'][rank])
        assert read_trace(one, rank) + read_trace(two, rank) == \
            read_trace(ref_trace, rank)
        got = torch.load(os.path.join(two, 'restored_rank%d.pt' % rank),
                         weights_only=False)
        assert got['start'] == (2, 0, 0) and got['optimizer_steps'] == [4]


def test_torchrun_children_take_sigusr1_and_exit_cleanly(tiny_assets,
                                                         tmp_path):
    """The launcher's signal path against the installed torchrun.

    As local/slurm/train_segment.sbatch does it: SIGUSR1 is ignored when
    torchrun is started (so its children inherit "ignored" until they
    install their handler), and the signal is sent to the children of the
    torchrun process, never to torchrun itself.
    """
    sets = make_run(tmp_path, tiny_assets, 2, 2, 400)
    trace = str(tmp_path / 'trace')
    os.makedirs(trace)
    env = dict(os.environ, OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', SIGNAL_ENV):
        env.pop(name, None)
    env[TRACE_ENV], env[DELAY_ENV] = trace, '0.1'
    with open(os.path.join(trace, 'torchrun.txt'), 'w') as out:
        torchrun = subprocess.Popen(
            [sys.executable, '-m', 'torch.distributed.run', '--standalone',
             '--nnodes=1', '--nproc_per_node=2', DRIVER] + argv_of(sets),
            env=env, stdout=out, stderr=subprocess.STDOUT, cwd=RECIPE_DIR,
            preexec_fn=lambda: signal.signal(signal.SIGUSR1, signal.SIG_IGN))
    try:
        files = [os.path.join(trace, 'trace_rank%d.txt' % r) for r in (0, 1)]
        deadline = time.time() + 300
        while time.time() < deadline and torchrun.poll() is None and not all(
                os.path.exists(f) and len(open(f).readlines()) >= 3
                for f in files):
            time.sleep(0.2)
        assert torchrun.poll() is None, open(
            os.path.join(trace, 'torchrun.txt')).read()[-3000:]
        children = subprocess.run(
            ['pgrep', '-P', str(torchrun.pid)], stdout=subprocess.PIPE,
            text=True, check=True).stdout.split()
        assert len(children) == 2  # the two training processes, nothing else
        for pid in children:
            os.kill(int(pid), signal.SIGUSR1)
        assert torchrun.wait(timeout=120) == 0
    finally:
        if torchrun.poll() is None:
            torchrun.kill()
    state = load_models(sets)('partial_state.pt')
    done = state['step']
    assert state['epoch'] == 1 and state['world'] == 2 and 3 <= done < 400
    # both processes stopped after the same step
    assert [len(read_trace(trace, r)) for r in (0, 1)] == [done, done]
    with open(os.path.join(tmp_path, 'exp', 'train.log')) as fin:
        assert 'partial state saved' in fin.read()


# ------------------------------------------- the real pipeline, end to end
def test_pipeline_data_wiring(monkeypatch):
    """The seam over the built objects: the built loader for (1, 0), a
    loader with the generator of the pass otherwise."""
    calls = []

    class Dataset:
        def set_epoch(self, epoch, stream=0):
            calls.append(('set_epoch', epoch, stream))

    def build_loader(dataset, cfg, generator=None):
        calls.append(('build_loader', dataset, cfg, generator))
        return 'new loader'

    monkeypatch.setattr(train_mod.api, 'build_loader', build_loader)
    dataset = Dataset()
    objects = type('Objects', (), {'dataset': dataset, 'loader': 'built'})()
    data = train_mod.PipelineData('cfg', 0, 1, objects)
    assert data.loader(1, 0, None) == 'built'
    assert calls == [('set_epoch', 1, 0)]
    gen = torch.Generator()
    assert data.loader(2, 1, gen) == 'new loader'
    assert calls[1:] == [('set_epoch', 2, 1),
                         ('build_loader', dataset, 'cfg', gen)]


def write_toy_run(tmp_path, tiny_assets, **run):
    """The toy recipe of the pipeline tests as a config file with an
    initial checkpoint; returns (config path, batch size)."""
    import pipeline_helpers as ph
    from ovsb.config.schema import build_config
    from ovsb.model.build import build_model
    from ovsb.model.head_record import write_head_sidecar

    recipe = ph.make_toy_recipe(tmp_path / 'toy', tiny_assets)
    tree = copy.deepcopy(recipe.tree)
    with open(tree['data']['lists']['train_label']) as fin:
        speakers = sorted({line.split()[1] for line in fin if line.strip()})
    init = str(tmp_path / 'init.pt')
    tree['run'].update(num_epochs=2, log_batch_interval=2)
    tree['run'].update(run)
    tree['model']['init']['checkpoint'] = init
    tree['loss']['pair_auc']['center']['warmup'] = 2
    config = str(tmp_path / 'toy.yaml')
    with open(config, 'w') as fout:
        yaml.safe_dump(tree, fout, sort_keys=False)
    torch.manual_seed(11)  # not the seed of the run
    model = build_model(build_config(copy.deepcopy(tree)), len(speakers))
    torch.save(model.state_dict(), init)
    write_head_sidecar(init, speakers,
                       model.state_dict()['projection.weight'].shape[0])
    return config, tree['data']['loader']['batch_size']


def test_entry_point_builds_what_build_training_builds(tiny_assets,
                                                       tmp_path):
    """The entry point adds no draw to ``api.build_training``: the random
    initialisation, the worker base seed of the first DataLoader iterator
    and the first batch are those of a bare ``build_training`` call, so
    what the parity gate pins for ``build_training`` holds for a run."""
    config, _ = write_toy_run(tmp_path, tiny_assets, num_epochs=1)
    env = dict(os.environ, OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    env.pop('WORLD_SIZE', None)
    results = {}
    for mode in ('reference', 'entry'):
        out = str(tmp_path / (mode + '.json'))
        proc = subprocess.run(
            [sys.executable, CONSTRUCTION, mode, config, out], env=env,
            cwd=RECIPE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=600)
        assert proc.returncode == 0, proc.stdout[-3000:]
        with open(out) as fin:
            results[mode] = json.load(fin)
    ref, got = results['reference'], results['entry']
    assert got['rc'] == 0
    assert got['base_seed'] == ref['base_seed']
    assert got['first_keys'] == ref['first_keys'] and len(ref['first_keys'])
    assert list(got['weights']) == list(ref['weights'])
    assert got['weights'] == ref['weights']
    # the weights compared are the random initialisation, not the loaded
    # checkpoint (which was drawn from another seed)
    init = torch.load(str(tmp_path / 'init.pt'), weights_only=False)
    head = hashlib.sha256(init['projection.weight'].numpy().tobytes())
    assert got['weights']['projection.weight'] != head.hexdigest()


def test_real_pipeline_segments_signal_and_resume(tiny_assets, tmp_path):
    """``python -m ovsb.bin.train`` on the toy corpus: DataLoader workers,
    sampler, crops, augmentation. A segment and its resume; SIGUSR1 while
    the workers run, then the rest of the epoch from the next stream."""
    config, batch = write_toy_run(tmp_path, tiny_assets)
    env = dict(os.environ, OMP_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join(
                   [RECIPE_DIR, os.path.join(RECIPE_DIR, '..', '..', '..'),
                    os.environ.get('PYTHONPATH', '')]))
    env.pop('WORLD_SIZE', None)

    def command(exp, steps, *flags):
        return [sys.executable, '-m', 'ovsb.bin.train', '--config', config,
                '--set', 'run.exp_dir=%s' % exp, '--set',
                'data.sampler.sample_num_per_epoch=%d' % (steps * batch)] \
            + list(flags)

    def train(exp, steps, *flags):
        return subprocess.run(command(exp, steps, *flags), env=env,
                              cwd=RECIPE_DIR, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              timeout=600)

    def log_of(exp):
        with open(os.path.join(exp, 'train.log')) as fin:
            return fin.read()

    # --- one epoch per process
    exp = str(tmp_path / 'exp_segments')
    first = train(exp, 6, '--resume', '--epochs_this_run', '1')
    assert first.returncode == 0, first.stdout[-3000:]
    assert os.path.exists(os.path.join(exp, 'models', 'train_state_1.pt'))
    assert '|         1|         6|' in log_of(exp)
    second = train(exp, 6, '--resume', '--epochs_this_run', '1')
    assert second.returncode == 0, second.stdout[-3000:]
    assert os.path.islink(os.path.join(exp, 'models', 'final_model.pt'))
    assert 'resume after epoch 1' in log_of(exp)
    assert '|         2|         6|' in log_of(exp)

    # --- SIGUSR1 while the workers deliver batches
    exp = str(tmp_path / 'exp_signal')
    steps = 300
    proc = subprocess.Popen(command(exp, steps), env=env, cwd=RECIPE_DIR,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    log = os.path.join(exp, 'train.log')
    deadline = time.time() + 300
    while time.time() < deadline and proc.poll() is None and not (
            os.path.exists(log) and '|         1|         2|' in log_of(exp)):
        time.sleep(0.1)
    assert proc.poll() is None
    proc.send_signal(signal.SIGUSR1)
    out, _ = proc.communicate(timeout=300)
    assert proc.returncode == 0, out[-3000:]
    state = torch.load(os.path.join(exp, 'models', 'partial_state.pt'),
                       weights_only=False)
    done = state['step']
    assert state['epoch'] == 1 and state['stream'] == 0 and 2 <= done < steps
    assert 'interrupted after step %d of %d' % (done, steps) in log_of(exp)
    # no worker of the stopped process is left behind
    assert subprocess.run(['pgrep', '-P', str(proc.pid)],
                          stdout=subprocess.PIPE).stdout == b''

    resumed = train(exp, steps, '--resume', '--epochs_this_run', '1')
    assert resumed.returncode == 0, resumed.stdout[-3000:]
    text = log_of(exp)
    assert 'resume inside epoch 1 at step %d, stream 1' % done in text
    assert '|         1|%10d|' % steps in text
    assert sorted(os.listdir(os.path.join(exp, 'models'))) == [
        'aux_state_1.pt', 'head_spk.json', 'model_1.pt', 'train_state_1.pt']
