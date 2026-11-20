"""Checkpoints: initial weights, epoch and partial saves, resume point.

A three-layer module stands in for the model (the functions only need a
``projection.weight``); the real tiny model goes through the same code in
tests/test_train_entry.py.
"""
import fnmatch
import glob
import os
import re

import pytest
import torch
import torch.distributed as dist

from ovsb.engine import checkpoint as ckpt
from ovsb.engine.checkpoint import (EPOCH, FRESH, PARTIAL, PARTIAL_MODEL,
                                    PARTIAL_STATE, ResumeError,
                                    find_resume_point, gather_center_states,
                                    load_init, resume, save_epoch,
                                    save_partial)
from ovsb.losses.pair_auc import RunningCenter
from ovsb.model.head_record import (RUN_RECORD, HeadMismatchError,
                                    write_head_sidecar)

CPU = torch.device('cpu')
DIM, CLASSES, ROWS = 6, 5, 15
SPKS = ['s%d' % i for i in range(CLASSES)]
EPOCH_ITER = 10


class Net(torch.nn.Module):
    def __init__(self, rows=ROWS):
        super().__init__()
        self.frontend = torch.nn.Linear(4, DIM)
        self.pool = torch.nn.Linear(DIM, DIM)
        self.projection = torch.nn.Linear(DIM, rows, bias=False)

    def forward(self, x):
        return self.projection(self.pool(self.frontend(x)))


class Sched:
    """The state the core saves of a scheduler: its scalar attributes."""

    def __init__(self):
        self.current_iter = 0
        self.max_iter = 30


def make(seed=0, rows=ROWS):
    torch.manual_seed(seed)
    model = Net(rows)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    return model, optimizer, Sched(), Sched()


def train(model, optimizer, sched, margin, steps, start=0):
    for i in range(start, start + steps):
        sched.current_iter = margin.current_iter = i + 1
        optimizer.zero_grad()
        model(torch.randn(3, 4)).pow(2).sum().backward()
        optimizer.step()


def center_with(n_batches, seed=0, arm=1):
    gen = torch.Generator().manual_seed(seed)
    center = RunningCenter(arm, 0.9, 2)
    for _ in range(n_batches):
        center.update(torch.randn(4, DIM, generator=gen),
                      torch.tensor([arm, arm, 0, arm]))
    return center


def same_center(a, b):
    assert set(a) == set(b)
    for key in a:
        if key == 'acc':
            assert (a[key] is None) == (b[key] is None)
            assert a[key] is None or torch.equal(a[key], b[key])
        else:
            assert a[key] == b[key], key


def same_optimizer(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa['param_groups'] == sb['param_groups']
    assert list(sa['state']) == list(sb['state'])
    for key in sa['state']:
        for name, value in sa['state'][key].items():
            assert torch.equal(torch.as_tensor(value),
                               torch.as_tensor(sb['state'][key][name])), name


def same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


# ------------------------------------------------------------ centre states
@pytest.mark.parametrize('n_batches', [0, 1, 5])
def test_gather_single_process_is_the_state_dict(n_batches):
    center = center_with(n_batches)
    states = gather_center_states(center, DIM, CPU)
    assert len(states) == 1
    same_center(states[0], center.state_dict())
    fresh = RunningCenter(1, 0.9, 2)
    fresh.load_state_dict(states[0])
    assert fresh.describe() == center.describe()
    assert gather_center_states(None, DIM, CPU) is None


def test_gather_wrong_size_is_refused():
    with pytest.raises(ValueError, match='expected 7'):
        gather_center_states(center_with(1), DIM + 1, CPU)


def test_gather_collects_every_rank(monkeypatch):
    """Two processes: each contributes its row, both get both states."""
    centers = [center_with(4, seed=1), center_with(0)]
    tables = {}
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'get_world_size', lambda: 2)
    for rank in (1, 0, 1):  # the first pass of rank 1 only records its row
        monkeypatch.setattr(dist, 'get_rank', lambda rank=rank: rank)

        def all_reduce(table, rank=rank):
            tables[rank] = table.clone()
            if 1 - rank in tables:
                table += tables[1 - rank]

        monkeypatch.setattr(dist, 'all_reduce', all_reduce)
        states = gather_center_states(centers[rank], DIM, CPU)
        tables[rank][1 - rank] = 0  # keep the own row only
    assert len(states) == 2
    for rank in (0, 1):
        same_center(states[rank], centers[rank].state_dict())
    assert states[1]['acc'] is None and states[0]['n_upd'] == 4


# -------------------------------------------------------------- epoch saves
def test_epoch_save_files_and_marker_order(tmp_path, monkeypatch):
    model, optimizer, sched, margin = make()
    train(model, optimizer, sched, margin, 3)
    order = []
    real = ckpt.save_epoch_artifacts

    def artifacts(model_dir, epoch, *args, **kwargs):
        order.append(sorted(os.listdir(model_dir)))
        return real(model_dir, epoch, *args, **kwargs)

    monkeypatch.setattr(ckpt, 'save_epoch_artifacts', artifacts)
    states = gather_center_states(center_with(3), DIM, CPU)
    d = str(tmp_path)
    save_epoch(d, 1, model, optimizer, sched, margin, states, 1)
    assert order == [['aux_state_1.pt']]  # before the model and the marker
    assert sorted(os.listdir(d)) == ['aux_state_1.pt', 'model_1.pt',
                                     'train_state_1.pt']
    save_epoch(d, 2, model, optimizer, sched, margin, states, 1, stream=1)
    assert sorted(os.listdir(d)) == ['aux_state_2.pt', 'model_1.pt',
                                     'model_2.pt', 'train_state_2.pt']
    aux = torch.load(os.path.join(d, 'aux_state_2.pt'), weights_only=False)
    assert (aux['epoch'], aux['stream'], aux['world']) == (2, 1, 1)
    point = find_resume_point(d)
    assert (point.kind, point.epoch, point.start_epoch, point.start_step,
            point.start_stream) == (EPOCH, 2, 3, 0, 0)
    assert point.weights.endswith('model_2.pt')


def test_epoch_resume_restores_everything(tmp_path):
    model, optimizer, sched, margin = make()
    train(model, optimizer, sched, margin, EPOCH_ITER)
    center = center_with(5, seed=3)
    d = str(tmp_path)
    write_head_sidecar(os.path.join(d, 'init.pt'), SPKS, ROWS)
    os.replace(os.path.join(d, 'init.pt.head.json'),
               os.path.join(d, RUN_RECORD))
    save_epoch(d, 1, model, optimizer, sched, margin,
               gather_center_states(center, DIM, CPU), 1)

    model2, optimizer2, sched2, margin2 = make(seed=9)
    center2 = RunningCenter(1, 0.9, 2)
    point = find_resume_point(d)
    assert load_init(model2, point.weights, SPKS, d, write=False) == 'run'
    resume(point, d, optimizer2, sched2, margin2, center2, 0, 1, EPOCH_ITER)
    same_weights(model, model2)
    same_optimizer(optimizer, optimizer2)
    assert sched2.current_iter == margin2.current_iter == EPOCH_ITER
    same_center(center.state_dict(), center2.state_dict())
    # both continue identically
    torch.manual_seed(1)
    train(model, optimizer, sched, margin, 2, EPOCH_ITER)
    torch.manual_seed(1)
    train(model2, optimizer2, sched2, margin2, 2, EPOCH_ITER)
    same_weights(model, model2)


def test_crash_between_model_and_marker_keeps_the_previous_epoch(
        tmp_path, monkeypatch):
    model, optimizer, sched, margin = make()
    d = str(tmp_path)
    states = gather_center_states(center_with(2), DIM, CPU)
    save_epoch(d, 1, model, optimizer, sched, margin, states, 1)
    train(model, optimizer, sched, margin, 2)

    def killed(model_dir, epoch, model, *args, **kwargs):
        ckpt.save_checkpoint(model, ckpt.model_path(model_dir, epoch))
        raise KeyboardInterrupt('killed before train_state')

    monkeypatch.setattr(ckpt, 'save_epoch_artifacts', killed)
    with pytest.raises(KeyboardInterrupt):
        save_epoch(d, 2, model, optimizer, sched, margin, states, 1)
    assert os.path.exists(os.path.join(d, 'model_2.pt'))
    point = find_resume_point(d)
    assert (point.kind, point.epoch) == (EPOCH, 1)
    assert os.path.exists(ckpt.aux_state_path(d, 1))  # still restorable
    model2, optimizer2, sched2, margin2 = make(seed=4)
    resume(point, d, optimizer2, sched2, margin2, RunningCenter(1, 0.9, 2),
           0, 1, EPOCH_ITER)
    # without a marker file (save_optimizer_state off) model_N.pt decides
    assert find_resume_point(d, save_optimizer_state=False).epoch == 2


def test_epoch_resume_checks_world_and_aux(tmp_path):
    model, optimizer, sched, margin = make()
    d = str(tmp_path)
    states = gather_center_states(center_with(2), DIM, CPU)
    save_epoch(d, 1, model, optimizer, sched, margin, states, 1)
    point = find_resume_point(d)
    with pytest.raises(ResumeError, match='1 process.*this run has 2'):
        resume(point, d, optimizer, sched, margin, RunningCenter(1, 0.9, 2),
               1, 2, EPOCH_ITER)
    os.remove(ckpt.aux_state_path(d, 1))
    with pytest.raises(ResumeError, match='aux_state_1.pt is missing'):
        resume(point, d, optimizer, sched, margin, RunningCenter(1, 0.9, 2),
               0, 1, EPOCH_ITER)


# ------------------------------------------------------------ partial saves
def partial(d, model, optimizer, center, epoch=2, step=4, stream=0,
            position=0):
    sched = Sched()
    sched.current_iter = position
    save_partial(d, epoch, step, stream, model, optimizer, sched, Sched(),
                 gather_center_states(center, DIM, CPU), 1, EPOCH_ITER)


def test_partial_names_are_not_epoch_checkpoints(tmp_path):
    model, optimizer, sched, margin = make()
    d = str(tmp_path)
    partial(d, model, optimizer, center_with(1))
    names = sorted(os.listdir(d))
    assert names == [PARTIAL_MODEL, PARTIAL_STATE]
    assert glob.glob(os.path.join(d, 'model_[0-9]*.pt')) == []
    assert glob.glob(os.path.join(d, 'model_*.pt')) == []
    for name in names + [PARTIAL_MODEL + ckpt.STAGING,
                         PARTIAL_STATE + ckpt.STAGING,
                         PARTIAL_MODEL + ckpt.STAGING + '.tmp']:
        assert not fnmatch.fnmatch(name, 'model_[0-9]*.pt')
        assert not re.fullmatch(r'model_\d+\.pt', name)  # average_model.py
    assert ckpt.saved_epochs(d) == []


def test_partial_resume_restores_everything(tmp_path):
    model, optimizer, sched, margin = make()
    train(model, optimizer, sched, margin, EPOCH_ITER + 4)
    center = center_with(6, seed=2)
    d = str(tmp_path)
    write_head_sidecar(os.path.join(d, 'x.pt'), SPKS, ROWS)
    os.replace(os.path.join(d, 'x.pt.head.json'), os.path.join(d, RUN_RECORD))
    partial(d, model, optimizer, center, epoch=2, step=4, stream=1,
            position=EPOCH_ITER + 4)
    point = find_resume_point(d)
    assert (point.kind, point.epoch, point.step, point.stream) == \
        (PARTIAL, 2, 4, 1)
    assert (point.start_epoch, point.start_step, point.start_stream) == \
        (2, 4, 2)  # same epoch, same step, the next stream
    assert 'epoch 2 at step 4, stream 2' in point.describe()

    model2, optimizer2, sched2, margin2 = make(seed=9)
    center2 = RunningCenter(1, 0.9, 2)
    load_init(model2, point.weights, SPKS, d, write=False)
    resume(point, d, optimizer2, sched2, margin2, center2, 0, 1, EPOCH_ITER)
    same_weights(model, model2)
    same_optimizer(optimizer, optimizer2)
    # each scheduler gets back its own counter (a constant margin never
    # advances its counter)
    assert (sched2.current_iter, margin2.current_iter) == (EPOCH_ITER + 4, 0)
    same_center(center.state_dict(), center2.state_dict())
    with pytest.raises(ResumeError, match='10 steps per epoch.*has 12'):
        resume(point, d, optimizer2, sched2, margin2, center2, 0, 1, 12)
    with pytest.raises(ResumeError, match='this run has 4'):
        resume(point, d, optimizer2, sched2, margin2, center2, 0, 4,
               EPOCH_ITER)


def test_kill_between_the_two_renames_leaves_the_epoch_state(
        tmp_path, monkeypatch):
    """Old pair (epoch 2, step 3), new save killed after the first rename:
    no mixture of new weights and old state is ever offered."""
    model, optimizer, sched, margin = make()
    d = str(tmp_path)
    states = gather_center_states(center_with(2), DIM, CPU)
    save_epoch(d, 1, model, optimizer, sched, margin, states, 1)
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    train(model, optimizer, sched, margin, 3)
    partial(d, model, optimizer, center_with(3), step=3)
    assert find_resume_point(d).kind == PARTIAL
    train(model, optimizer, sched, margin, 2)

    real, calls = ckpt._commit, []

    def commit(staged, final):
        calls.append(os.path.basename(final))
        if len(calls) == 2:
            raise KeyboardInterrupt('killed between the renames')
        real(staged, final)

    monkeypatch.setattr(ckpt, '_commit', commit)
    with pytest.raises(KeyboardInterrupt):
        partial(d, model, optimizer, center_with(4), step=5)
    assert calls == [PARTIAL_MODEL, PARTIAL_STATE]  # the state is last
    assert not os.path.exists(os.path.join(d, PARTIAL_STATE))
    point = find_resume_point(d)
    assert (point.kind, point.epoch) == (EPOCH, 1)
    model2, optimizer2, sched2, margin2 = make(seed=3)
    ckpt.load_checkpoint(model2, point.weights)
    resume(point, d, optimizer2, sched2, margin2, RunningCenter(1, 0.9, 2),
           0, 1, EPOCH_ITER)
    assert all(torch.equal(v, saved[k])
               for k, v in model2.state_dict().items())
    # the next save works over the left-over staging file
    monkeypatch.setattr(ckpt, '_commit', real)
    partial(d, model, optimizer, center_with(4), step=5)
    assert find_resume_point(d).step == 5


def test_kill_while_writing_keeps_the_old_partial(tmp_path, monkeypatch):
    model, optimizer, sched, margin = make()
    d = str(tmp_path)
    train(model, optimizer, sched, margin, 3)
    center = center_with(3)
    partial(d, model, optimizer, center, step=3)
    kept = {k: v.clone() for k, v in model.state_dict().items()}
    train(model, optimizer, sched, margin, 2)

    def killed(obj, path):
        with open(path + '.tmp', 'wb') as fout:
            fout.write(b'truncated')
        raise KeyboardInterrupt('killed while writing the state')

    monkeypatch.setattr(ckpt, 'atomic_save', killed)
    with pytest.raises(KeyboardInterrupt):
        partial(d, model, optimizer, center_with(4), step=5)
    point = find_resume_point(d)
    assert (point.kind, point.step) == (PARTIAL, 3)
    model2 = Net()
    ckpt.load_checkpoint(model2, point.weights)
    assert all(torch.equal(v, kept[k]) for k, v in model2.state_dict().items())


def test_epoch_save_removes_the_partial_and_stale_partial_is_ignored(
        tmp_path, monkeypatch):
    model, optimizer, sched, margin = make()
    d = str(tmp_path)
    center = center_with(2)
    states = gather_center_states(center, DIM, CPU)
    partial(d, model, optimizer, center, epoch=1, step=4)
    save_epoch(d, 1, model, optimizer, sched, margin, states, 1, stream=1)
    assert not os.path.exists(os.path.join(d, PARTIAL_STATE))
    assert not os.path.exists(os.path.join(d, PARTIAL_MODEL))
    # a partial of an epoch that is complete (the process died between the
    # epoch marker and the removal) is not a resume point
    monkeypatch.setattr(ckpt, 'remove_partial', lambda model_dir: None)
    partial(d, model, optimizer, center, epoch=2, step=4)
    save_epoch(d, 2, model, optimizer, sched, margin, states, 1)
    assert os.path.exists(os.path.join(d, PARTIAL_STATE))
    point = find_resume_point(d)
    assert (point.kind, point.epoch) == (EPOCH, 2)
    # a partial of a later epoch wins over the complete one
    partial(d, model, optimizer, center, epoch=3, step=1)
    assert find_resume_point(d).kind == PARTIAL


def test_fresh_point(tmp_path):
    point = find_resume_point(str(tmp_path / 'missing'))
    assert (point.kind, point.start_epoch, point.start_step,
            point.start_stream, point.weights) == (FRESH, 1, 0, 0, None)
    resume(point, str(tmp_path), None, None, None, None, 0, 1, EPOCH_ITER)


# ----------------------------------------------------------- initial weights
def test_load_init_checks_record_and_shape(tmp_path):
    d = str(tmp_path / 'models')
    os.makedirs(d)
    init = str(tmp_path / 'init.pt')
    source, _, _, _ = make(seed=2)
    torch.save(source.state_dict(), init)
    model, _, _, _ = make(seed=3)
    with pytest.raises(HeadMismatchError, match='neither'):
        load_init(model, init, SPKS, d)           # no head record at all
    write_head_sidecar(init, SPKS, ROWS)
    assert load_init(model, init, SPKS, d, write=True) == 'sidecar'
    same_weights(model, source)
    assert os.path.exists(os.path.join(d, RUN_RECORD))
    with pytest.raises(HeadMismatchError, match='another class list'):
        load_init(model, init, SPKS[:-1] + ['other'], d)

    # a larger head: the core loader would keep its first rows
    big, _, _, _ = make(rows=ROWS + 3)
    big_path = str(tmp_path / 'big.pt')
    torch.save(big.state_dict(), big_path)
    write_head_sidecar(big_path, SPKS, ROWS)      # a record that lies
    before = model.projection.weight.detach().clone()
    with pytest.raises(HeadMismatchError, match=r'shape \(18, 6\).*\(15, 6\)'):
        load_init(model, big_path, SPKS, d)
    assert torch.equal(model.projection.weight, before)  # nothing loaded

    headless = {k: v for k, v in source.state_dict().items()
                if not k.startswith('projection.')}
    path = str(tmp_path / 'headless.pt')
    torch.save({'state_dict': headless}, path)
    write_head_sidecar(path, SPKS, ROWS)
    with pytest.raises(HeadMismatchError, match='no projection.weight'):
        load_init(model, path, SPKS, d)
