"""Helpers of the loop, checkpoint and entry-point tests.

``FakeData`` stands in for the data pipeline behind the loop's data seam.
Its batches are a pure function of ``(run seed, rank, epoch, stream, index
in the pass, base seed)``. The base seed is drawn the way a DataLoader
draws its worker base seed: one int64 from the generator it is given, or
from the torch CPU generator when that is None. Batches therefore differ
when the process's generators or the stream id differ, and are equal when
both are equal, which is what the resume tests compare.

Environment (read at construction, so subprocesses can be steered):

  OVSB_TEST_SIGNAL   "epoch:stream:index:rank:SIGNAME[,...]": the process
                     of that rank sends itself the signal while it hands
                     out that batch (the step of the batch still runs)
  OVSB_TEST_TRACE    directory: every batch handed out is appended to
                     ``trace_rank<r>.txt`` as "epoch stream index seed"
  OVSB_TEST_DELAY    seconds to wait before every batch
"""
import os
import random
import signal
import time
from types import SimpleNamespace

import numpy as np
import torch

from conftest import RECIPE_DIR
from test_build import TINY_CLASSES, TINY_LAYERS, TINY_MODEL, tiny_config
from test_step import make_batch

from ovsb import api
from ovsb.config import schema
from ovsb.model.build import build_model
from ovsb.model.head_record import write_head_sidecar
from wespeaker.utils.utils import set_seed

SIGNAL_ENV, TRACE_ENV = 'OVSB_TEST_SIGNAL', 'OVSB_TEST_TRACE'
DELAY_ENV = 'OVSB_TEST_DELAY'
ROWS = 8
SPEAKERS = ['spk%03d' % i for i in range(TINY_CLASSES)]


def fake_batch(seed, arm_order):
    """A collated batch of 8 rows: a plain or (30 %) a pair batch."""
    rng = random.Random(seed)
    arms = [rng.choice(arm_order) for _ in range(4)]
    if rng.random() < 0.5:
        arms[0] = 'vox2'  # rows of the centre arm
    if rng.random() < 0.3:
        spk = rng.sample(range(TINY_CLASSES), 2)
        return make_batch(
            seed, arm_order,
            [n for _ in range(4) for n in (16000, rng.randrange(50, 76) * 160)],
            [spk[0]] * 4 + [spk[1]] * 4, [arms[0]] * 4 + [arms[1]] * 4,
            roles=[1, 2] * 4)
    spk = rng.sample(range(TINY_CLASSES), 4)
    return make_batch(
        seed, arm_order, [rng.randrange(60, 101) * 160 for _ in range(ROWS)],
        [s for s in spk for _ in range(2)], [a for a in arms for _ in range(2)])


class FakeData:
    """In-memory ``TrainData`` (module docstring).

    Attributes:
        calls: ``(epoch, stream, generator given)`` per ``loader`` call.
        seen: ``(epoch, stream, index, batch seed)`` per batch handed out.
    """

    def __init__(self, cfg, rank=0, world=1, objects=None):
        self.cfg, self.rank = cfg, rank
        self.calls, self.seen = [], []
        self.signals = {}
        for item in filter(None, os.environ.get(SIGNAL_ENV, '').split(',')):
            epoch, stream, index, who, name = item.split(':')
            if int(who) == rank:
                self.signals[(int(epoch), int(stream), int(index))] = \
                    getattr(signal, name)
        self.trace = os.environ.get(TRACE_ENV)
        self.delay = float(os.environ.get(DELAY_ENV) or 0)

    def loader(self, epoch, stream, generator):
        self.calls.append((epoch, stream, generator is not None))
        base = int(torch.empty((), dtype=torch.int64).random_(
            generator=generator).item())
        return self._batches(epoch, stream, base)

    def _batches(self, epoch, stream, base):
        order = self.cfg.data.sampler.arm_order
        index = 0
        while True:  # endless, as the sampler streams are
            seed = random.Random(repr((self.cfg.run.seed, self.rank, epoch,
                                       stream, index, base))).getrandbits(31)
            time.sleep(self.delay)
            self.seen.append((epoch, stream, index, seed))
            if self.trace:
                with open(os.path.join(self.trace, 'trace_rank%d.txt' %
                                       self.rank), 'a') as fout:
                    fout.write('%d %d %d %d\n' % (epoch, stream, index, seed))
            if (epoch, stream, index) in self.signals:
                os.kill(os.getpid(), self.signals[(epoch, stream, index)])
            yield fake_batch(seed, order)
            index += 1

    def stat_lines(self):
        return ['fake data: %d batches handed out' % len(self.seen)]


def fake_training(cfg, rank, world, device):
    """``api.build_training`` without the data side: same order (seeding,
    model on the device, optimiser, schedulers, objective), a table that
    only names the 40 classes, no dataset and no loader."""
    set_seed(cfg.run.seed + rank)
    table = SimpleNamespace(class_names=np.array(SPEAKERS),
                            n_class=TINY_CLASSES)
    model = build_model(cfg, TINY_CLASSES)
    model.to(device)
    optimizer = api.build_optimizer(model, cfg)
    scheduler, margin = api.build_schedulers(optimizer, model, cfg)
    return api.TrainingObjects(table, None, None, None, model,
                               api.build_losses(cfg), optimizer, scheduler,
                               margin)


def make_run(tmp_path, assets, world=1, epochs=3, steps=6, *overrides):
    """Files of a tiny run under ``tmp_path``.

    Writes the training list (40 speakers) and an initial checkpoint with
    its head record. Returns the ``--set`` overrides that turn
    ``conf/stage2_v9c.yaml`` into the tiny run: ``steps`` steps per epoch
    on ``world`` process(es), log every 2 steps, centre warm-up 2.
    """
    tmp_path = str(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)
    label = os.path.join(tmp_path, 'utt2spk')
    with open(label, 'w') as fout:
        for i, spk in enumerate(SPEAKERS):
            fout.write('utt%04d %s\n' % (i, spk))
    sets = [
        'run.exp_dir=%s' % os.path.join(tmp_path, 'exp'),
        'run.gpus=%s' % list(range(world)), 'run.num_epochs=%d' % epochs,
        'run.log_batch_interval=2', 'data.lists.train_label=%s' % label,
        'data.sampler.P=4', 'data.loader.batch_size=%d' % ROWS,
        'data.sampler.sample_num_per_epoch=%d' % (steps * world * ROWS),
        'loss.pair_auc.center.warmup=2', 'optim.warm_up_epoch=1.0',
        'optim.initial_lr=0.004', 'optim.final_lr=0.0004',
        'model.init.checkpoint=%s' % os.path.join(tmp_path, 'init.pt'),
    ] + list(overrides)
    cfg = tiny_config(assets, *sets)
    assert schema.epoch_iter(cfg) == steps
    init = cfg.model.init.checkpoint
    if not os.path.exists(init):
        torch.manual_seed(5)
        model = build_model(cfg, TINY_CLASSES)
        torch.save(model.state_dict(), init)
        write_head_sidecar(init, SPEAKERS,
                           model.state_dict()['projection.weight'].shape[0])
    return tiny_sets(assets) + sets


def tiny_sets(assets):
    """The overrides of ``tiny_config`` as ``--set`` strings."""
    return ['model.frontend.assets=%s' % assets,
            'model.frontend.num_layers=%d' % TINY_LAYERS] + \
        ['model.backend.%s=%d' % kv for kv in TINY_MODEL.items()]


def argv_of(sets, *flags):
    """Command line of ``ovsb.bin.train`` for a list of overrides."""
    argv = ['--config', os.path.join(RECIPE_DIR, 'conf', 'stage2_v9c.yaml')]
    for item in sets:
        argv += ['--set', item]
    return argv + list(flags)
