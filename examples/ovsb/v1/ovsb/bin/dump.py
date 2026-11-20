# Copyright (c) 2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Print fingerprints of what a config makes the trainer do (JSON).

    python -m ovsb.bin.dump conf/stage2_v9c.yaml --what sampler,batches \\
        --epoch 1 --rank 0 --worker 0 --batches 4

Sections (``--what``, comma separated):

    sampler   sha256 of the first ``--episodes`` episodes of the sampler
              cell (epoch, rank, worker): per row the utterance key,
              speaker, arm, language, session, position, positive type and
              recording conditions; rows per arm; the first rows in clear
    batches   sha256, shape and dtype of every field of the first
              ``--batches`` batches of that worker
    schedule  sha256 over the learning rates and the margin of every step,
              and their values at a few steps
    model     sha256 of the state-dict keys, shapes and dtypes; parameter
              counts

Two dumps of one config are equal; a change of the data stream, the
schedule or the model layout changes the matching digest. Nothing is
written to the experiment directory and no training step is taken. The
working directory must be the recipe directory (relative paths of the
config). ``sampler`` reads the training list, ``batches`` also the audio
and the noise / RIR sources, ``schedule`` and ``model`` build the model.

The batches of a worker depend on the base seed the data loader hands to
its workers. Without ``--base-seed`` the trainer's construction order is
followed (``ovsb.api.build_training``), which gives the base seed of
epoch 1 of a run started from this config.
"""

import argparse
import hashlib
import json
import struct
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ovsb import api
from ovsb.config import schema
from ovsb.config.schema import Config, ConfigError
from ovsb.data.loader import worker_batches
from ovsb.data.sampler import EpisodicSampler

SECTIONS = ('sampler', 'batches', 'schedule', 'model')
EPISODE_FIELDS = ('key', 'spk', 'arm', 'lang', 'vid', 'kpos', 'ptype', 'seg',
                  'dev', 'dcls', 'dial', 'genre')


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def tensor_digest(value: Any) -> Dict[str, Any]:
    """Shape, dtype and sha256 of the bytes of a tensor."""
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def dump_sampler(sampler: EpisodicSampler, epoch: int, rank: int, world: int,
                 worker: int, num_workers: int,
                 episodes: int) -> Dict[str, Any]:
    """Digest of the first episodes of one sampler cell (no audio)."""
    table = sampler.table
    digest = hashlib.sha256()
    arm_rows = np.zeros(len(table.arm_names), np.int64)
    first: List[List[Any]] = []
    n_rows = n_episodes = 0
    for ep in sampler.iter_episodes(epoch=epoch, rank=rank, world=world,
                                    wid=worker, nw=num_workers,
                                    max_episodes=episodes):
        d = {k: v.tolist() for k, v in sampler.describe(ep).items()}
        for i in range(len(d['row'])):
            row = [table.index.key(d['row'][i]),
                   str(table.spk_names[d['spk'][i]])] + \
                [int(d[f][i]) for f in EPISODE_FIELDS[2:]]
            digest.update(('\t'.join(map(str, row)) + '\n').encode())
            arm_rows[row[2]] += 1
            if len(first) < 4:
                first.append(row)
        digest.update(b'--\n')
        n_episodes += 1
        n_rows += len(d['row'])
    return dict(epoch=epoch, rank=rank, world=world, worker=worker,
                num_workers=num_workers, fields=list(EPISODE_FIELDS),
                episodes=n_episodes, rows=n_rows, sha256=digest.hexdigest(),
                arm_names=list(table.arm_names), arm_rows=arm_rows.tolist(),
                first_rows=first)


def dump_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Digest of every field of a collated batch."""
    out: Dict[str, Any] = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[name] = tensor_digest(value)
        else:
            out[name] = dict(n=len(value), first=list(value[:2]),
                             sha256=sha_text('\n'.join(map(str, value))))
    return out


def dump_schedule(cfg: Config, model: torch.nn.Module) -> Dict[str, Any]:
    """Digest of (encoder lr, base lr, margin) of every step."""
    optimizer = api.build_optimizer(model, cfg)
    scheduler, margin = api.build_schedulers(optimizer, model, cfg)
    epoch_iter = schema.epoch_iter(cfg)
    total = cfg.run.num_epochs * epoch_iter
    warm = int(cfg.optim.warm_up_epoch * epoch_iter)
    show = sorted({0, 1, warm - 1, warm, epoch_iter, total // 2, total - 1}
                  & set(range(total)))
    digest = hashlib.sha256()
    samples = {}
    for step in range(total):
        scheduler.step(step)
        margin.step(step)
        values = [float(g['lr']) for g in optimizer.param_groups] + \
            [float(margin.get_margin())]
        digest.update(struct.pack('<%dd' % len(values), *values))
        if step in show:
            samples[str(step)] = values
    return dict(
        num_epochs=cfg.run.num_epochs, epoch_iter=epoch_iter,
        total_steps=total, sha256=digest.hexdigest(), samples=samples,
        columns=['lr of the %s group' % (
            'encoder' if g['_is_frontend'] else 'base')
            for g in optimizer.param_groups] + ['margin'],
        group_tensors=[len(g['params']) for g in optimizer.param_groups])


def dump_model(model: torch.nn.Module) -> Dict[str, Any]:
    """Digest of the state-dict layout of the model."""
    state = model.state_dict()
    names = sorted(state)
    return dict(
        n_keys=len(names), keys_sha256=sha_text('\n'.join(names)),
        shapes_sha256=sha_text(json.dumps(
            [[k, list(state[k].shape)] for k in names])),
        dtypes_sha256=sha_text(json.dumps(
            [[k, str(state[k].dtype)] for k in names])),
        top_level_modules=[n for n, _ in model.named_children()],
        n_params=int(sum(p.numel() for p in model.parameters())))


def run(cfg: Config, what: List[str], epoch: int = 1, rank: int = 0,
        worker: int = 0, episodes: int = 200, n_batches: int = 4,
        base_seed: Optional[int] = None) -> Dict[str, Any]:
    """Build what the sections need and return their dumps."""
    world = schema.world_size(cfg)
    num_workers = max(1, cfg.data.loader.num_workers)
    out: Dict[str, Any] = {}
    need_model = bool({'schedule', 'model'} & set(what)) or (
        'batches' in what and base_seed is None)
    model = None
    if need_model:
        objects = api.build_training(cfg, rank, world, torch.device('cpu'))
        derived = int(torch.empty((), dtype=torch.int64).random_().item())
        sampler, dataset, loader = (objects.sampler, objects.dataset,
                                    objects.loader)
        model = objects.model
        if base_seed is None:
            base_seed = derived
            out['base_seed_source'] = 'construction order of the trainer'
    else:
        sampler = api.build_sampler(cfg, api.build_table(cfg))
        dataset = loader = None
        if 'batches' in what:
            dataset = api.build_dataset(cfg, sampler)
            loader = api.build_loader(dataset, cfg)
    if 'sampler' in what:
        out['sampler'] = dump_sampler(sampler, epoch, rank, world, worker,
                                      num_workers, episodes)
    if 'batches' in what:
        if cfg.data.loader.num_workers < 1:
            raise ConfigError('data.loader.num_workers: the batches section '
                              'needs at least one worker')
        dataset.set_epoch(epoch)
        batches = worker_batches(loader, worker, n_batches, base_seed,
                                 rank=rank, world=world)
        out['batches'] = dict(
            epoch=epoch, rank=rank, world=world, worker=worker,
            num_workers=num_workers, base_seed=base_seed,
            batches=[dump_batch(b) for b in batches])
    if 'schedule' in what:
        out['schedule'] = dump_schedule(cfg, model)
    if 'model' in what:
        out['model'] = dump_model(model)
    return out


def get_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', help='recipe config (YAML)')
    parser.add_argument('--legacy', metavar='AUTHORED_CONF', default=None,
                        help='the file is a config saved by the original '
                        'wespeaker-schema trainer; AUTHORED_CONF is the '
                        'hand-written conf of that run')
    parser.add_argument('--set', dest='overrides', action='append',
                        default=[], metavar='PATH=VALUE',
                        help='override, e.g. run.seed=1 (repeatable)')
    parser.add_argument('--what', default='sampler',
                        help='sections, comma separated: %s (default: '
                        'sampler)' % ', '.join(SECTIONS))
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=200,
                        help='episodes hashed by the sampler section')
    parser.add_argument('--batches', type=int, default=4,
                        help='batches hashed by the batches section')
    parser.add_argument('--base-seed', dest='base_seed', type=int,
                        default=None, help='base seed of the data-loader '
                        'workers (default: the one the trainer gets in '
                        'epoch 1)')
    parser.add_argument('--out', default=None,
                        help='write the JSON here instead of stdout')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = get_args(argv)
    what = [w for w in args.what.split(',') if w]
    unknown = [w for w in what if w not in SECTIONS]
    if unknown or not what:
        print('unknown section(s) %s; choose from %s' % (
            unknown, ', '.join(SECTIONS)), file=sys.stderr)
        return 2
    try:
        cfg = api.load_config(args.config, legacy=args.legacy,
                              overrides=args.overrides)
        result = run(cfg, what, epoch=args.epoch, rank=args.rank,
                     worker=args.worker, episodes=args.episodes,
                     n_batches=args.batches, base_seed=args.base_seed)
    except ConfigError as exc:
        print('INVALID %s\n%s' % (args.config, exc), file=sys.stderr)
        return 2
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, 'w') as fout:
            fout.write(text + '\n')
    else:
        sys.stdout.write(text + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
