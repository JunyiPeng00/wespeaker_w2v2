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
"""Public construction API of the recipe.

One import for scripts, notebooks and tests that need a piece of the
training set-up without starting a training run: nothing in this module
launches training, touches ``exp_dir`` or initialises a process group.

Pieces (each usable on its own):

* :func:`load_config` - a recipe YAML, or a config saved by the original
  wespeaker-schema trainer;
* :func:`build_table`, :func:`build_sampler`, :func:`open_sources`,
  :func:`build_dataset`, :func:`build_loader` - the data side;
* :func:`build_model`, :func:`build_losses`, :func:`build_optimizer`,
  :func:`build_schedulers` - the model side;
* :func:`forward_losses` - one forward of a batch with the objective.

:func:`build_training` builds all of them in the one order the trainer
uses. The order matters for reproducibility: the random initialisation of
the model consumes the global torch generator, and the data loader takes
the base seed of its workers from the same generator when its first
iterator is created (see ``ovsb.data.loader``). Seeding, then the data
side, then the model, and no other draw from the global torch generator
before the first ``iter(loader)``, gives every rank the worker seeds of the
reference run.

Paths in a config are relative to the working directory (the recipe
directory).
"""

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ovsb.config import legacy as _legacy
from ovsb.config import load as _load
from ovsb.config import schema
from ovsb.config.schema import Config
from ovsb.data.lmdb_multi import open_noise_source, open_reverb_source
from ovsb.data.loader import build_loader
from ovsb.data.pipeline import (TrainDataset, build_train_dataset,
                                sampler_from_config, table_from_config)
from ovsb.data.sampler import EpisodicSampler
from ovsb.data.speakers import SpeakerTable
from ovsb.engine.optim import build_optimizer, build_schedulers
from ovsb.engine.step import forward_losses
from ovsb.losses.bundle import LossBundle
from ovsb.losses.pair_auc import RunningCenter
from ovsb.model.build import build_model
from wespeaker.utils.utils import set_seed

__all__ = ['TrainingObjects', 'build_dataset', 'build_loader', 'build_losses',
           'build_model', 'build_optimizer', 'build_sampler',
           'build_schedulers', 'build_table', 'build_training',
           'forward_losses', 'load_config', 'open_sources']

logger = logging.getLogger(__name__)


def load_config(path: str, legacy: Optional[str] = None,
                overrides: Iterable[str] = ()) -> Config:
    """Read a config file into a validated :class:`Config`.

    Args:
        path: a recipe config (YAML, with its ``base`` overlays), or with
            ``legacy`` the ``config.yaml`` a run of the original
            wespeaker-schema trainer saved.
        legacy: the hand-written conf that run was launched with. A saved
            config has its keys sorted, so the order of the sampler arms
            (which is behaviour) is taken from this file.
        overrides: ``dotted.path=value`` strings applied after the
            overlays; not available with ``legacy``.
    """
    if legacy:
        if list(overrides):
            raise schema.ConfigError(
                'overrides are not available with a legacy config')
        return _legacy.load_legacy(path, legacy)
    return _load.load(path, overrides)


def build_table(cfg: Config) -> SpeakerTable:
    """The speaker table of the training list (reads the label file)."""
    return table_from_config(cfg)


def build_sampler(cfg: Config, table: SpeakerTable) -> EpisodicSampler:
    """The episodic sampler over ``table`` (seed ``cfg.run.seed``)."""
    return sampler_from_config(cfg, table)


def open_sources(cfg: Config) -> Tuple[Any, Any]:
    """(reverb, noise) LMDB sources of ``cfg.data.augment``.

    An LMDB environment can be opened once per process: open the sources
    once, in the main process, and hand them to :func:`build_dataset`.
    """
    return (open_reverb_source(cfg.data.augment),
            open_noise_source(cfg.data.augment))


def build_dataset(cfg: Config, sampler: EpisodicSampler, reverb: Any = None,
                  noise: Any = None) -> TrainDataset:
    """The training pipeline behind ``sampler``.

    ``reverb`` / ``noise`` are the sources of :func:`open_sources`; when
    both are None they are opened here.
    """
    if reverb is None and noise is None:
        reverb, noise = open_sources(cfg)
    return build_train_dataset(cfg, sampler, reverb, noise)


def build_losses(cfg: Config) -> LossBundle:
    """The objective of ``cfg.loss`` with a fresh running centre.

    The bundle holds per-process state (the centre): build one per process
    and keep it for the whole run.
    """
    pauc = cfg.loss.pair_auc
    center = RunningCenter(
        cfg.data.sampler.arm_order.index(pauc.center.arm),
        pauc.center.momentum, pauc.center.warmup)
    return LossBundle(cfg.loss.consistency_weight, pauc.weight, pauc.scale,
                      center)


@dataclass
class TrainingObjects:
    """Everything :func:`build_training` builds for one process."""
    table: SpeakerTable
    sampler: EpisodicSampler
    dataset: TrainDataset
    loader: DataLoader
    model: torch.nn.Module  # unwrapped, on the device, no checkpoint loaded
    bundle: LossBundle
    optimizer: torch.optim.Optimizer
    scheduler: Any
    margin_scheduler: Any


def build_training(cfg: Config, rank: int, world: int,
                   device: torch.device) -> TrainingObjects:
    """Build the training objects of one process in the canonical order.

    1. seed python, numpy and torch with ``cfg.run.seed + rank``;
    2. speaker table, sampler, noise / RIR sources, stage chain;
    3. model (random initialisation of backend and head, pretrained
       encoder), moved to ``device``;
    4. optimiser and schedulers, objective;
    5. data loader (``generator=None``).

    Steps 2, 4 and 5 draw nothing from the global torch generator. The
    caller must not draw from it either before the first ``iter(loader)``
    if the worker seeds of epoch 1 are to be those of the reference run.
    Loading a checkpoint into ``model`` (which the caller does: weights,
    then optimiser state on a resume) draws nothing.

    Args:
        cfg: the recipe config.
        rank: index of this process among the training processes.
        world: number of training processes; must be ``len(cfg.run.gpus)``,
            from which the steps per epoch and the learning-rate scale are
            derived.
        device: device of the model.

    Returns:
        :class:`TrainingObjects`. ``model`` is not wrapped for distributed
        training and carries no checkpoint weights yet.
    """
    if not 0 <= int(rank) < int(world):
        raise ValueError('rank %r is not in [0, %r)' % (rank, world))
    if int(world) != schema.world_size(cfg):
        raise ValueError(
            'world size %d, but run.gpus lists %d devices: the steps per '
            'epoch and the learning-rate scale follow run.gpus' %
            (world, schema.world_size(cfg)))
    set_seed(cfg.run.seed + int(rank))
    table = build_table(cfg)
    sampler = build_sampler(cfg, table)
    reverb, noise = open_sources(cfg)
    dataset = build_train_dataset(cfg, sampler, reverb, noise)
    model = build_model(cfg, table.n_class,
                        logger if rank == 0 else None)
    model.to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler, margin_scheduler = build_schedulers(optimizer, model, cfg)
    bundle = build_losses(cfg)
    loader = build_loader(dataset, cfg, generator=None)
    return TrainingObjects(table, sampler, dataset, loader, model, bundle,
                           optimizer, scheduler, margin_scheduler)
