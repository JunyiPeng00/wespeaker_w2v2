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
"""The training pipeline: config -> speaker table, sampler, stage chain.

Stages of :func:`build_train_dataset`, in order (``Processor`` /
``EpochProcessor`` are the stock wespeaker stream wrappers):

1. :class:`~ovsb.data.sampler.EpisodicSampler`: P x K episodes with audio;
2. ``filter``: utterances longer than ``data.filter.max_sec`` are cut to a
   random chunk of that length (one draw of ``random`` each). Utterances
   shorter than ``min_sec`` would be dropped here, which would break the
   K-groups; for audio stored at 16 kHz none arrives, because the speaker
   table leaves out everything below the same length (in index samples);
3. ``spk_to_id``: speaker label -> classifier class;
4. ``resample`` to ``data.resample_rate``;
5. :func:`~ovsb.data.crop.random_chunk_var`: variable crops and pair
   batches. This stage owns the batch boundaries, see below;
6. :func:`~ovsb.data.zero_run.zero_run_guard`: zero-run statistics, dither;
7. :func:`~ovsb.data.augment.draw_aug_type`;
8. :func:`~ovsb.data.augment.add_reverb_noise`;
9. :func:`~ovsb.data.channel.add_channel_asym` (only when
   ``data.channel.prob`` > 0; FFmpeg is checked when the chain is built);
10. :func:`~ovsb.data.collate.pad_batch`.

Every stage after the sampler is 1:1 except stage 5, which yields two rows
per utterance in a pair batch. The data loader must batch exactly
``data.loader.batch_size`` consecutive rows of one worker's stream
(:func:`ovsb.data.loader.build_loader`): the crop stage counts the rows it
yields and starts a new batch (new length cap, new pair decision) every
``batch_size`` rows. A pair batch therefore consumes ``batch_size / 2``
sampler rows, i.e. half an episode, and the batches after it straddle two
episodes.

Epochs and streams
    :meth:`TrainDataset.set_epoch` must be called in the main process
    before the workers of an epoch are started. It passes the epoch to the
    sampler and to stages 5 and 6, whose random sequences are keyed by
    (seed, epoch, worker). ``stream`` is 0 for a run that goes through the
    epoch once and is then not passed to the stages at all; a restart
    inside an epoch passes a larger id, which selects fresh sequences.

Processes
    The rank and world size of a distributed run and the worker id reach
    the sampler and stage 6 through ``torch.distributed`` and
    ``torch.utils.data.get_worker_info()`` inside the worker process;
    nothing is passed at build time. The crop sequence (stage 5) does not
    depend on the rank: all ranks draw the same batch lengths. The LMDB
    sources are opened once, in the main process, before the workers fork.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

from torch.utils.data import IterableDataset

import wespeaker.dataset.processor as core
from ovsb.config.schema import Config, FilterCfg
from ovsb.data import augment, channel, crop, zero_run
from ovsb.data.collate import pad_batch
from ovsb.data.sampler import EpisodicSampler
from ovsb.data.speakers import (ArmSpec, CellSpec, DialectMass, PolicySpec,
                                SpeakerTable)
from ovsb.data.weights import build_weights
from wespeaker.dataset.dataset import EpochProcessor, Processor

# hop of the encoder frames in which the length filter counts
FRAME_SHIFT_MS = 20
# sampling rate the index lengths are measured at when utterances shorter
# than data.filter.min_sec are left out of the speaker table
INDEX_SAMPLE_RATE = 16000


def arm_index(cfg: Config) -> Dict[str, int]:
    """Arm name -> the integer the sampler writes into ``sample['arm']``."""
    return {name: i for i, name in enumerate(cfg.data.sampler.arm_order)}


def filter_frames(cfg: FilterCfg) -> Tuple[int, int]:
    """(min, max) utterance length in encoder frames of a filter config."""
    def frames(seconds: float) -> int:
        return int(round(float(seconds) * 1000.0 / FRAME_SHIFT_MS))
    return frames(cfg.min_sec), frames(cfg.max_sec)


def table_min_len(cfg: Config) -> int:
    """Shortest utterance (index samples) kept in the speaker table."""
    return int(FRAME_SHIFT_MS / 1000 * filter_frames(cfg.data.filter)[0] *
               INDEX_SAMPLE_RATE)


def table_from_config(cfg: Config) -> SpeakerTable:
    """The speaker table of ``cfg.data.lists`` and ``cfg.data.sampler``.

    Relative paths are relative to the working directory. Arms and cells
    are taken in ``arm_order`` / ``cell_order``.
    """
    lists, sampler = cfg.data.lists, cfg.data.sampler
    arms = [ArmSpec(name, sampler.arms[name].weight,
                    tuple(sampler.arms[name].corpus))
            for name in sampler.arm_order]
    cells = [CellSpec(name, sampler.cells[name].arm,
                      sampler.cells[name].share,
                      sampler.cells[name].select.spk2lang)
             for name in sampler.cell_order]
    policies = {}
    for name, pol in sampler.policies.items():
        mass = pol.dialect_mass
        policies[name] = PolicySpec(
            pair_policy=pol.pair_policy, p_dialect=pol.p_dialect,
            segment_first=pol.segment_first,
            dialect_mass=None if mass is None else DialectMass(
                boost=mass.boost, cap_x_median=mass.cap_x_median),
            w_bridge=pol.w_bridge, subdialect_file=pol.subdialect_file,
            spk_weight_file=pol.spk_weight_file,
            spk_weight_share=pol.spk_weight_share)
    return SpeakerTable.from_files(
        lists.train_label, lists.index, lists.spk2corpus, lists.spk2lang,
        arms=arms, cells=cells, policies=policies,
        vox2_lang=sampler.vox2_lang, K=sampler.K,
        forbid_same_segment=sampler.forbid_same_segment,
        min_len=table_min_len(cfg), index_root=lists.index_root)


def sampler_from_config(cfg: Config, table: SpeakerTable) -> EpisodicSampler:
    """The episodic sampler of ``cfg`` over ``table``.

    Its seed is ``cfg.run.seed`` on every rank; the rank enters the seed
    keys inside the worker processes.
    """
    sampler = cfg.data.sampler
    if sampler.P * sampler.K != cfg.data.loader.batch_size:
        raise ValueError(
            'data.sampler: P x K (%d) must equal data.loader.batch_size (%d)'
            % (sampler.P * sampler.K, cfg.data.loader.batch_size))
    return EpisodicSampler(
        table, build_weights(table), sampler.P, sampler.K,
        sample_num_per_epoch=sampler.sample_num_per_epoch,
        seed=cfg.run.seed, stratify=sampler.stratify,
        log_every=sampler.log_every,
        rank_split_arms=sampler.rank_split_arms)


class TrainDataset(IterableDataset):
    """The stage chain of :func:`build_train_dataset`.

    Attributes:
        source: the last stage (a ``Processor``); following ``.source``
            from stage to stage ends at the sampler.
        sampler: the episodic sampler at the start of the chain.
        epoch, stream: as set by :meth:`set_epoch`.
    """

    def __init__(self, source: Processor, sampler: EpisodicSampler,
                 epoch_stages: List[EpochProcessor]) -> None:
        self.source = source
        self.sampler = sampler
        self._epoch_stages = list(epoch_stages)
        self.epoch = 0
        self.stream = 0

    def set_epoch(self, epoch: int, stream: int = 0) -> None:
        """Set the epoch (and stream id) of the following iterations.

        Call it in the main process before the data loader starts the
        workers of the epoch. ``stream`` > 0 only for the rest of an epoch
        that is restarted in the middle.
        """
        self.epoch, self.stream = int(epoch), int(stream)
        self.source.set_epoch(self.epoch)  # epoch stages and the sampler
        self.sampler.set_epoch(self.epoch, self.stream)
        for stage in self._epoch_stages:
            if self.stream:
                stage.kw['stream'] = self.stream
            else:
                stage.kw.pop('stream', None)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.source)


def build_train_dataset(cfg: Config, sampler: EpisodicSampler, reverb: Any,
                        noise: Any) -> TrainDataset:
    """Chain the stages of the module doc behind ``sampler``.

    Args:
        cfg: the recipe config.
        sampler: the episodic sampler (:func:`sampler_from_config`).
        reverb: the RIR source
            (:func:`ovsb.data.lmdb_multi.open_reverb_source`).
        noise: the noise source
            (:func:`ovsb.data.lmdb_multi.open_noise_source`).

    Raises:
        RuntimeError: the channel chain is configured but torchaudio cannot
            reach FFmpeg.
    """
    data = cfg.data
    if reverb is None or noise is None:
        raise ValueError('the pipeline needs a reverb and a noise source')
    table = sampler.table
    arms = arm_index(cfg)
    if list(table.arm_names) != list(data.sampler.arm_order):
        raise ValueError('the sampler arms %s are not data.sampler.arm_order '
                         '%s' % (list(table.arm_names),
                                 data.sampler.arm_order))
    rate, seed, batch_size = data.resample_rate, cfg.run.seed, \
        data.loader.batch_size
    spk2id = {str(name): i for i, name in enumerate(table.class_names)}
    min_frames, max_frames = filter_frames(data.filter)

    chain: Any = sampler
    chain = Processor(chain, core.filter, frame_shift=FRAME_SHIFT_MS,
                      data_type='shard', min_num_frames=min_frames,
                      max_num_frames=max_frames)
    chain = Processor(chain, core.spk_to_id, spk2id)
    chain = Processor(chain, core.resample, rate)
    crop_stage = chain = EpochProcessor(
        chain, crop.random_chunk_var,
        **crop.crop_args(data.crop, arms, batch_size, rate, seed))
    dither_stage = chain = EpochProcessor(
        chain, zero_run.zero_run_guard,
        **zero_run.zero_run_args(data.zero_run_dither, arms, rate, seed))
    chain = Processor(chain, augment.draw_aug_type,
                      **augment.draw_args(data.augment, arms))
    chain = Processor(chain, augment.add_reverb_noise, reverb, noise, rate,
                      reverberant_noise=data.augment.reverberant_noise)
    channel.require_ffmpeg(data.channel)
    if data.channel.prob > 0.0:
        chain = Processor(chain, channel.add_channel_asym,
                          **channel.channel_args(data.channel, arms, rate))
    chain = Processor(chain, pad_batch)
    return TrainDataset(chain, sampler, [crop_stage, dither_stage])


def find_sampler(dataset: Any) -> Optional[EpisodicSampler]:
    """The episodic sampler at the start of a stage chain, or None."""
    while dataset is not None:
        if isinstance(dataset, EpisodicSampler):
            return dataset
        dataset = getattr(dataset, 'source', None)
    return None
