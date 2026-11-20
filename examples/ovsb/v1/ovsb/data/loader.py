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
"""The data loader of the training pipeline.

The dataset is an endless iterable; every worker runs its own copy and the
loader stacks ``batch_size`` consecutive rows of one worker with the default
collate. The crop stage relies on exactly this batching (see
``ovsb.data.pipeline``), so ``batch_size`` here is the one the pipeline was
built with and there is no custom collate function or batch sampler.

Workers are not persistent: a new iterator (one per epoch) starts new
worker processes, which see the epoch set on the dataset before. There is
no ``worker_init_fn``; torch seeds ``random``, numpy and torch of worker
``w`` from ``base_seed + w``, and ``base_seed`` is one draw from
``generator`` made when the iterator is created.

Random streams of a worker
    The sampler draws, the per-batch crop caps and the dither noise have
    their own generators keyed by (seed, epoch, rank, worker). Everything
    else (crop positions, augmentation types, noise and RIR picks, codec
    choices) comes from the worker's ``random`` module, i.e. from
    ``base_seed``.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data._utils import ExceptionWrapper
from torch.utils.data._utils import worker as torch_worker

from ovsb.config.schema import Config


def loader_args(cfg: Config) -> Dict[str, Any]:
    """Keyword arguments of the ``DataLoader`` for ``cfg.data.loader``."""
    loader = cfg.data.loader
    args: Dict[str, Any] = dict(batch_size=loader.batch_size,
                                num_workers=loader.num_workers,
                                pin_memory=loader.pin_memory,
                                drop_last=loader.drop_last)
    if loader.num_workers > 0:  # torch refuses the option without workers
        args['prefetch_factor'] = loader.prefetch_factor
    return args


def build_loader(dataset: IterableDataset, cfg: Config,
                 generator: Optional[torch.Generator] = None) -> DataLoader:
    """The training ``DataLoader`` over ``dataset``.

    Args:
        dataset: the dataset of
            :func:`ovsb.data.pipeline.build_train_dataset`.
        cfg: the recipe config (``data.loader``).
        generator: source of the workers' base seed. None means the global
            torch generator: creating an iterator then consumes one int64
            draw from it, so the base seed of an epoch depends on every
            draw made from the global generator before. ``DataLoader``
            reads the attribute ``generator`` when an iterator is created;
            it may be replaced between epochs.
    """
    return DataLoader(dataset, generator=generator, **loader_args(cfg))


def worker_batches(loader: DataLoader, worker_id: int, n_batches: int,
                   base_seed: int, rank: Optional[int] = None,
                   world: Optional[int] = None,
                   timeout: float = 900.0) -> List[Dict[str, Any]]:
    """The first batches of one worker of ``loader``, for inspection.

    Runs torch's own worker loop (seeding of ``random`` / numpy / torch
    from ``base_seed + worker_id``, worker info, fetching, collation) for
    worker ``worker_id`` of ``loader.num_workers`` in one forked process,
    without starting the other workers. The batches are those the worker
    delivers in a real run whose iterator drew ``base_seed``. Set the
    epoch on the dataset before the call.

    Args:
        loader: a loader of :func:`build_loader` with ``num_workers`` > 0.
        worker_id: which worker to run.
        n_batches: number of batches to fetch.
        base_seed: the base seed of the emulated iterator.
        rank, world: when both are given, ``torch.distributed`` reports
            this rank and world size inside the forked process, so the
            dataset behaves as on that process of a distributed run.
        timeout: seconds to wait for one batch.
    """
    num_workers = loader.num_workers
    if num_workers < 1:
        raise ValueError('the loader has no worker processes')
    if not 0 <= worker_id < num_workers:
        raise ValueError('worker %r is not in [0, %d)' %
                         (worker_id, num_workers))
    if (rank is None) != (world is None):
        raise ValueError('give both rank and world, or neither')
    context = multiprocessing.get_context('fork')
    index_queue, data_queue = context.Queue(), context.Queue()
    done = context.Event()

    def target() -> None:
        if rank is not None:
            import torch.distributed as dist
            dist.is_initialized = lambda: True
            dist.get_rank = lambda *a, **k: rank
            dist.get_world_size = lambda *a, **k: world
        torch_worker._worker_loop(
            loader._dataset_kind, loader.dataset, index_queue, data_queue,
            done, loader._auto_collation, loader.collate_fn,
            loader.drop_last, base_seed, loader.worker_init_fn, worker_id,
            num_workers, loader.persistent_workers, None)

    process = context.Process(target=target, daemon=True)
    process.start()
    batches = []
    try:
        indices = iter(loader.batch_sampler)
        for k in range(n_batches):
            index_queue.put((k, next(indices)))
        for k in range(n_batches):
            idx, data = data_queue.get(timeout=timeout)
            if isinstance(data, ExceptionWrapper):
                data.reraise()
            assert idx == k, (idx, k)
            # copy out of shared memory before the worker goes away
            batches.append({key: (value.clone()
                                  if isinstance(value, torch.Tensor)
                                  else value)
                            for key, value in data.items()})
    finally:
        done.set()
        index_queue.put(None)
        process.join(60)
        if process.is_alive():
            process.terminate()
    return batches
