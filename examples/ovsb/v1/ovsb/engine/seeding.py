"""Seeding of a training process, per epoch and per stream.

A training process (one DDP rank) uses four global generators: Python
``random``, numpy, the torch CPU generator and the torch CUDA generators.
They are consumed by the model construction, by the frame masking of the
encoder (every step) and, through the DataLoader, by the workers: when a
DataLoader starts an epoch it draws one base seed and seeds worker ``w``
with ``base seed + w``; augmentation type, crop start, noise and codec
choices of the workers follow from it. The sampler, the per-batch crop
lengths and the dither have their own generators keyed by ``(seed, epoch,
rank, worker)``; they do not depend on anything here.

An epoch is identified by ``(epoch, stream)``. ``stream`` is 0 when the
epoch is run from its first step and is increased by one every time the
epoch is restarted in the middle after an interruption.

Rules

* Process start: ``ovsb.api.build_training`` seeds the four generators
  with ``seed + rank`` (the wespeaker ``set_seed``, as the original trainer
  does) before it builds the data side and the model. Not done here.
* Start of ``(epoch 1, stream 0)``: nothing is done and the DataLoader gets
  no generator, so it takes its base seed from the torch CPU generator.
  The first epoch of a run started from scratch therefore consumes the
  generators exactly as the original wespeaker-schema trainer does.
* Start of any other ``(epoch, stream)``: the four generators are reseeded
  with ``epoch_seed(seed, rank, epoch, stream)`` and the DataLoader gets
  its own generator seeded from the same key (``loader_generator``).

What is reproducible across an interruption

* A run stopped at an epoch boundary and resumed at epoch ``e`` starts
  ``e`` with the generators, the DataLoader base seed and the sampler keys
  of an uninterrupted run. Together with the restored weights, optimiser
  and running centre the continuation is the uninterrupted run, bit for bit
  on a deterministic backend (CPU). On CUDA it is as reproducible as two
  runs of the same job (cuDNN autotuning and non-deterministic kernels).
* A run stopped inside epoch ``e`` at step ``s`` resumes at step ``s`` with
  ``stream + 1``: the remaining steps see data drawn from fresh generators
  (sampler cells, crops, augmentation and frame masks), never a replay of
  the steps already trained on, and not the data the uninterrupted run
  would have seen there. Weights therefore differ from the uninterrupted
  run from step ``s`` on. Epochs ``e + 1, ...`` again start from the same
  generator states as in the uninterrupted run.

What is not reproduced

* The original trainer does not reseed: epoch 2 onwards continues the
  generators of epoch 1 there and is reseeded here, so from the second
  epoch on the two trainers draw different (equally distributed) data
  and masks. In the original a resumed process replays the process-start
  generators (the DataLoader base seed and the masks of epoch 1) in the
  first epoch after every restart; here it does not.
"""
import random
from typing import Optional

import numpy as np
import torch

# last entry of the seed key: which generator the value is for
_GLOBAL, _LOADER = 0, 1


def _seed_all(value: int) -> None:
    """The seeding calls of ``wespeaker.utils.utils.set_seed``, in its
    order, without its change of the cuDNN flags."""
    np.random.seed(value)
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)  # ignored without CUDA
    torch.cuda.manual_seed_all(value)


def is_first_pass(epoch: int, stream: int) -> bool:
    """True for ``(epoch 1, stream 0)``, which keeps the process seeding."""
    return int(epoch) == 1 and int(stream) == 0


def _derive(seed: int, rank: int, epoch: int, stream: int, purpose: int,
            dtype: type) -> int:
    key = [int(seed), int(rank), int(epoch), int(stream), purpose]
    if min(key) < 0:
        raise ValueError('seed, rank, epoch and stream must be >= 0, got %r'
                         % (key[:4], ))
    return int(np.random.SeedSequence(key).generate_state(1, dtype)[0])


def epoch_seed(seed: int, rank: int, epoch: int, stream: int = 0) -> int:
    """32-bit seed of the global generators for ``(epoch, stream)``."""
    return _derive(seed, rank, epoch, stream, _GLOBAL, np.uint32)


def loader_generator(seed: int, rank: int, epoch: int,
                     stream: int = 0) -> Optional[torch.Generator]:
    """Generator the DataLoader of ``(epoch, stream)`` draws its base seed
    from; None for ``(epoch 1, stream 0)`` (the torch CPU generator)."""
    if is_first_pass(epoch, stream):
        return None
    value = _derive(seed, rank, epoch, stream, _LOADER, np.uint64)
    return torch.Generator().manual_seed(value % (1 << 63))


def begin_epoch(seed: int, rank: int, epoch: int,
                stream: int = 0) -> Optional[torch.Generator]:
    """Set the generators for the start of ``(epoch, stream)``.

    Call once per process before the DataLoader of the epoch is created.

    Returns:
        The generator to pass to the DataLoader (None for ``(epoch 1,
        stream 0)``, where nothing is reseeded either).
    """
    if not is_first_pass(epoch, stream):
        _seed_all(epoch_seed(seed, rank, epoch, stream))
    return loader_generator(seed, rank, epoch, stream)
