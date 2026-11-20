"""Process and epoch seeding (ovsb/engine/seeding.py)."""
import random

import numpy as np
import pytest
import torch

from ovsb.engine import seeding
from wespeaker.utils.utils import set_seed

SEED = 716


def draws(generator=None):
    """One value from each global generator (and from ``generator``)."""
    out = [random.random(), float(np.random.rand()), float(torch.rand(()))]
    if generator is not None:
        out.append(int(torch.empty((), dtype=torch.int64).random_(
            generator=generator)))
    return out


def snapshot():
    return (random.getstate(), np.random.get_state()[1].tolist(),
            np.random.get_state()[2], torch.get_rng_state().tolist())


def seed_process(seed, rank):
    """Process start, as ``ovsb.api.build_training`` does it."""
    set_seed(seed + rank)


def test_reseed_makes_the_calls_of_the_core_set_seed():
    benchmark = torch.backends.cudnn.benchmark
    try:
        set_seed(4242)
        want = snapshot()
    finally:
        torch.backends.cudnn.benchmark = benchmark
    seeding._seed_all(4242)
    assert snapshot() == want
    assert torch.backends.cudnn.benchmark == benchmark  # flags untouched


def test_first_pass_keeps_the_process_stream():
    seed_process(SEED, 1)
    draws()  # model construction consumes the generators
    before = snapshot()
    assert seeding.begin_epoch(SEED, 1, 1, 0) is None
    assert snapshot() == before
    assert seeding.loader_generator(SEED, 1, 1, 0) is None
    assert seeding.is_first_pass(1, 0)
    assert not seeding.is_first_pass(1, 1) and not seeding.is_first_pass(2, 0)


@pytest.mark.parametrize('epoch,stream', [(2, 0), (1, 1), (5, 2)])
def test_other_passes_do_not_depend_on_the_history(epoch, stream):
    seed_process(SEED, 0)
    for _ in range(7):
        draws()
    first = draws(seeding.begin_epoch(SEED, 0, epoch, stream))
    seed_process(SEED + 99, 0)  # another history
    again = draws(seeding.begin_epoch(SEED, 0, epoch, stream))
    assert first == again and len(first) == 4


def test_resumed_epoch_starts_like_the_uninterrupted_one():
    """The invariant of the resume fix, on the generator states."""
    def uninterrupted():
        seed_process(SEED, 2)
        draws()                              # model construction
        seeding.begin_epoch(SEED, 2, 1, 0)
        for _ in range(11):
            draws()                          # epoch 1
        gen = seeding.begin_epoch(SEED, 2, 2, 0)
        return snapshot(), gen.get_state().tolist()

    def resumed():
        seed_process(SEED, 2)
        draws()                              # model construction
        gen = seeding.begin_epoch(SEED, 2, 2, 0)
        return snapshot(), gen.get_state().tolist()

    assert uninterrupted() == resumed()
    # without the reseed the resumed process would be at the process-start
    # stream, i.e. where the uninterrupted run was at the start of epoch 1
    seed_process(SEED, 2)
    draws()
    replay = snapshot()
    assert replay != resumed()[0]


def test_keys_give_distinct_seeds():
    keys = [(SEED, r, e, s) for r in range(8) for e in range(1, 15)
            for s in range(3)]
    seeds = [seeding.epoch_seed(*k) for k in keys]
    assert len(set(seeds)) == len(keys)
    assert all(0 <= s < 2 ** 32 for s in seeds)
    assert seeding.epoch_seed(SEED, 0, 2, 0) == seeding.epoch_seed(
        SEED, 0, 2, 0)
    assert seeding.epoch_seed(SEED, 0, 2) != seeding.epoch_seed(SEED + 1, 0, 2)
    # the DataLoader generator is not the global seed of the same key
    gen = seeding.loader_generator(SEED, 0, 2, 0)
    assert gen.initial_seed() != seeding.epoch_seed(SEED, 0, 2, 0)
    other = seeding.loader_generator(SEED, 1, 2, 0)
    assert gen.initial_seed() != other.initial_seed()
    assert gen.initial_seed() == seeding.loader_generator(
        SEED, 0, 2, 0).initial_seed()


def test_negative_key_is_refused():
    with pytest.raises(ValueError, match='>= 0'):
        seeding.epoch_seed(SEED, 0, -1, 0)
