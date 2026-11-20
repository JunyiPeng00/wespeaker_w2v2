"""Data loader: arguments, worker seeding, real worker processes."""

import os
import sys

import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_helpers as ph  # noqa: E402
from conftest import RECIPE_DIR  # noqa: E402

from ovsb.config import load_config  # noqa: E402
from ovsb.data import pipeline  # noqa: E402
from ovsb.data.collate import BATCH_FIELDS  # noqa: E402
from ovsb.data.lmdb_multi import (open_noise_source,  # noqa: E402
                                  open_reverb_source)
from ovsb.data.loader import (build_loader, loader_args,  # noqa: E402
                              worker_batches)

BATCH = ph.P * ph.K
WORKERS = 2


@pytest.fixture(scope='module')
def recipe(tmp_path_factory):
    r = ph.make_toy_recipe(tmp_path_factory.mktemp('toy_loader'))
    r.table = pipeline.table_from_config(r.cfg)
    r.reverb = open_reverb_source(r.cfg.data.augment)
    r.noise = open_noise_source(r.cfg.data.augment)
    r.dataset = pipeline.build_train_dataset(
        r.cfg, pipeline.sampler_from_config(r.cfg, r.table), r.reverb,
        r.noise)
    return r


def generator(seed=5):
    return torch.Generator().manual_seed(seed)


def base_seed_of(seed=5):
    """The base seed an iterator draws from ``generator(seed)``."""
    return int(torch.empty((), dtype=torch.int64).random_(
        generator=generator(seed)).item())


def run_epoch(recipe, epoch, n, stream=0, seed=5):
    """The first ``n`` batches of a real multi-worker pass over an epoch."""
    recipe.dataset.set_epoch(epoch, stream)
    loader = build_loader(recipe.dataset, recipe.cfg, generator(seed))
    return ph.take(loader, n)


def same_batch(a, b):
    assert list(a) == list(b)
    for name in a:
        if isinstance(a[name], torch.Tensor):
            assert a[name].dtype == b[name].dtype, name
            assert torch.equal(a[name], b[name]), name
        else:
            assert a[name] == b[name], name
    return True


# ---------------------------------------------------------------- arguments
def test_arguments_of_the_reference_config():
    cfg = load_config(os.path.join(RECIPE_DIR, 'conf', 'stage2_v9c.yaml'))
    assert loader_args(cfg) == dict(batch_size=48, num_workers=12,
                                    prefetch_factor=8, pin_memory=False,
                                    drop_last=True)
    assert cfg.data.loader.batch_size == \
        cfg.data.sampler.P * cfg.data.sampler.K


def test_loader_object(recipe):
    loader = build_loader(recipe.dataset, recipe.cfg)
    assert isinstance(loader, DataLoader)
    assert loader.dataset is recipe.dataset
    assert (loader.batch_size, loader.num_workers, loader.prefetch_factor,
            loader.pin_memory, loader.drop_last) == (BATCH, WORKERS, 2,
                                                     False, True)
    # plain torch behaviour: default collate, no init hook, fresh workers
    # for every iterator, base seed from the global generator
    assert loader.collate_fn is default_collate
    assert loader.worker_init_fn is None
    assert loader.persistent_workers is False
    assert loader.generator is None
    assert loader.batch_sampler.batch_size == BATCH
    assert loader.batch_sampler.drop_last is True
    gen = generator()
    assert build_loader(recipe.dataset, recipe.cfg, gen).generator is gen


def test_no_workers_drops_the_prefetch_option(recipe):
    cfg = ph.variant(recipe, **{'data.loader.num_workers': 0})
    assert 'prefetch_factor' not in loader_args(cfg)
    loader = build_loader(recipe.dataset, cfg)
    recipe.dataset.set_epoch(1)
    ph.seed_all(1)
    batch = next(iter(loader))
    assert tuple(batch) == BATCH_FIELDS and len(batch['key']) == BATCH


def test_generator_none_takes_the_base_seed_from_the_global_generator(recipe):
    cfg = ph.variant(recipe, **{'data.loader.num_workers': 1})
    recipe.dataset.set_epoch(1)
    torch.manual_seed(1234)
    want = int(torch.empty((), dtype=torch.int64).random_().item())
    after = torch.rand(1).item()
    torch.manual_seed(1234)
    iterator = iter(build_loader(recipe.dataset, cfg))
    assert iterator._base_seed == want
    assert torch.rand(1).item() == after  # exactly one draw was consumed
    del iterator
    # a generator of its own leaves the global one untouched
    torch.manual_seed(1234)
    iterator = iter(build_loader(recipe.dataset, cfg, generator(5)))
    assert iterator._base_seed == base_seed_of(5)
    assert int(torch.empty((), dtype=torch.int64).random_().item()) == want
    del iterator


# ------------------------------------------------------------ real workers
def test_batches_of_a_worker_are_its_own_consecutive_rows(recipe):
    n = 10
    got = run_epoch(recipe, 1, n)
    assert len(got) == n
    sampler = recipe.dataset.sampler
    speakers = []
    for worker in range(WORKERS):
        mine = got[worker::WORKERS]  # batch k comes from worker k % WORKERS
        used = ph.utterance_keys(mine)
        # the worker's batches, one after the other, are the episode stream
        # of its own sampler cell; nothing is skipped or interleaved
        stream = ph.cell_keys(sampler, 1, 1, worker, WORKERS, len(mine))
        assert used == stream[:len(used)], worker
        for batch in mine:
            assert tuple(batch) == BATCH_FIELDS
            assert len(batch['key']) == BATCH
            assert batch['wav'].shape[:2] == (BATCH, 1)
        speakers.append({label for b in mine for label in
                         b['label'].tolist()})
    # the cells of the two workers share no speaker
    assert speakers[0] and speakers[1]
    assert not speakers[0] & speakers[1]
    assert any((b['pair_role'] > 0).any() for b in got)
    assert any(not (b['pair_role'] > 0).any() for b in got)


def test_same_epoch_and_stream_repeat_exactly(recipe):
    first = run_epoch(recipe, 1, 6)
    again = run_epoch(recipe, 1, 6)
    assert all(same_batch(a, b) for a, b in zip(first, again))
    # another base seed: same utterances (sampler) and caps, other crops
    other = run_epoch(recipe, 1, 6, seed=6)
    assert ph.utterance_keys(other) == ph.utterance_keys(first)
    assert [b['wav'].shape for b in other] == [b['wav'].shape for b in first]
    assert any(not torch.equal(a['wav_len'], b['wav_len'])
               for a, b in zip(first, other))


def test_epochs_and_streams_differ(recipe):
    first = run_epoch(recipe, 1, 6)
    second = run_epoch(recipe, 2, 6)
    restart = run_epoch(recipe, 1, 6, stream=1)
    keys = [ph.utterance_keys(x) for x in (first, second, restart)]
    assert keys[0] != keys[1] and keys[0] != keys[2] and keys[1] != keys[2]
    digests = [{ph.row_digest(b) for b in x}
               for x in (first, second, restart)]
    assert not digests[0] & digests[1] and not digests[0] & digests[2]
    assert all(same_batch(a, b) for a, b in
               zip(restart, run_epoch(recipe, 1, 6, stream=1)))
    # back to stream 0 of epoch 1: the first pass again
    assert all(same_batch(a, b) for a, b in
               zip(first, run_epoch(recipe, 1, 6)))


# ------------------------------------------------------- single-worker view
def test_worker_batches_equal_the_real_loader(recipe):
    real = run_epoch(recipe, 1, 6)
    loader = build_loader(recipe.dataset, recipe.cfg)
    recipe.dataset.set_epoch(1)
    for worker in range(WORKERS):
        alone = worker_batches(loader, worker, 3, base_seed_of(5))
        assert all(same_batch(a, b)
                   for a, b in zip(alone, real[worker::WORKERS]))


def test_worker_batches_as_another_rank(recipe):
    loader = build_loader(recipe.dataset, recipe.cfg)
    recipe.dataset.set_epoch(1)
    sampler = recipe.dataset.sampler
    seed = base_seed_of(5)
    for rank in (0, 1):
        got = worker_batches(loader, 1, 2, seed, rank=rank, world=2)
        used = ph.utterance_keys(got)
        stream = ph.cell_keys(sampler, 1, 2, 1, WORKERS, 2, rank=rank)
        assert used == stream[:len(used)]
    with pytest.raises(ValueError, match='both rank and world'):
        worker_batches(loader, 0, 1, seed, rank=1)
    with pytest.raises(ValueError, match='not in'):
        worker_batches(loader, WORKERS, 1, seed)
    cfg = ph.variant(recipe, **{'data.loader.num_workers': 0})
    with pytest.raises(ValueError, match='no worker'):
        worker_batches(build_loader(recipe.dataset, cfg), 0, 1, seed)
