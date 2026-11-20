"""Public API: pieces, the canonical construction order, the dump tool."""

import dataclasses
import json
import os
import random
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_helpers as ph  # noqa: E402
from conftest import RECIPE_DIR  # noqa: E402
from test_build import AUTHORED, RUN_DIR, make_tiny_assets  # noqa: E402

from ovsb import api  # noqa: E402
from ovsb.bin import dump  # noqa: E402
from ovsb.config import ConfigError, to_dict  # noqa: E402
from ovsb.data.collate import BATCH_FIELDS  # noqa: E402
from ovsb.data.loader import worker_batches  # noqa: E402
from ovsb.data.pipeline import TrainDataset  # noqa: E402
from ovsb.data.sampler import EpisodicSampler  # noqa: E402
from ovsb.losses.bundle import LossBundle  # noqa: E402

CPU = torch.device('cpu')
RECIPE_CONFIG = os.path.join(RECIPE_DIR, 'conf', 'stage2_v9c.yaml')
SOURCE_RECIPE = os.environ.get('OVSB_SOURCE_RECIPE')


@pytest.fixture(scope='module')
def recipe(tmp_path_factory):
    assets = make_tiny_assets(tmp_path_factory.mktemp('tiny_w2vbert'))
    r = ph.make_toy_recipe(tmp_path_factory.mktemp('toy_api'), assets)
    # an LMDB environment is opened once per process: every construction
    # of this module reuses the sources of the first one
    r.objects = api.build_training(r.cfg, 0, 1, CPU)
    r.base_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    stage = {s.f.__name__: s for s in _stages(r.objects.dataset)}
    r.sources = stage['add_reverb_noise'].args[:2]
    r.config_path = os.path.join(r.root, 'toy.yaml')
    with open(r.config_path, 'w') as fout:
        yaml.safe_dump(r.tree, fout, sort_keys=False)
    return r


def _stages(dataset):
    out, node = [], dataset.source
    while not isinstance(node, EpisodicSampler):
        out.append(node)
        node = node.source
    return out


@pytest.fixture()
def shared_sources(recipe, monkeypatch):
    """Later constructions reuse the LMDB sources opened by the first."""
    monkeypatch.setattr(api, 'open_sources', lambda cfg: recipe.sources)


# ------------------------------------------------------------------ pieces
def test_surface():
    for name in ('load_config', 'build_table', 'build_sampler',
                 'build_dataset', 'build_loader', 'build_model',
                 'build_losses', 'build_optimizer', 'build_schedulers',
                 'forward_losses', 'build_training', 'TrainingObjects'):
        assert name in api.__all__ and callable(getattr(api, name)), name
    assert [f.name for f in dataclasses.fields(api.TrainingObjects)] == [
        'table', 'sampler', 'dataset', 'loader', 'model', 'bundle',
        'optimizer', 'scheduler', 'margin_scheduler']


def test_load_config(recipe):
    cfg = api.load_config(RECIPE_CONFIG)
    assert cfg.run.seed == 716 and cfg.derived is None
    over = api.load_config(RECIPE_CONFIG, overrides=['run.seed=5'])
    assert over.run.seed == 5
    assert to_dict(api.load_config(recipe.config_path)) == to_dict(recipe.cfg)
    with pytest.raises(ConfigError, match='overrides'):
        api.load_config(RECIPE_CONFIG, legacy='conf.yaml',
                        overrides=['run.seed=5'])
    with pytest.raises(ConfigError, match='cannot read'):
        api.load_config(os.path.join(recipe.root, 'absent.yaml'))


@pytest.mark.skipif(not SOURCE_RECIPE, reason='set OVSB_SOURCE_RECIPE')
def test_load_config_legacy_equals_the_recipe_config_but_for_paths():
    saved = os.path.join(SOURCE_RECIPE, RUN_DIR, 'config.yaml')
    authored = os.path.join(SOURCE_RECIPE, AUTHORED)
    legacy = to_dict(api.load_config(saved, legacy=authored))
    mine = to_dict(api.load_config(RECIPE_CONFIG))
    assert legacy.pop('derived') == dict(epoch_iter=3645, scale_ratio=6.0,
                                         num_class=152367)
    assert mine.pop('derived') is None
    assert legacy['data']['sampler']['arm_order'] == \
        mine['data']['sampler']['arm_order']
    for tree in (legacy, mine):
        tree['run'].pop('exp_dir')
        tree['data'].pop('lists')
        for key in ('noise_lmdb', 'reverb_lmdb', 'reverb_sources',
                    'noise_exclude_list'):
            tree['data']['augment'].pop(key)
        for pol in tree['data']['sampler']['policies'].values():
            pol.pop('subdialect_file'), pol.pop('spk_weight_file')
        tree['model']['init'].pop('checkpoint')
    assert legacy == mine


def test_build_losses(recipe):
    cfg = recipe.cfg
    bundle = api.build_losses(cfg)
    assert isinstance(bundle, LossBundle)
    assert bundle.consistency_weight == cfg.loss.consistency_weight == 0.5
    assert (bundle.pair_auc_weight, bundle.pair_auc_scale) == (0.2, 16.0)
    center = bundle.center
    assert center.arm == cfg.data.sampler.arm_order.index('vox2')
    assert (center.momentum, center.warmup, center.n_upd) == (0.99, 20, 0)
    assert api.build_losses(cfg).center is not center  # fresh state
    off = ph.variant(recipe, **{'loss.pair_auc.weight': 0.0})
    assert api.build_losses(off).center is None


def test_pieces_compose(recipe, shared_sources):
    cfg, obj = recipe.cfg, recipe.objects
    sampler = api.build_sampler(cfg, obj.table)
    assert isinstance(sampler, EpisodicSampler) and sampler.table is obj.table
    dataset = api.build_dataset(cfg, sampler)  # opens (here: reuses) sources
    assert isinstance(dataset, TrainDataset) and dataset.sampler is sampler
    explicit = api.build_dataset(cfg, sampler, *recipe.sources)
    assert [s.f for s in _stages(explicit)] == [s.f for s in _stages(dataset)]
    loader = api.build_loader(dataset, cfg)
    assert loader.dataset is dataset and loader.generator is None
    table = api.build_table(cfg)
    assert table.n_class == obj.table.n_class
    np.testing.assert_array_equal(table.rows, obj.table.rows)


# ------------------------------------------------------ canonical ordering
def test_training_objects(recipe):
    cfg, obj = recipe.cfg, recipe.objects
    assert obj.sampler.table is obj.table
    assert obj.dataset.sampler is obj.sampler
    assert obj.loader.dataset is obj.dataset and obj.loader.generator is None
    assert obj.sampler.seed == cfg.run.seed  # no rank in the sampler seed
    head = obj.model.projection.weight
    assert head.shape[0] == obj.table.n_class * cfg.model.head.K
    assert head.device == CPU
    groups = obj.optimizer.param_groups
    assert [g['_is_frontend'] for g in groups] == [True, False]
    assert sum(len(g['params']) for g in groups) == \
        len(list(obj.model.parameters()))
    assert obj.margin_scheduler.get_margin() == cfg.model.head.margin
    assert obj.bundle.center.n_upd == 0


def test_data_side_draws_nothing_from_the_torch_generator(recipe):
    """The base seed after build_training is the one that follows seeding
    and the model initialisation alone."""
    cfg = recipe.cfg
    api.set_seed(cfg.run.seed + 0)
    api.build_model(cfg, recipe.objects.table.n_class)
    want = int(torch.empty((), dtype=torch.int64).random_().item())
    assert recipe.base_seed == want


def test_build_training_is_reproducible_and_rank_dependent(recipe,
                                                           shared_sources):
    def run(cfg, rank, world):
        random.seed(99), np.random.seed(99), torch.manual_seed(99)
        obj = api.build_training(cfg, rank, world, CPU)
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        return obj, seed, random.random(), float(np.random.rand())

    obj, seed, py, npy = run(recipe.cfg, 0, 1)
    assert seed == recipe.base_seed
    for a, b in zip(obj.model.parameters(),
                    recipe.objects.model.parameters()):
        assert torch.equal(a, b)
    again = run(recipe.cfg, 0, 1)
    assert again[1:] == (seed, py, npy)
    two = ph.variant(recipe, **{'run.gpus': [0, 1]})
    obj1, seed1, py1, _ = run(two, 1, 2)
    assert seed1 != seed and py1 != py
    assert not torch.equal(obj1.model.projection.weight,
                           obj.model.projection.weight)
    # the sampler seed stays the run seed on every rank
    assert obj1.sampler.seed == recipe.cfg.run.seed
    with pytest.raises(ValueError, match='run.gpus'):
        api.build_training(recipe.cfg, 0, 2, CPU)
    with pytest.raises(ValueError, match='rank'):
        api.build_training(two, 2, 2, CPU)


def test_first_loader_iterator_gets_the_derived_base_seed(recipe,
                                                          shared_sources):
    obj = api.build_training(recipe.cfg, 0, 1, CPU)
    obj.dataset.set_epoch(1)
    iterator = iter(obj.loader)
    assert iterator._base_seed == recipe.base_seed
    batch = next(iterator)
    del iterator
    # ... and worker 0 delivers what the single-worker view predicts
    alone = worker_batches(obj.loader, 0, 1, recipe.base_seed)[0]
    assert alone['key'] == batch['key']
    assert torch.equal(alone['wav'], batch['wav'])


def test_forward_losses_on_a_pipeline_batch(recipe):
    cfg, obj = recipe.cfg, recipe.objects
    obj.dataset.set_epoch(1)
    batches = worker_batches(obj.loader, 0, 6, recipe.base_seed)
    assert all(tuple(b) == BATCH_FIELDS for b in batches)
    bundle = api.build_losses(cfg)
    obj.model.train()
    torch.manual_seed(1)
    seen = set()
    for batch in batches:
        out = api.forward_losses(obj.model, bundle, batch, cfg, CPU)
        assert torch.isfinite(out.loss) and out.loss.requires_grad
        assert out.logits.shape == (len(batch['key']), obj.table.n_class)
        paired = bool((batch['pair_role'] > 0).any())
        assert (out.parts['cons'] is not None) == paired
        seen.add(paired)
    assert seen == {True, False}
    out.loss.backward()
    assert obj.model.projection.weight.grad is not None


# ------------------------------------------------------------------- dump
def test_dump_sampler_and_batches(recipe, shared_sources, capsys):
    args = [recipe.config_path, '--what', 'sampler,batches', '--epoch', '2',
            '--worker', '1', '--episodes', '5', '--batches', '2',
            '--base-seed', '77']
    assert dump.main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert dump.main(args) == 0
    assert json.loads(capsys.readouterr().out) == first
    sam = first['sampler']
    assert (sam['epoch'], sam['rank'], sam['worker'], sam['num_workers'],
            sam['world']) == (2, 0, 1, 2, 1)
    assert sam['episodes'] == 5 and sam['rows'] == 5 * ph.P * ph.K
    assert sum(sam['arm_rows']) == sam['rows']
    assert sam['arm_names'] == list(recipe.cfg.data.sampler.arm_order)
    assert len(sam['first_rows'][0]) == len(sam['fields']) == 12
    bat = first['batches']
    assert bat['base_seed'] == 77 and len(bat['batches']) == 2
    assert set(bat['batches'][0]) == set(BATCH_FIELDS)
    assert bat['batches'][0]['wav']['shape'][:2] == [ph.P * ph.K, 1]
    assert 'schedule' not in first and 'model' not in first
    # another epoch or worker: another stream
    other = [recipe.config_path, '--what', 'sampler', '--epoch', '3',
             '--worker', '1', '--episodes', '5']
    assert dump.main(other) == 0
    assert json.loads(capsys.readouterr().out)['sampler']['sha256'] != \
        sam['sha256']


def test_dump_schedule_and_model(recipe, shared_sources, tmp_path):
    out = str(tmp_path / 'dump.json')
    assert dump.main([recipe.config_path, '--what', 'schedule,model,batches',
                      '--batches', '1', '--out', out]) == 0
    with open(out) as fin:
        got = json.load(fin)
    cfg = recipe.cfg
    sched = got['schedule']
    assert sched['epoch_iter'] == 20 and \
        sched['total_steps'] == 20 * cfg.run.num_epochs
    assert sched['samples']['0'] == [0.0, 0.0, cfg.model.head.margin]
    assert len(sched['columns']) == 3 and len(sched['sha256']) == 64
    model = got['model']
    state = recipe.objects.model.state_dict()
    assert model['n_keys'] == len(state)
    assert model['top_level_modules'][-2:] == ['frontend', 'projection']
    # no --base-seed: the trainer's construction order gives the seed
    assert got['batches']['base_seed'] == recipe.base_seed
    assert got['base_seed_source']


def test_dump_refuses_bad_input(recipe, capsys):
    assert dump.main([recipe.config_path, '--what', 'sampler,nothing']) == 2
    assert 'unknown section' in capsys.readouterr().err
    assert dump.main([os.path.join(recipe.root, 'absent.yaml')]) == 2
    assert 'INVALID' in capsys.readouterr().err
