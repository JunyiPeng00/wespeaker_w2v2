"""Toy recipe of the pipeline, loader and API tests.

:func:`make_toy_recipe` writes the synthetic corpus of ``data_parity``
(tar shards of 75-150 ms noise bursts, label and side files, a tar index)
and the tiny noise / RIR LMDBs of ``proc_parity``, and returns the
reference recipe config shrunk onto them: 7 arms, P x K = 12 x 2, crops of
a few hundred samples, two loader workers, the channel chain off (no
FFmpeg needed), optionally the tiny encoder of ``test_build``.
"""

import copy
import hashlib
import itertools
import os
import random
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

import data_parity as dp
import proc_parity as pp
from conftest import RECIPE_DIR

from ovsb.config.load import read_tree
from ovsb.config.schema import build_config
from ovsb.data.index import build_index

SEED = 3
P, K = 12, 2
CAPS = [[0.05, 0.5], [0.075, 0.5]]  # 800 / 1200 samples


def toy_tree(toy: Dict[str, Any], lmdbs: Dict[str, Any],
             assets: Optional[str] = None) -> Dict[str, Any]:
    """The reference config tree rewritten for the toy corpus."""
    root, files, ep = toy['root'], toy['files'], toy['episodic']
    tree = read_tree(os.path.join(RECIPE_DIR, 'conf', 'stage2_v9c.yaml'))
    tree['run'].update(seed=SEED, gpus=[0],
                       exp_dir=os.path.join(root, 'exp'))
    data = tree['data']
    data['lists'] = dict(train_label=files['utt2spk'],
                         index=os.path.join(root, 'index'),
                         spk2corpus=files['spk2corpus'],
                         spk2lang=files['spk2lang'], index_root=root)
    policies = copy.deepcopy(ep['construction']['arms'])
    for pol in policies.values():
        for key in ('subdialect_file', 'spk_weight_file'):
            if key in pol:
                pol[key] = os.path.join(root, pol[key])
    arms = {name: dict(weight=arm['weight'], corpus=(
        [arm['corpus']] if isinstance(arm['corpus'], str)
        else list(arm['corpus']))) for name, arm in ep['arms'].items()}
    data['sampler'] = dict(
        P=P, K=K, sample_num_per_epoch=20 * P * K, vox2_lang='en',
        stratify='systematic', log_every=500, arms=arms,
        cells=copy.deepcopy(ep['cells']), forbid_same_segment=True,
        policies=policies)
    # utterances have 1200-2400 samples: none is dropped (the one of 500
    # samples is), those above 1920 are cut to a random chunk
    data['filter'] = dict(min_sec=0.06, max_sec=0.12)
    data['crop'].update(
        max_secs=CAPS, min_sec=0.03, onset_max_sec=0.04,
        pair_anchor_sec=0.09, pair_view_secs=[0.03, 0.06],
        arm_caps={'cn2': CAPS, 'cn1': CAPS})
    data['crop']['energy_gate']['gate_below_sec'] = 0.06
    data['augment'].update(
        arm_probs={n: {'noise': 0.2} for n in ('cn2', '3dspeaker', 'cn1')},
        reverb_sources=['simulated', lmdbs['reverb_real']],
        noise_lmdb=lmdbs['noise_lmdb'], reverb_lmdb=lmdbs['reverb_lmdb'],
        noise_exclude_list=lmdbs['noise_exclude_list'])
    data['channel'].update(prob=0.0, isolate_ffmpeg=False)
    data['loader'].update(batch_size=P * K, num_workers=2, prefetch_factor=2)
    if assets is not None:
        from test_build import TINY_LAYERS, TINY_MODEL
        tree['model']['frontend'].update(assets=assets,
                                         num_layers=TINY_LAYERS)
        tree['model']['backend'].update(TINY_MODEL)
    return tree


def make_toy_recipe(root: Any, assets: Optional[str] = None
                    ) -> SimpleNamespace:
    """Write the toy corpus under ``root``; returns ``(cfg, tree, toy)``.

    ``tree`` is the plain config mapping: a test copies and changes it and
    calls ``build_config`` for a variant.
    """
    root = str(root)
    toy = dp.make_toy(os.path.join(root, 'corpus'))
    build_index(toy['files']['shard.list'],
                os.path.join(toy['root'], 'index'), nj=1, root=toy['root'])
    lmdbs = pp.make_aug_lmdbs(os.path.join(root, 'aug'))
    tree = toy_tree(toy, lmdbs, assets)
    return SimpleNamespace(cfg=build_config(copy.deepcopy(tree)), tree=tree,
                           toy=toy, root=root)


def variant(recipe: SimpleNamespace, **changes: Any) -> Any:
    """Config with dotted-path changes, e.g. ``**{'run.seed': 4}``."""
    tree = copy.deepcopy(recipe.tree)
    for dotted, value in changes.items():
        node = tree
        keys = dotted.split('.')
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return build_config(tree)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def take(iterable: Iterable[Any], n: int) -> List[Any]:
    """The first ``n`` items (the iterator is dropped afterwards)."""
    return list(itertools.islice(iter(iterable), n))


def row_digest(row: Dict[str, Any]) -> str:
    """One digest over every field of a pipeline row or a batch."""
    h = hashlib.sha256()
    for name in sorted(row):
        value = row[name]
        if isinstance(value, torch.Tensor):
            h.update(np.ascontiguousarray(value.numpy()).tobytes())
        h.update(('%s=%r;' % (name, value if not isinstance(
            value, torch.Tensor) else tuple(value.shape))).encode())
    return h.hexdigest()


def cell_keys(sampler: Any, epoch: int, world: int, worker: int,
              num_workers: int, episodes: int, rank: int = 0) -> List[str]:
    """Utterance keys of the first episodes of a sampler cell, in order."""
    keys = []
    table = sampler.table
    for ep in sampler.iter_episodes(epoch=epoch, rank=rank, world=world,
                                    wid=worker, nw=num_workers,
                                    max_episodes=episodes):
        keys.extend(table.index.key(int(r))
                    for r in sampler.describe(ep)['row'])
    return keys


def utterance_keys(batches: Iterable[Dict[str, Any]]) -> List[str]:
    """Keys of the sampler rows behind collated batches, in order.

    A pair batch holds every utterance twice (anchor, view).
    """
    keys: List[str] = []
    for batch in batches:
        rows = list(batch['key'])
        if bool((batch['pair_role'] > 0).any()):
            assert rows[0::2] == rows[1::2]
            assert batch['pair_role'].tolist() == [1, 2] * (len(rows) // 2)
            rows = rows[0::2]
        keys.extend(rows)
    return keys
