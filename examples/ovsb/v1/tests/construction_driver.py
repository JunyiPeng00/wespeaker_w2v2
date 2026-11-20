"""What a training process is built from, for tests/test_train_entry.py.

    python tests/construction_driver.py reference CONFIG OUT.json
    python tests/construction_driver.py entry CONFIG OUT.json [--set ...]

``reference`` calls ``ovsb.api.build_training`` and asks its loader for the
first batch. ``entry`` runs ``ovsb.bin.train`` on the same config and
records the same things where the trainer uses them. Both write: the
sha256 of every model tensor before any checkpoint is loaded, the base
seed the first DataLoader iterator drew for its workers, and the keys of
the first batch. One process each: an LMDB environment cannot be opened
twice in a process.
"""
import hashlib
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import RECIPE_DIR  # noqa: E402  (sets the import paths)

from ovsb import api  # noqa: E402
from ovsb.bin import train as train_mod  # noqa: E402
from ovsb.engine import loop as loop_mod  # noqa: E402


def digest(model):
    return {name: hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, tensor in model.state_dict().items()}


def reference(config):
    cfg = api.load_config(config)
    obj = api.build_training(cfg, 0, 1, torch.device('cpu'))
    out = {'weights': digest(obj.model)}
    obj.dataset.set_epoch(1)
    iterator = iter(obj.loader)
    out['base_seed'] = int(iterator._base_seed)
    out['first_keys'] = list(next(iterator)['key'])
    return out


def entry(config, extra):
    out = {}
    real_iter, real_load = DataLoader.__iter__, train_mod.checkpoint.load_init
    real_step = loop_mod.train_step

    def first_iter(self):
        iterator = real_iter(self)
        out.setdefault('base_seed', int(iterator._base_seed))
        return iterator

    def load_init(model, *args, **kwargs):
        out['weights'] = digest(model)
        return real_load(model, *args, **kwargs)

    def step(model, bundle, batch, *args, **kwargs):
        out.setdefault('first_keys', list(batch['key']))
        return real_step(model, bundle, batch, *args, **kwargs)

    DataLoader.__iter__ = first_iter
    train_mod.checkpoint.load_init = load_init
    loop_mod.train_step = step
    out['rc'] = train_mod.main(['--config', config] + extra)
    return out


if __name__ == '__main__':
    assert os.path.isdir(os.path.join(RECIPE_DIR, 'ovsb'))
    torch.set_num_threads(2)
    mode, config_path, out_path = sys.argv[1:4]
    result = reference(config_path) if mode == 'reference' else \
        entry(config_path, sys.argv[4:])
    with open(out_path, 'w') as fout:
        json.dump(result, fout)
