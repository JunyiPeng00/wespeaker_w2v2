"""Read a recipe config: YAML, overlays, overrides, derived values.

Overlay mechanism: a config file may name another file with the top-level
key ``base`` (path relative to the file that names it). The base is read
first and the rest of the file is merged into it: mappings are merged key
by key, any other value (scalar, list) replaces the base value. A base may
itself have a base.

The order of ``data.sampler.arms`` (and ``cells``) is behaviour. Merging
keeps it: an entry the overlay changes stays at its position, an entry the
overlay adds goes to the end. To choose another order an overlay sets
``data.sampler.arm_order`` explicitly.

Overrides are ``dotted.path=value`` strings (value in YAML syntax) applied
after the overlays, e.g. ``run.exp_dir=exp/trial``.

Relative paths inside a config are relative to the recipe directory (the
working directory of the tools), not to the config file.
"""

import copy
import dataclasses
import os
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

from ovsb.config import schema
from ovsb.config.schema import Config, ConfigError, DerivedCfg

BASE_KEY = 'base'


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a mapping with a repeated key."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if key in seen:
                raise ConfigError('%s line %d: key %r is repeated' % (
                    key_node.start_mark.name, key_node.start_mark.line + 1,
                    key))
            seen.add(key)
        return super().construct_mapping(node, deep)


def read_yaml(path: str) -> Dict[str, Any]:
    """One YAML file -> mapping (no overlay handling)."""
    try:
        with open(path, 'r', encoding='utf8') as fin:
            tree = yaml.load(fin, Loader=_UniqueKeyLoader)
    except OSError as exc:
        raise ConfigError('%s: cannot read config file (%s)' %
                          (path, exc.strerror)) from exc
    except yaml.YAMLError as exc:
        raise ConfigError('%s: not valid YAML (%s)' % (path, exc)) from exc
    if not isinstance(tree, dict):
        raise ConfigError('%s: the top level must be a mapping' % path)
    return tree


def merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> Dict[str, Any]:
    """``over`` merged into a copy of ``base`` (mappings merge, rest replaces)."""
    out = copy.deepcopy(dict(base))
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def read_tree(path: str, _chain: tuple = ()) -> Dict[str, Any]:
    """A config file with its chain of bases merged in."""
    real = os.path.realpath(path)
    if real in _chain:
        raise ConfigError('%s: %s cycle through %s' % (path, BASE_KEY, real))
    tree = read_yaml(path)
    base = tree.pop(BASE_KEY, None)
    if base is None:
        return tree
    if not isinstance(base, str) or not base:
        raise ConfigError('%s: %s must be a path, got %r' %
                          (path, BASE_KEY, base))
    base_path = os.path.join(os.path.dirname(path), base)
    return merge(read_tree(base_path, _chain + (real,)), tree)


def apply_overrides(tree: Mapping[str, Any],
                    overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply ``dotted.path=value`` strings to a copy of ``tree``."""
    out = copy.deepcopy(dict(tree))
    for item in overrides:
        dotted, sep, text = str(item).partition('=')
        keys = dotted.strip().split('.')
        if not sep or not all(keys):
            raise ConfigError('override %r: expected dotted.path=value' %
                              (item,))
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError('override %s: value is not valid YAML (%s)' %
                              (dotted, exc)) from exc
        node = out
        for depth, key in enumerate(keys[:-1]):
            if not isinstance(node.get(key), dict):
                raise ConfigError('override %s: %s is not a mapping of the '
                                  'config' % (dotted,
                                              '.'.join(keys[:depth + 1])))
            node = node[key]
        node[keys[-1]] = value
    return out


def load(path: str, overrides: Iterable[str] = ()) -> Config:
    """File (+ bases, + overrides) -> validated :class:`Config`.

    ``derived`` is filled only when the file carries it (a resolved config
    written by the trainer); call :func:`resolve` to compute it.
    """
    return schema.build_config(apply_overrides(read_tree(path), overrides))


def count_speakers(train_label: str) -> int:
    """Distinct speakers (second column) of an utt2spk file."""
    speakers = set()
    with open(train_label, 'r', encoding='utf8') as fin:
        for number, line in enumerate(fin, 1):
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) < 2:
                raise ConfigError('data.lists.train_label: %s line %d has no '
                                  'speaker column' % (train_label, number))
            speakers.add(tokens[1])
    return len(speakers)


def resolve(cfg: Config, num_class: Optional[int] = None,
            root: Optional[str] = None) -> Config:
    """A copy of ``cfg`` with ``derived`` filled.

    * ``epoch_iter`` = sample_num_per_epoch // processes // batch_size
    * ``scale_ratio`` = processes x batch_size / 64
    * ``num_class`` = ``num_class`` if given, else the number of distinct
      speakers of ``data.lists.train_label`` (relative to ``root``, default
      the working directory).

    A ``derived`` section already present must agree with these values.
    """
    if num_class is None:
        label = cfg.data.lists.train_label
        if not os.path.isabs(label):
            label = os.path.join(root or os.getcwd(), label)
        try:
            num_class = count_speakers(label)
        except OSError as exc:
            raise ConfigError(
                'data.lists.train_label: cannot read %s to count the '
                'classes (%s)' % (label, exc.strerror)) from exc
    if isinstance(num_class, bool) or not isinstance(num_class, int) \
            or num_class < 1:
        raise ConfigError('derived.num_class: must be an integer >= 1, got '
                          '%r' % (num_class,))
    derived = DerivedCfg(epoch_iter=schema.epoch_iter(cfg),
                         scale_ratio=schema.scale_ratio(cfg),
                         num_class=num_class)
    if cfg.derived is not None and cfg.derived != derived:
        for fld in dataclasses.fields(DerivedCfg):
            old, new = getattr(cfg.derived, fld.name), getattr(derived,
                                                               fld.name)
            if old != new:
                raise ConfigError(
                    'derived.%s: %r in the config, %r computed' %
                    (fld.name, old, new))
    return schema.validate(dataclasses.replace(cfg, derived=derived))


def to_dict(cfg: Config) -> Dict[str, Any]:
    """Plain nested dict; ``schema.build_config`` reads it back."""
    return dataclasses.asdict(cfg)


def dump(cfg: Config) -> str:
    """YAML text of the config, keys in schema order."""
    return yaml.safe_dump(to_dict(cfg), sort_keys=False,
                          default_flow_style=None)
