"""Recipe config: schema, overlays, legacy import, core view, derived values.

Tests marked ``source`` compare against the original training tree and are
skipped unless ``OVSB_SOURCE_TREE`` points at its repository root.
"""

import copy
import glob
import inspect
import os
import subprocess
import sys

import pytest
import yaml

from conftest import RECIPE_DIR, REPO_ROOT
from ovsb.config import core_view, legacy, schema
from ovsb.config import load as cfg_load
from ovsb.config.schema import ConfigError

CONF = os.path.join(RECIPE_DIR, 'conf')
BASE = os.path.join(CONF, 'stage2_v9c.yaml')
SOURCE_TREE = os.environ.get('OVSB_SOURCE_TREE')
SOURCE_RECIPE = os.path.join(SOURCE_TREE or '', 'examples', 'voxceleb',
                             'v2_w2vbert_sv')
RESOLVED_V9C = os.path.join(SOURCE_RECIPE, 'exp', 's3', 'base_v9c',
                            'config.yaml')

source = pytest.mark.skipif(
    not (SOURCE_TREE and os.path.isfile(RESOLVED_V9C)),
    reason='OVSB_SOURCE_TREE is unset or has no resolved reference config')

# Site-dependent locations: compared for presence, not for value.
PATH_FIELDS = (
    'run.exp_dir',
    'data.lists.train_label',
    'data.lists.index',
    'data.lists.spk2corpus',
    'data.lists.spk2lang',
    'data.sampler.policies.mandarin_t1.spk_weight_file',
    'data.sampler.policies.kespeech.subdialect_file',
    'data.augment.reverb_sources',  # entries other than 'simulated'
    'data.augment.noise_lmdb',
    'data.augment.reverb_lmdb',
    'data.augment.noise_exclude_list',
    'model.frontend.assets',
    'model.init.checkpoint',
)
# Keys the original launcher passes on the command line or the trainer fills
# at start-up; an unresolved conf/*.yaml of the original tree lacks them.
RUNTIME_KEYS = ('train_label', 'noise_data', 'reverb_data')

# The order the reference run used (its hand-written conf, its train log).
ARM_ORDER = ['vox2', 'vb2', 'cn2', 'mandarin_t1', '3dspeaker', 'kespeech',
             'tvx', 'cn1']
CELL_ORDER = ['vb2_zh']
AUTHORED_V9C = os.path.join(SOURCE_RECIPE, 'conf', 's3_base_v9c.yaml')

OVERLAYS = {
    # overlay: (original conf, dotted paths that differ from stage2_v9c)
    'stage2_v8c1.yaml': ('s3_base_v8c1.yaml', {
        'run.exp_dir', 'run.num_epochs',
        'data.sampler.arms.vb2.weight', 'data.sampler.arms.cn2.weight',
        'data.sampler.arms.cn1.weight'}),
    'stage2_v9a.yaml': ('s3_base_v9a.yaml', {
        'run.exp_dir',
        'data.sampler.arms.vb2.weight', 'data.sampler.arms.cn2.weight',
        'data.sampler.arms.cn1.weight'}),
    'stage2_v9b.yaml': ('s3_base_v9b.yaml', {
        'run.exp_dir', 'run.num_epochs'}),
    'cont_c40.yaml': ('qd_c40.yaml', {
        'run.exp_dir', 'run.num_epochs', 'model.init.checkpoint',
        'data.sampler.arms.vox2.weight', 'data.sampler.arms.vb2.weight',
        'data.sampler.arms.3dspeaker.weight', 'data.sampler.arms.cn1.weight',
        'optim.initial_lr', 'optim.warm_up_epoch'}),
}

# (original conf, legacy dotted path): closed-experiment keys as they are set
# in real configs of the original tree. '*' = the first entry that has it.
CLOSED_EXAMPLES = (
    ('rl_R1c.yaml', 'mixture_controller'),
    ('rl_R1c.yaml', 'ptp_v2'),
    ('rl_R1c.yaml', 'tail'),
    ('rl_R2.yaml', 'grpo'),
    ('c31_star_dann.yaml', 'genre_adversary'),
    ('c31_star_ema.yaml', 'ema_consistency'),
    ('c31_star_lw10.yaml', 'optimizer_groups'),
    ('c31_star_k1.yaml', 'projection_args.k_per_class_file'),
    ('phase1_stage2.yaml', 'l2sp'),
    ('phase1_stage2.yaml', 'model_args.quality_coord'),
    ('phase1_stage2.yaml', 'model_args.layer_gates'),
    ('qd_phf3.yaml', 'model_args.per_head'),
    ('qd_xu.yaml', 'consistency_target'),
    ('qd_en.yaml', 'consistency_enrol'),
    ('qd_xg.yaml', 'pair_auc.cross_gpu'),
    ('qd_xg.yaml', 'pair_auc.logn_correction'),
    ('qd_pat.yaml', 'pair_auc.grad'),
    ('qd_lp.yaml', 'dataset_args.var_crop.pair_law'),
    ('qd_cg.yaml', 'dataset_args.episodic_sampler.cond_groups'),
    ('qd_sc.yaml', 'dataset_args.episodic_sampler.block'),
    ('qd_ps.yaml',
     'dataset_args.episodic_sampler.construction.arms.*.p_same_session'),
    ('c31_construction.yaml',
     'dataset_args.episodic_sampler.construction.pass_cap'),
    ('c31_construction.yaml',
     'dataset_args.episodic_sampler.construction.dcls_edges_m'),
    ('s1c_C4.yaml',
     'dataset_args.episodic_sampler.construction.parallel_view'),
    ('s2_base.yaml',
     'dataset_args.episodic_sampler.construction.arms.*.p_hard_pair'),
    ('s2_base.yaml',
     'dataset_args.episodic_sampler.construction.arms.*.genre_only_weight'),
    ('s2_base.yaml', 'dataset_args.episodic_sampler.cells.*.row_floor'),
    ('s2_base.yaml', 'dataset_args.speed_perturb_arms'),
    ('s2_base.yaml', 'dataset_args.interferer_args'),
    ('p1z_s2_datav2.yaml', 'dataset_args.channel_args'),
    ('tb_tail_m0_lr4x.yaml', 'dataset_args.token_batching'),
    ('tb_tail_m0_lr4x.yaml', 'margin_schedule'),
)


# ----------------------------------------------------------------- helpers
def _read(path):
    with open(path, 'r', encoding='utf8') as fin:
        return yaml.safe_load(fin)


def _get(tree, dotted):
    for key in dotted.split('.'):
        tree = tree[key]
    return tree


def _set(tree, dotted, value):
    keys = dotted.split('.')
    for key in keys[:-1]:
        tree = tree.setdefault(key, {})
    tree[keys[-1]] = value


def _delete(tree, dotted):
    keys = dotted.split('.')
    parent = _get(tree, '.'.join(keys[:-1])) if keys[:-1] else tree
    del parent[keys[-1]]


def _leaves(tree, prefix=''):
    """{dotted path: leaf value}; lists are leaves."""
    out = {}
    for key, value in tree.items():
        path = prefix + str(key)
        if isinstance(value, dict) and value:
            out.update(_leaves(value, path + '.'))
        else:
            out[path] = value
    return out


def _normalised(cfg):
    """to_dict with every site-dependent path replaced by a marker."""
    tree = cfg_load.to_dict(cfg)
    for dotted in PATH_FIELDS:
        value = _get(tree, dotted)
        assert value, dotted
        if isinstance(value, list):
            value = [v if v == schema.SIMULATED_REVERB else '<path>'
                     for v in value]
        else:
            assert isinstance(value, str), dotted
            value = '<path>'
        _set(tree, dotted, value)
    return tree


def _diff(a, b):
    la, lb = _leaves(cfg_load.to_dict(a)), _leaves(cfg_load.to_dict(b))
    return {k for k in set(la) | set(lb) if la.get(k) != lb.get(k)}


def _from_legacy(tree):
    """from_legacy with the order of the reference run."""
    return legacy.from_legacy(tree, arm_order=ARM_ORDER)


def _base_tree():
    return cfg_load.read_tree(BASE)


def _resolved_v9c():
    return _read(RESOLVED_V9C)


def _source_conf(name):
    """An unresolved conf of the original tree, completed as at launch."""
    tree = _read(os.path.join(SOURCE_RECIPE, 'conf', name))
    resolved = _resolved_v9c()
    for key in RUNTIME_KEYS:
        tree.setdefault(key, resolved[key])
    if tree.get('model_args', {}).get('feat_dim') == -1:
        # the trainer replaces -1 by the encoder width
        tree['model_args']['feat_dim'] = resolved['model_args']['feat_dim']
    return tree


def _expand(tree, dotted):
    """Resolve '*' in a dotted path against ``tree``."""
    keys = dotted.split('.')
    if '*' not in keys:
        return dotted
    i = keys.index('*')
    head, tail = '.'.join(keys[:i]), '.'.join(keys[i + 1:])
    for name, sub in _get(tree, head).items():
        try:
            _get(sub, tail)
        except (KeyError, TypeError):
            continue
        return '.'.join([head, name, tail])
    raise KeyError(dotted)


# ------------------------------------------------- the recipe config itself
def test_base_config_loads_and_round_trips():
    cfg = cfg_load.load(BASE)
    assert cfg.derived is None
    assert schema.build_config(cfg_load.to_dict(cfg)) == cfg
    assert schema.build_config(yaml.safe_load(cfg_load.dump(cfg))) == cfg
    assert cfg.data.sampler.arm_order == ARM_ORDER
    assert list(cfg.data.sampler.arms) == ARM_ORDER
    assert cfg.data.sampler.cell_order == CELL_ORDER


def test_committed_configs_have_no_site_paths():
    files = glob.glob(os.path.join(CONF, '*.yaml'))
    assert len(files) >= 5
    for path in files:
        tree = cfg_load.read_tree(path)
        for dotted, value in _leaves(tree).items():
            values = value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, str):
                    assert not os.path.isabs(item), (path, dotted, item)
                    assert not item.startswith('~'), (path, dotted, item)


def test_config_package_needs_no_torch_or_numpy():
    code = ('import sys; import ovsb.config, ovsb.bin.validate_config; '
            'bad = [m for m in ("torch", "numpy") if m in sys.modules]; '
            'sys.exit(1 if bad else 0)')
    env = dict(os.environ, PYTHONPATH=RECIPE_DIR)
    assert subprocess.run([sys.executable, '-c', code], env=env).returncode \
        == 0


# ------------------------------------------- (1) legacy import == clean yaml
@source
def test_legacy_v9c_equals_recipe_config():
    old = legacy.load_legacy(RESOLVED_V9C, AUTHORED_V9C)
    assert old == _from_legacy(_resolved_v9c())
    new = cfg_load.resolve(cfg_load.load(BASE),
                           num_class=old.derived.num_class)
    assert old.data.sampler.arm_order == new.data.sampler.arm_order \
        == ARM_ORDER
    assert list(old.data.sampler.arms) == list(new.data.sampler.arms) \
        == ARM_ORDER
    assert old.data.sampler.cell_order == new.data.sampler.cell_order
    assert _normalised(old) == _normalised(new)
    # nothing but the declared path fields differs before normalisation
    differing = _diff(old, new)
    assert differing <= set(PATH_FIELDS), differing - set(PATH_FIELDS)
    assert schema.SIMULATED_REVERB in old.data.augment.reverb_sources
    assert old.data.augment.reverb_sources.index(schema.SIMULATED_REVERB) == \
        new.data.augment.reverb_sources.index(schema.SIMULATED_REVERB)


@source
def test_legacy_v9c_covers_every_legacy_leaf():
    """No legacy key is silently dropped: each one is live, fixed, ignored
    or closed, and a misspelt key is refused (except inside the blocks that
    have no effect on this training path)."""
    no_effect = ('dataset_args.cmvn_args.', 'dataset_args.shuffle_args.',
                 'dataset_args.spec_aug_args.', 'grad_norm_adam.')
    tree = _resolved_v9c()
    checked = 0
    for dotted in _leaves(tree):
        if dotted.startswith(no_effect):
            continue
        broken = copy.deepcopy(tree)
        _set(broken, dotted + '_typo', 1)
        with pytest.raises(ConfigError, match='_typo'):
            _from_legacy(broken)
        checked += 1
    assert checked > 150


# -------------------------------------------- (2) overlays == original confs
@source
@pytest.mark.parametrize('overlay', sorted(OVERLAYS))
def test_overlay_matches_original_conf(overlay):
    conf, expected = OVERLAYS[overlay]
    new = cfg_load.load(os.path.join(CONF, overlay))
    old = _from_legacy(_source_conf(conf))
    assert _normalised(old) == _normalised(new)
    assert _diff(new, cfg_load.load(BASE)) == expected
    old_base = _from_legacy(_source_conf('s3_base_v9c.yaml'))
    old_diff = _diff(old, old_base)
    if overlay != 'cont_c40.yaml':
        assert old_diff == expected
    else:  # the original names its init on the command line
        assert old_diff | {'model.init.checkpoint'} == expected
    for dotted in expected - set(PATH_FIELDS):
        assert _get(cfg_load.to_dict(new), dotted) == \
            _get(cfg_load.to_dict(old), dotted), dotted


def test_overlays_change_only_the_documented_keys():
    base = cfg_load.load(BASE)
    for overlay, (_, expected) in OVERLAYS.items():
        path = os.path.join(CONF, overlay)
        assert _diff(cfg_load.load(path), base) == expected, overlay
        own = cfg_load.read_yaml(path)
        assert own.pop('base') == 'stage2_v9c.yaml'
        assert set(_leaves(own)) == expected, overlay


def test_overlay_mechanism(tmp_path):
    (tmp_path / 'a.yaml').write_text(
        'base: sub/b.yaml\nrun: {seed: 3}\nextra: {x: [1]}\n')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'b.yaml').write_text(
        'run: {seed: 1, gpus: [0, 1]}\nextra: {x: [5, 6], y: 2}\n')
    tree = cfg_load.read_tree(str(tmp_path / 'a.yaml'))
    assert tree == {'run': {'seed': 3, 'gpus': [0, 1]},
                    'extra': {'x': [1], 'y': 2}}
    (tmp_path / 'loop.yaml').write_text('base: loop.yaml\n')
    with pytest.raises(ConfigError, match='cycle'):
        cfg_load.read_tree(str(tmp_path / 'loop.yaml'))
    (tmp_path / 'dup.yaml').write_text('run: {seed: 1}\nrun: {seed: 2}\n')
    with pytest.raises(ConfigError, match="'run' is repeated"):
        cfg_load.read_tree(str(tmp_path / 'dup.yaml'))
    with pytest.raises(ConfigError, match='cannot read'):
        cfg_load.read_tree(str(tmp_path / 'missing.yaml'))


def test_overrides():
    cfg = cfg_load.load(BASE, ['run.exp_dir=exp/x',
                               'data.sampler.arms.cn1.weight=0.3',
                               'data.zero_run_dither.arms=[cn1, cn2]'])
    assert cfg.run.exp_dir == 'exp/x'
    assert cfg.data.sampler.arms['cn1'].weight == 0.3
    assert cfg.data.zero_run_dither.arms == ['cn1', 'cn2']
    assert cfg_load.load(BASE, ['optim.initial_lr=1.0e-5']).optim.initial_lr \
        == 1e-5
    for bad, where in ((['run.sed=1'], 'run.sed'),
                       (['run.seed'], 'run.seed'),
                       (['runn.seed=1'], 'runn'),
                       (['run.seed=abc'], 'run.seed')):
        with pytest.raises(ConfigError, match=where):
            cfg_load.load(BASE, bad)


def test_scientific_notation_without_dot_is_refused():
    # YAML 1.1 reads 1e-5 as a string; the error names the field
    with pytest.raises(ConfigError, match='optim.initial_lr: must be a '
                       'number'):
        cfg_load.load(BASE, ['optim.initial_lr=1e-5'])


# --------------------------------------------------- arm order is behaviour
def _reordered(tree, dotted, order):
    mapping = _get(tree, dotted)
    _set(tree, dotted, {key: mapping[key] for key in order})
    return tree


@pytest.mark.parametrize('name', ['stage2_v9c.yaml'] + sorted(OVERLAYS))
def test_target_configs_use_the_reference_arm_order(name):
    sampler = cfg_load.load(os.path.join(CONF, name)).data.sampler
    assert sampler.arm_order == ARM_ORDER
    assert list(sampler.arms) == ARM_ORDER
    assert sampler.cell_order == list(sampler.cells) == CELL_ORDER
    tree = cfg_load.to_dict(cfg_load.load(os.path.join(CONF, name)))
    assert list(tree['data']['sampler']['arms']) == ARM_ORDER


@source
@pytest.mark.parametrize('name,conf', [('stage2_v9c.yaml',
                                        's3_base_v9c.yaml')] +
                         [(k, v[0]) for k, v in sorted(OVERLAYS.items())])
def test_target_config_order_equals_the_authored_source_conf(name, conf):
    authored = _read(os.path.join(SOURCE_RECIPE, 'conf', conf))
    arm_order, cell_order = legacy.authored_order(authored)
    assert arm_order == ARM_ORDER  # the constant is the authored order
    sampler = cfg_load.load(os.path.join(CONF, name)).data.sampler
    assert sampler.arm_order == arm_order
    assert sampler.cell_order == cell_order


def test_reordered_arms_compare_unequal():
    base = cfg_load.load(BASE)
    swapped = ARM_ORDER[:]
    swapped[0], swapped[1] = swapped[1], swapped[0]
    for order in (swapped, sorted(ARM_ORDER), ARM_ORDER[::-1]):
        other = schema.build_config(
            _reordered(_base_tree(), 'data.sampler.arms', order))
        assert other.data.sampler.arm_order == order
        assert other.data.sampler.arms == base.data.sampler.arms  # as sets
        assert other != base
        assert other.data.sampler != base.data.sampler
        assert cfg_load.to_dict(other)['data']['sampler']['arm_order'] != \
            cfg_load.to_dict(base)['data']['sampler']['arm_order']
        assert _diff(other, base) == {'data.sampler.arm_order'}
    # the order of policies / per-arm tables has no effect and no field
    same = schema.build_config(_reordered(
        _base_tree(), 'data.sampler.policies',
        sorted(base.data.sampler.policies)))
    assert same == base


def test_reordered_cells_compare_unequal():
    tree = _base_tree()
    tree['data']['sampler']['cells']['vb2_en'] = {
        'arm': 'vb2', 'share': 0.01, 'select': {'spk2lang': 'en'}}
    first = schema.build_config(copy.deepcopy(tree))
    assert first.data.sampler.cell_order == ['vb2_zh', 'vb2_en']
    second = schema.build_config(_reordered(
        tree, 'data.sampler.cells', ['vb2_en', 'vb2_zh']))
    assert second.data.sampler.cell_order == ['vb2_en', 'vb2_zh']
    assert first != second


def test_explicit_arm_order_is_checked_and_applied():
    tree = _base_tree()
    tree['data']['sampler']['arm_order'] = ARM_ORDER[::-1]
    cfg = schema.build_config(tree)
    assert cfg.data.sampler.arm_order == list(cfg.data.sampler.arms) \
        == ARM_ORDER[::-1]
    for bad in (ARM_ORDER[:-1], ARM_ORDER + ['cn3'],
                ARM_ORDER[:-1] + ['vox2']):
        tree = _base_tree()
        tree['data']['sampler']['arm_order'] = bad
        with pytest.raises(ConfigError, match='^data.sampler.arm_order: must '
                           'list every entry of data.sampler.arms exactly '
                           'once'):
            schema.build_config(tree)
    tree = _base_tree()
    tree['data']['sampler']['cell_order'] = ['nope']
    with pytest.raises(ConfigError, match='^data.sampler.cell_order'):
        schema.build_config(tree)
    # a Config edited in place cannot disagree with its order
    cfg = cfg_load.load(BASE)
    cfg.data.sampler.arm_order = sorted(ARM_ORDER)
    with pytest.raises(ConfigError, match='^data.sampler.arm_order: .* is '
                       'not the order of data.sampler.arms'):
        schema.validate(cfg)


def test_order_survives_a_dump_with_sorted_keys():
    cfg = cfg_load.resolve(cfg_load.load(BASE), num_class=5)
    for sort_keys in (False, True):
        text = yaml.safe_dump(cfg_load.to_dict(cfg), sort_keys=sort_keys)
        back = schema.build_config(yaml.safe_load(text))
        assert back == cfg
        assert list(back.data.sampler.arms) == ARM_ORDER
    assert schema.build_config(yaml.safe_load(cfg_load.dump(cfg))) == cfg
    assert 'arm_order' in cfg_load.dump(cfg)


def test_overlay_cannot_reorder_arms_silently(tmp_path):
    base = tmp_path / 'base.yaml'
    base.write_text(open(BASE).read())
    over = tmp_path / 'over.yaml'
    # written in another order, one arm changed, one arm added
    over.write_text(
        'base: base.yaml\n'
        'data:\n'
        '  sampler:\n'
        '    arms:\n'
        '      new: {weight: 0.05, corpus: [new]}\n'
        '      cn1: {weight: 0.2, corpus: [cn1, cn1x]}\n'
        '      vox2: {weight: 0.1184}\n')
    cfg = cfg_load.load(str(over))
    assert cfg.data.sampler.arm_order == ARM_ORDER + ['new']
    assert list(cfg.data.sampler.arms) == ARM_ORDER + ['new']
    assert cfg.data.sampler.arms['cn1'].corpus == ['cn1', 'cn1x']
    # --set behaves the same
    cfg = cfg_load.load(BASE, ['data.sampler.arms.cn2.weight=0.2',
                               'data.sampler.arms.zz={weight: 0.1, '
                               'corpus: [zz]}'])
    assert cfg.data.sampler.arm_order == ARM_ORDER + ['zz']
    # reordering is possible only by naming the order
    over.write_text('base: base.yaml\ndata:\n  sampler:\n    arm_order: %s\n'
                    % sorted(ARM_ORDER))
    assert cfg_load.load(str(over)).data.sampler.arm_order == \
        sorted(ARM_ORDER)
    # a base that names its order (a printed config) refuses an unplaced arm
    base.write_text(cfg_load.dump(cfg_load.load(BASE)))
    over.write_text('base: base.yaml\ndata:\n  sampler:\n    arms:\n'
                    '      new: {weight: 0.05, corpus: [new]}\n')
    with pytest.raises(ConfigError, match=r"^data.sampler.arm_order: .*not "
                       r"listed: \['new'\]"):
        cfg_load.load(str(over))


@source
def test_from_legacy_never_takes_the_order_of_its_input():
    tree = _resolved_v9c()
    saved_order = list(tree['dataset_args']['episodic_sampler']['arms'])
    assert saved_order == sorted(ARM_ORDER) != ARM_ORDER  # dumped sorted
    with pytest.raises(ConfigError) as err:
        legacy.from_legacy(tree)
    text = str(err.value)
    assert 'dataset_args.episodic_sampler.arms: the order of these 8' in text
    assert 'sorted on dump' in text and 'load_legacy' in text
    # also refused for a hand-written conf: the caller states the order
    with pytest.raises(ConfigError, match='the order of these 8'):
        legacy.from_legacy(_source_conf('s3_base_v9c.yaml'))
    reference = legacy.from_legacy(tree, arm_order=ARM_ORDER)
    assert reference.data.sampler.arm_order == ARM_ORDER
    assert list(reference.data.sampler.arms) == ARM_ORDER
    assert reference == legacy.from_legacy(tree, ARM_ORDER, CELL_ORDER)
    alphabetical = legacy.from_legacy(tree, arm_order=saved_order)
    assert alphabetical != reference
    new = cfg_load.resolve(cfg_load.load(BASE), num_class=152367)
    assert _normalised(alphabetical) != _normalised(new)
    assert _normalised(reference) == _normalised(new)
    for bad in (ARM_ORDER[:-1], ARM_ORDER + ['x'], ['vox2'] * 8):
        with pytest.raises(ConfigError, match='the given order .* is not a '
                           'permutation'):
            legacy.from_legacy(tree, arm_order=bad)
    with pytest.raises(ConfigError, match='episodic_sampler.cells: the given '
                       'order'):
        legacy.from_legacy(tree, arm_order=ARM_ORDER, cell_order=['x'])


@source
def test_load_legacy_takes_the_order_from_the_authored_conf(tmp_path):
    cfg = legacy.load_legacy(RESOLVED_V9C, AUTHORED_V9C)
    assert cfg.data.sampler.arm_order == ARM_ORDER
    assert cfg.derived.num_class == 152367
    conf = os.path.join(SOURCE_RECIPE, 'conf')
    # the conf of another run (other arm weights) is refused
    with pytest.raises(ConfigError, match='episodic_sampler.arms differs '
                       r"from the saved config .* at \['cn1', 'cn2', 'vb2'\]"):
        legacy.load_legacy(RESOLVED_V9C,
                           os.path.join(conf, 's3_base_v8c1.yaml'))
    # same arms in the same order: accepted (v9b differs in num_epochs only)
    assert legacy.load_legacy(
        RESOLVED_V9C, os.path.join(conf, 's3_base_v9b.yaml')) == cfg
    empty = tmp_path / 'empty.yaml'
    empty.write_text('seed: 1\n')
    with pytest.raises(ConfigError, match='authored conf: no dataset_args'):
        legacy.load_legacy(RESOLVED_V9C, str(empty))


# ------------------------------------------- (3) errors name the dotted path
BAD_VALUES = (
    # unknown keys
    ('runn', {}, 'runn: unknown key'),
    ('run.sed', 1, 'run.sed: unknown key'),
    ('data.sampler.stratfy', 'systematic',
     'data.sampler.stratfy: unknown key'),
    ('data.sampler.arms.vox2.lang', ['en'],
     'data.sampler.arms.vox2.lang: unknown key'),
    ('data.sampler.policies.cn1.p_hard_pair', 0.5,
     'data.sampler.policies.cn1.p_hard_pair: unknown key'),
    ('data.crop.energy_gate.max_redraw', 4,
     'data.crop.energy_gate.max_redraw: unknown key'),
    ('model.frontend.time_mask.width', 10,
     'model.frontend.time_mask.width: unknown key'),
    ('loss.pair_auc.center.momentun', 0.9,
     'loss.pair_auc.center.momentun: unknown key'),
    # wrong types
    ('run.seed', '716', 'run.seed: must be an integer'),
    ('run.seed', 716.0, 'run.seed: must be an integer'),
    ('run.gpus', '[0,1]', 'run.gpus: must be a list'),
    ('run.gpus', [0, 'a'], r'run.gpus\[1\]: must be an integer'),
    ('engine.tf32', 'false', 'engine.tf32: must be a boolean'),
    ('engine.ddp', True, 'engine.ddp: must be a mapping'),
    ('optim.initial_lr', True, 'optim.initial_lr: must be a number'),
    ('optim.initial_lr', None, 'optim.initial_lr: must be a number'),
    ('data.loader.batch_size', True,
     'data.loader.batch_size: must be an integer'),
    ('data.crop.max_secs', 'x', 'data.crop.max_secs: must be a list'),
    ('data.crop.max_secs', [[2.0, 'a']],
     r'data.crop.max_secs\[0\]\[1\]: must be a number'),
    ('data.sampler.arms', ['vox2'], 'data.sampler.arms: must be a mapping'),
    ('data.sampler.arms.vox2.corpus', 'vox2',
     'data.sampler.arms.vox2.corpus: must be a list'),
    ('data.lists.index', 3, 'data.lists.index: must be a string'),
    # out of range / not a choice
    ('run.num_epochs', 0, 'run.num_epochs: must be >= 1'),
    ('run.gpus', [], 'run.gpus: needs at least 1 entry'),
    ('data.crop.full_prob', 1.5, 'data.crop.full_prob: must be <= 1'),
    ('data.crop.pair_view_secs', [0.5],
     'data.crop.pair_view_secs: needs exactly 2 entries'),
    ('data.sampler.arms.vox2.weight', 0,
     'data.sampler.arms.vox2.weight: must be > 0'),
    ('data.sampler.cells.vb2_zh.share', 1.0,
     'data.sampler.cells.vb2_zh.share: must be < 1'),
    ('data.sampler.stratify', 'choice',
     'data.sampler.stratify: must be one of'),
    ('data.sampler.policies.cn1.pair_policy', 'random',
     'data.sampler.policies.cn1.pair_policy: must be one of'),
    ('data.channel.codecs', ['mp3', 'aac'],
     r'data.channel.codecs\[1\]: must be one of'),
    ('data.augment.probs.noise', -0.1,
     'data.augment.probs.noise: must be >= 0'),
    ('data.augment.arm_probs.cn1.noise', 1.2,
     'data.augment.arm_probs.cn1.noise: must be <= 1'),
    ('model.frontend.time_mask.prob', 0.6,
     'model.frontend.time_mask.prob: must be <= 0.5'),
    ('loss.pair_auc.center.momentum', 1.0,
     'loss.pair_auc.center.momentum: must be < 1'),
    ('optim.initial_lr', float('nan'), 'optim.initial_lr: must be finite'),
    ('engine.amp_dtype', 'float16', 'engine.amp_dtype: must be one of'),
    # checks across fields
    ('data.sampler.P', 20, r'data.sampler.P: P \* K \(40\) must equal'),
    ('data.sampler.cells.vb2_zh.share', 0.5,
     'data.sampler.cells.vb2_zh.share: 0.5 must be below'),
    ('data.sampler.cells.vb2_zh.arm', 'vb3',
     "data.sampler.cells.vb2_zh.arm: 'vb3' is not an arm"),
    ('data.sampler.policies.vox3', {'pair_policy': 'xsession'},
     "data.sampler.policies: 'vox3' is not an arm"),
    ('data.sampler.policies.kespeech.subdialect_file', None,
     'data.sampler.policies.kespeech.subdialect_file: set exactly when'),
    ('data.sampler.policies.mandarin_t1.spk_weight_file', None,
     'data.sampler.policies.mandarin_t1.spk_weight_share: needs'),
    ('data.crop.arm_caps.cn3', [[2.0, 1.0]],
     "data.crop.arm_caps: 'cn3' is not an arm"),
    ('data.crop.max_secs', [[2.0, 0.5, 1.0]],
     r'data.crop.max_secs\[0\]: must be \[seconds > 0, weight > 0\]'),
    ('data.augment.probs.overlap', 0.1,
     'data.augment.probs.overlap: unknown augmentation type'),
    ('data.augment.probs.noise', 0.9,
     'data.augment.probs: probabilities sum to'),
    ('data.augment.reverb_weights', [1.0],
     'data.augment.reverb_weights: needs one weight per entry'),
    ('data.channel.arms', ['t1'], "data.channel.arms: 't1' is not an arm"),
    ('data.channel.kpos', [2], 'data.channel.kpos: positions must be below'),
    ('data.zero_run_dither.arms', ['nope'],
     "data.zero_run_dither.arms: 'nope' is not an arm"),
    ('data.filter.min_sec', 9.0, 'data.filter.min_sec: 9 must be below'),
    ('model.backend.nb_layer', 25,
     'model.backend.nb_layer: 25 must be model.frontend.num_layers'),
    ('loss.pair_auc.center.arm', 'vox1',
     "loss.pair_auc.center.arm: 'vox1' is not an arm"),
    ('optim.final_lr', 1.0, 'optim.final_lr: 1 exceeds'),
    ('run.num_avg', 15, 'run.num_avg: 15 exceeds'),
    ('derived', {'epoch_iter': 3000, 'scale_ratio': 6.0, 'num_class': 5},
     'derived.epoch_iter: 3000 does not match'),
)


@pytest.mark.parametrize('dotted,value,message', BAD_VALUES,
                         ids=['%s=%r' % (b[0], b[1]) for b in BAD_VALUES])
def test_invalid_value_names_the_dotted_path(dotted, value, message):
    tree = _base_tree()
    _set(tree, dotted, value)
    with pytest.raises(ConfigError, match='^' + message):
        schema.build_config(tree)


@pytest.mark.parametrize('dotted', [
    'run', 'run.seed', 'data.lists.train_label', 'data.sampler.arms',
    'data.sampler.arms.cn1.corpus', 'data.crop.max_secs',
    'data.channel.codecs', 'model.frontend.time_mask.span',
    'loss.pair_auc.center', 'optim.grad_clip', 'engine.ddp.static_graph'])
def test_missing_key_names_the_dotted_path(dotted):
    tree = _base_tree()
    _delete(tree, dotted)
    with pytest.raises(ConfigError, match='^%s: required key is missing'
                       % dotted):
        schema.build_config(tree)


def test_optional_sections_have_documented_defaults():
    tree = _base_tree()
    for dotted in ('data.sampler.cells', 'data.sampler.policies',
                   'data.crop.arm_caps', 'data.crop.energy_gate',
                   'data.crop.pair_view_onset_prob',
                   'data.augment.arm_probs',
                   'data.augment.noise_exclude_list',
                   'data.zero_run_dither.arms'):
        _delete(tree, dotted)
    cfg = schema.build_config(tree)
    assert cfg.data.sampler.cells == {} and cfg.data.sampler.policies == {}
    assert cfg.data.crop.energy_gate is None
    assert cfg.data.lists.index_root is None
    policy = cfg_load.load(BASE).data.sampler.policies['mandarin_t1']
    assert (policy.p_dialect, policy.segment_first, policy.w_bridge) == \
        (0.5, True, 1.0)


# ----------------------------------- (4) closed-experiment keys are refused
@source
@pytest.mark.parametrize('conf,dotted', CLOSED_EXAMPLES,
                         ids=['%s:%s' % c for c in CLOSED_EXAMPLES])
def test_legacy_refuses_closed_key(conf, dotted):
    """The value a real config gives a closed key, transplanted into the
    accepted reference config, is refused by name."""
    donor = _read(os.path.join(SOURCE_RECIPE, 'conf', conf))
    concrete = _expand(donor, dotted)
    value = _get(donor, concrete)
    assert value not in (None, False, 0, {}, [])
    tree = _resolved_v9c()
    _from_legacy(tree)  # accepted before the transplant
    _set(tree, concrete, copy.deepcopy(value))
    with pytest.raises(ConfigError) as err:
        _from_legacy(tree)
    lines = [line.strip() for line in str(err.value).splitlines()]
    assert any(line.startswith(concrete + ': closed-experiment key is set')
               for line in lines), str(err.value)


@source
@pytest.mark.parametrize('conf', sorted({c for c, _ in CLOSED_EXAMPLES}))
def test_legacy_refuses_closed_experiment_conf(conf):
    donor = _read(os.path.join(SOURCE_RECIPE, 'conf', conf))
    wanted = [_expand(donor, d) for c, d in CLOSED_EXAMPLES if c == conf]
    tree = _source_conf(conf)
    sampler = tree['dataset_args'].get('episodic_sampler') or {}
    with pytest.raises(ConfigError) as err:
        legacy.from_legacy(tree, arm_order=list(sampler.get('arms') or {}),
                           cell_order=list(sampler.get('cells') or {}))
    for concrete in wanted:
        assert concrete + ': closed-experiment key is set' in str(err.value)


CLOSED_SYNTHETIC = (
    # closed keys no stored config sets: values in the original's format
    ('dataset_args.w2vbert_sv_args.cmvn_shrink',
     {'n0_frames': 100, 'stats': 'stats.npz'}),
    ('dataset_args.episodic_sampler.arms.vb2.lang', ['zh']),
    ('dataset_args.episodic_sampler.max_utts_per_spk_per_pass', 20),
    ('dataset_args.episodic_sampler.construction.arms.3dspeaker.p_farfar',
     0.3),
    ('dataset_args.episodic_sampler.construction.arms.cn2.w_bridge_hard',
     3.0),
    ('dataset_args.episodic_sampler.cells.vb2_zh.select.genre', 'drama'),
    ('dataset_args.aug_probs.mixup', 0.1),
    ('dataset_args.aug_arm_probs.cn1.overlap', 0.1),
    ('dataset_args.asym_channel.codec_prob', 0.5),
    ('dataset_args.var_crop.par_jitter_sec', 1.0),
    ('scheduler_args.frontend_layer_decay', 0.9),
    ('grad_norm_adam', {'enable': True}),
    ('model_init', 'exp/x/model.pt'),
    ('do_lm', True),
    ('allow_projection_resize', True),
    ('llrd', {'decay': 0.9}),
)


@source
@pytest.mark.parametrize('dotted,value', CLOSED_SYNTHETIC,
                         ids=[c[0] for c in CLOSED_SYNTHETIC])
def test_legacy_refuses_closed_key_synthetic(dotted, value):
    tree = _resolved_v9c()
    _set(tree, dotted, value)
    with pytest.raises(ConfigError, match=dotted.replace('.', r'\.')
                       + ': closed-experiment key is set'):
        _from_legacy(tree)


@source
@pytest.mark.parametrize('dotted,value', [
    ('consistency_target', 'same'), ('pair_auc.grad', 'both'),
    ('pair_auc.cross_gpu', False), ('model_init', None), ('l2sp', None),
    ('dataset_args.episodic_sampler.cond_groups', None),
    ('dataset_args.aug_probs.mixup', 0.0),
    ('dataset_args.var_crop.pair_law', None),
    ('grad_norm_adam', {'enable': False})])
def test_legacy_accepts_closed_key_at_its_off_value(dotted, value):
    tree = _resolved_v9c()
    reference = _from_legacy(tree)
    _set(tree, dotted, value)
    assert _from_legacy(tree) == reference


@source
@pytest.mark.parametrize('dotted,value', [
    ('enable_amp', False), ('data_type', 'raw'), ('loss', 'FocalLoss'),
    ('dataset_args.aug_mode', 'legacy'), ('dataset_args.aug_prob', 0.0),
    ('dataset_args.arm_logging', False), ('dataset_args.cmvn', True),
    ('dataset_args.speed_perturb', True), ('dataset_args.spec_aug', True),
    ('dataset_args.filter', False), ('dataset_args.frontend', 'fbank'),
    ('dataset_args.episodic_sampler.cross_video', False),
    ('dataset_args.episodic_sampler.spk_weighting', 'utts'),
    ('dataset_args.w2vbert_sv_args.frozen', True),
    ('dataset_args.w2vbert_sv_args.feature_extractor', 'hf'),
    ('projection_args.project_type', 'arc_margin'),
    ('projection_args.easy_margin', True),
    ('margin_update.update_margin', True)])
def test_legacy_refuses_unsupported_fixed_value(dotted, value):
    tree = _resolved_v9c()
    _set(tree, dotted, value)
    with pytest.raises(ConfigError, match=dotted.replace('.', r'\.')
                       + ': .* is not supported by the recipe'):
        _from_legacy(tree)


@source
def test_legacy_reports_every_problem_at_once():
    tree = _resolved_v9c()
    tree['grpo'] = {'weight': 1.0}
    tree['head_spk_chek'] = 'required'
    del tree['seed']
    tree['margin_update']['final_margin'] = 0.3
    with pytest.raises(ConfigError) as err:
        _from_legacy(tree)
    text = str(err.value)
    assert '4 problems' in text
    for part in ('grpo: closed-experiment key is set',
                 'head_spk_chek: unknown key',
                 'seed: required key is missing',
                 'margin_update.final_margin: 0.3 differs'):
        assert part in text


# ------------------------------------------------------- (5) the core view
def _contains_key(tree, name):
    if isinstance(tree, dict):
        return name in tree or any(_contains_key(v, name)
                                   for v in tree.values())
    return False


def test_core_view_content():
    cfg = cfg_load.load(BASE)
    core = core_view.to_core_dict(cfg)
    assert set(core) == {'model', 'model_args', 'dataset_args'}
    assert not _contains_key(core, 'time_mask')
    ds = core['dataset_args']
    assert set(ds) == {'frontend', 'frontend_num_layers', 'w2vbert_sv_args',
                       'resample_rate', 'num_frms', 'cmvn'}
    assert ds['cmvn'] is False and ds['num_frms'] == 150
    assert ds['frontend_num_layers'] == 22
    assert core['model_args']['nb_layer'] == 23
    assert yaml.safe_load(yaml.safe_dump(core)) == core  # plain YAML types


def test_core_view_is_accepted_by_the_stock_extraction_reader():
    """Replays what wespeaker/bin/extract.py reads from its config."""
    pytest.importorskip('torch')
    from wespeaker.frontend import frontend_class_dict
    from wespeaker.models.speaker_model import get_speaker_model
    from wespeaker.utils.frontend_utils import maybe_truncate_frontend

    core = core_view.to_core_dict(cfg_load.load(BASE))
    # command-line keys of tools/extract_embedding.sh on top of the file
    configs = dict(core, model_path='m.pt', embed_ark='e.ark',
                   data_type='shard', data_list='l', batch_size=1,
                   num_workers=1)
    test_conf = copy.deepcopy(configs['dataset_args'])
    frontend_type = test_conf.get('frontend', 'fbank')
    assert frontend_type == 'w2vbert_sv'
    frontend_cls = frontend_class_dict[frontend_type]
    inspect.signature(frontend_cls.__init__).bind(
        None, **test_conf[frontend_type + '_args'],
        sample_rate=test_conf['resample_rate'])
    model_cls = get_speaker_model(configs['model'])
    inspect.signature(model_cls.__init__).bind(None, **configs['model_args'])
    # the truncation hook reads these two; a mismatch would assert there
    source_text = inspect.getsource(maybe_truncate_frontend)
    assert "dataset_args.get('frontend_num_layers'" in source_text
    assert configs['model_args'].get('nb_layer') == \
        int(test_conf['frontend_num_layers']) + 1
    # ragged whole-utterance batching needs cmvn off in the file
    assert not test_conf.get('cmvn', True)
    assert not test_conf.get('spec_aug', False)
    # every key extract.py reads from the file (not the command line) exists
    with open(os.path.join(REPO_ROOT, 'wespeaker', 'bin', 'extract.py')) as f:
        text = f.read()
    for key in ("configs['dataset_args']", "configs['model']",
                "configs['model_args']"):
        assert key in text


@source
def test_core_view_equals_the_legacy_values():
    tree = _resolved_v9c()
    core = core_view.to_core_dict(_from_legacy(tree))
    assert core['model'] == tree['model']
    assert core['model_args'] == tree['model_args']
    old_ds, new_ds = tree['dataset_args'], core['dataset_args']
    for key in ('frontend', 'frontend_num_layers', 'resample_rate',
                'num_frms', 'cmvn'):
        assert new_ds[key] == old_ds[key], key
    old_fe = dict(old_ds['w2vbert_sv_args'])
    assert old_fe.pop('time_mask') == {'prob': 0.05, 'span': 10}
    old_fe['upstream_args'] = {
        'path_or_url': old_fe['upstream_args']['path_or_url']}
    assert new_ds['w2vbert_sv_args'] == old_fe


# --------------------------------------------------- (6) derived quantities
def test_derived_values_follow_the_trainer_formulas(tmp_path):
    cfg = cfg_load.load(BASE)
    world, batch = len(cfg.run.gpus), cfg.data.loader.batch_size
    assert (world, batch) == (8, 48)
    # wespeaker/bin/train.py: sample_num_per_epoch // world_size // batch_size
    assert schema.epoch_iter(cfg) == 1400000 // world // batch == 3645
    # ... and scale_ratio = 1.0 * world_size * batch_size / 64
    assert schema.scale_ratio(cfg) == 1.0 * world * batch / 64 == 6.0
    label = tmp_path / 'utt2spk'
    label.write_text('u1 spkB\nu2 spkA\n\nu3 spkB\nu4 spkC extra\n')
    resolved = cfg_load.resolve(
        cfg_load.load(BASE, ['data.lists.train_label=%s' % label]))
    assert resolved.derived == schema.DerivedCfg(3645, 6.0, 3)
    relative = cfg_load.resolve(
        cfg_load.load(BASE, ['data.lists.train_label=utt2spk']),
        root=str(tmp_path))
    assert relative.derived.num_class == 3
    assert cfg_load.resolve(resolved).derived == resolved.derived
    assert schema.build_config(cfg_load.to_dict(resolved)) == resolved
    four = cfg_load.resolve(cfg_load.load(BASE, ['run.gpus=[0, 1, 2, 3]']),
                            num_class=7)
    assert four.derived == schema.DerivedCfg(7291, 3.0, 7)


def test_resolve_errors(tmp_path):
    cfg = cfg_load.load(BASE)
    with pytest.raises(ConfigError, match='^data.lists.train_label: cannot '
                       'read'):
        cfg_load.resolve(cfg, root=str(tmp_path))
    (tmp_path / 'data' / 'train').mkdir(parents=True)
    bad = tmp_path / 'data' / 'train' / 'utt2spk'
    bad.write_text('u1 spkA\nu2\n')
    with pytest.raises(ConfigError, match='line 2 has no speaker column'):
        cfg_load.resolve(cfg, root=str(tmp_path))
    with pytest.raises(ConfigError, match='^derived.num_class'):
        cfg_load.resolve(cfg, num_class=0)
    resolved = cfg_load.resolve(cfg, num_class=10)
    with pytest.raises(ConfigError, match='^derived.num_class: 10 in the '
                       'config, 11 computed'):
        cfg_load.resolve(resolved, num_class=11)


@source
def test_legacy_derived_values_are_validated():
    tree = _resolved_v9c()
    cfg = _from_legacy(tree)
    assert cfg.derived == schema.DerivedCfg(
        epoch_iter=3645, scale_ratio=6.0, num_class=152367)
    assert cfg.derived.epoch_iter == tree['scheduler_args']['epoch_iter'] \
        == tree['margin_update']['epoch_iter']
    assert cfg.derived.scale_ratio == tree['scheduler_args']['scale_ratio']
    assert cfg.derived.num_class == tree['projection_args']['num_class']
    for dotted, value in (('scheduler_args.epoch_iter', 3646),
                          ('margin_update.epoch_iter', 3000),
                          ('scheduler_args.scale_ratio', 1.0),
                          ('scheduler_args.num_epochs', 9),
                          ('optimizer_args.lr', 1e-4)):
        broken = copy.deepcopy(tree)
        _set(broken, dotted, value)
        with pytest.raises(ConfigError, match=dotted.replace('.', r'\.')):
            _from_legacy(broken)
    # a different process count changes both derived values
    broken = copy.deepcopy(tree)
    broken['gpus'] = [0, 1, 2, 3]
    with pytest.raises(ConfigError, match='scheduler_args.epoch_iter: 3645 '
                       'does not match the value 7291'):
        _from_legacy(broken)
    unresolved = _from_legacy(_source_conf('s3_base_v9c.yaml'))
    assert unresolved.derived is None
    assert cfg_load.resolve(unresolved, num_class=152367).derived == \
        cfg.derived


@source
def test_num_class_equals_the_speakers_of_the_training_list():
    tree = _resolved_v9c()
    label = os.path.join(SOURCE_RECIPE, tree['train_label'])
    if not os.path.isfile(label):
        pytest.skip('training list %s is not available' % tree['train_label'])
    cfg = cfg_load.load(BASE, ['data.lists.train_label=%s' % label])
    assert cfg_load.resolve(cfg).derived.num_class == \
        tree['projection_args']['num_class'] == 152367


# ------------------------------------------------------------ command line
def test_validate_config_cli(capsys):
    from ovsb.bin import validate_config
    assert validate_config.main([BASE, '--view', 'none']) == 0
    assert validate_config.main(
        [os.path.join(CONF, 'cont_c40.yaml'), '--num_class', '9']) == 0
    out = yaml.safe_load(capsys.readouterr().out)
    assert out['derived'] == {'epoch_iter': 3645, 'scale_ratio': 6.0,
                              'num_class': 9}
    assert out['optim']['initial_lr'] == 5e-06
    assert validate_config.main([BASE, '--view', 'core']) == 0
    assert 'time_mask' not in capsys.readouterr().out
    assert validate_config.main([BASE, '--set', 'run.sed=1']) == 2
    assert 'run.sed: unknown key' in capsys.readouterr().err


@source
def test_validate_config_cli_legacy(capsys):
    from ovsb.bin import validate_config
    assert validate_config.main([RESOLVED_V9C, '--legacy', AUTHORED_V9C]) == 0
    out = yaml.safe_load(capsys.readouterr().out)
    assert out['data']['sampler']['arm_order'] == ARM_ORDER
    assert list(out['data']['sampler']['arms']) == ARM_ORDER


def test_rank_split_arms_default_and_check():
    assert cfg_load.load(BASE).data.sampler.rank_split_arms == []
    tree = _base_tree()
    tree['data']['sampler']['rank_split_arms'] = ['cn1']
    assert schema.build_config(tree).data.sampler.rank_split_arms == ['cn1']
    for bad in (['nope'], ['cn1', 'cn1']):
        tree = _base_tree()
        tree['data']['sampler']['rank_split_arms'] = bad
        with pytest.raises(ConfigError,
                           match='^data.sampler.rank_split_arms'):
            schema.build_config(tree)
    split = cfg_load.load(os.path.join(CONF, 'cont_c40_ranksplit.yaml'))
    c40 = cfg_load.load(os.path.join(CONF, 'cont_c40.yaml'))
    assert _diff(split, c40) == {'run.exp_dir',
                                 'data.sampler.rank_split_arms'}
