"""Import a resolved config of the original (wespeaker-schema) training tree.

:func:`from_legacy` maps such a config onto the recipe :class:`Config`. Every
legacy key falls in one of four classes, declared per mapping below:

* live    - mapped to a recipe field;
* fixed   - the recipe implements one value; any other value is refused;
* ignored - has no effect on this training path (or is a run-time pointer);
* closed  - belongs to a feature the recipe does not implement; refused as
  soon as it is set to anything but its off value.

A key in none of the classes is refused as unknown. All problems are
collected and reported in one :class:`ConfigError`, each with the dotted
path of the legacy key.
"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import yaml

from ovsb.config import schema
from ovsb.config.core_view import FRAME_MS, FRONTEND
from ovsb.config.schema import Config, ConfigError

Accept = Union[Tuple[Any, ...], Callable[[Any], bool]]


def _any(value: Any) -> bool:
    return True


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def _not_enabled(value: Any) -> bool:
    return not value or (isinstance(value, Mapping)
                         and not value.get('enable', False))


# off values of a closed key: absent, null, false, 0, empty
OFF = (None, False, 0, {}, [], '')
_REQUIRED = object()

PROJECT_TYPE = 'arc_margin_intertopk_subcenter'

TOP_FIXED: Dict[str, Accept] = {
    'data_type': ('shard',),
    'enable_amp': (True,),
    'loss': ('CrossEntropyLoss',),
    'loss_args': (None, {}),
    'margin_scheduler': ('MarginScheduler',),
}
TOP_IGNORED = ('train_data', 'checkpoint')
TOP_CLOSED: Dict[str, Accept] = dict(
    {k: OFF for k in (
        'allow_projection_resize', 'consistency_enrol', 'do_lm', 'ema_decay',
        'ema_warmup', 'ema_consistency', 'embed_kl', 'genre_adversary',
        'grad_clip_on_scaled_grads', 'grpo', 'keep_all_train_states', 'l2sp',
        'layer_gate_l1', 'llrd', 'margin_schedule', 'mixture_controller',
        'model_init', 'optimizer_groups', 'par_consistency_weight',
        'profile_steps', 'ptp', 'ptp_v2', 'tail', 'train_lmdb')},
    consistency_target=(None, 'same'), grad_norm_adam=_not_enabled)

DATASET_FIXED: Dict[str, Accept] = {
    'arm_logging': (True,),
    'aug_mode': ('mixed',),
    'aug_prob': _positive,
    'cmvn': (False,),
    'filter': (True,),
    'frontend': (FRONTEND,),
    'speed_perturb': (False,),
    'spec_aug': (False,),
}
DATASET_IGNORED = ('cmvn_args', 'shuffle', 'shuffle_args', 'spec_aug_args',
                   'spk_alias')
DATASET_CLOSED: Dict[str, Accept] = dict(
    {k: OFF for k in (
        'channel_args', 'chunk_info_args', 'farfield_args', 'fbank_args',
        'frontend_args', 'interferer_args', 'max_num_frames',
        'min_num_frames', 'num_frms_choices', 'omniasr_sv_args',
        'overlap_noise_prob', 'raw_wav', 'resample_args',
        'speed_perturb_args', 'speed_perturb_arms', 'speed_perturb_mode',
        'speech_snr', 'token_batching', 'utt_weights')},
    crop_sync_ranks=(None, True), episodic_source_rate=(None, 16000))

SAMPLER_FIXED: Dict[str, Accept] = {
    'cross_video': (True,),
    'spk_weighting': ('uniform',),
}
SAMPLER_IGNORED = ('spk2category',)
SAMPLER_CLOSED = {k: OFF for k in ('block', 'cond_groups',
                                   'max_utts_per_spk_per_pass')}
ARM_CLOSED = {k: OFF for k in ('lang', 'not_lang', 'category',
                               'not_category')}
CELL_CLOSED = {'row_floor': OFF}
CELL_SELECT_CLOSED = {'genre': OFF}
CONSTRUCTION_CLOSED = {k: OFF for k in ('dcls_edges_m', 'parallel_view',
                                        'pass_cap', 'utt2info')}
POLICY_CLOSED = {k: OFF for k in (
    'far_min_m', 'genre_only_weight', 'hard_genres', 'p_farfar',
    'p_hard_pair', 'p_same_session', 'spoken_genres', 'w_bridge_hard')}
CROP_CLOSED = {k: OFF for k in ('pair_law', 'par_jitter_sec')}
AUG_CLOSED = {k: OFF for k in ('farfield', 'mixup', 'overlap')}
CHANNEL_CLOSED = {k: OFF for k in ('codec_prob', 'isolate_dump_dir',
                                   'isolate_timeout')}
FRONTEND_FIXED: Dict[str, Accept] = {
    'feature_extractor': ('torch',),
    'frozen': (False,),
    'frame_shift': (FRAME_MS,),
    'frame_length': (FRAME_MS,),
    'max_input_seconds': (10, None),
}
FRONTEND_IGNORED = ('download_dir',)
FRONTEND_CLOSED = {'cmvn_shrink': OFF}
LOADER_CLOSED: Dict[str, Accept] = {'persistent_workers': OFF,
                                    'in_order': (None, True)}
MODEL_CLOSED = {k: OFF for k in ('layer_gates', 'per_head', 'ph_fc_dim',
                                 'ph_key_dim', 'pre_layers', 'quality_coord',
                                 'quality_init_std')}
PROJECTION_FIXED: Dict[str, Accept] = {
    'project_type': (PROJECT_TYPE,),
    'easy_margin': (False,),
}
PROJECTION_CLOSED = {k: OFF for k in ('curvature', 'do_lm',
                                      'k_per_class_file', 'lanbuda',
                                      'margin_type', 't')}
MARGIN_FIXED: Dict[str, Accept] = {'update_margin': (False,)}
MARGIN_IGNORED = ('epoch_iter', 'fix_start_epoch', 'increase_start_epoch',
                  'increase_type')
SCHEDULER_CLOSED = {'frontend_layer_decay': OFF}
PAIR_AUC_CLOSED: Dict[str, Accept] = {
    'cross_gpu': OFF, 'logn_correction': OFF,
    'grad': (None, 'both'), 'pool_block': (None, 'none')}


def _accepts(accept: Accept, value: Any) -> bool:
    if callable(accept):
        return bool(accept(value))
    # compare by type as well: 0 == False and 1 == True in Python
    return any(value == a and isinstance(value, bool) == isinstance(a, bool)
               for a in accept)


class _Node:
    """A legacy mapping being read; records every key it is asked for."""

    def __init__(self, tree: Any, path: str, problems: List[str],
                 absent: bool = False):
        self.path = path
        self.problems = problems
        self.absent = absent  # already reported: stay quiet about its keys
        if tree is None:
            tree = {}
        if not isinstance(tree, Mapping):
            problems.append('%s: must be a mapping, got %r' % (path, tree))
            tree = {}
        self.tree = dict(tree)
        self.seen = set()

    def where(self, key: str) -> str:
        return '%s.%s' % (self.path, key) if self.path else str(key)

    def get(self, key: str, default: Any = _REQUIRED) -> Any:
        """Value of a live key; ``default`` is the original tree's default."""
        self.seen.add(key)
        if key in self.tree:
            return self.tree[key]
        if default is not _REQUIRED:
            return default
        if not self.absent:
            self.problems.append('%s: required key is missing' %
                                 self.where(key))
        return None

    def has(self, key: str) -> bool:
        self.seen.add(key)
        return self.tree.get(key) is not None

    def child(self, key: str, required: bool = True) -> '_Node':
        value = self.get(key) if required else self.get(key, None)
        return _Node(value, self.where(key), self.problems,
                     absent=required and value is None)

    def done(self, fixed: Optional[Mapping[str, Accept]] = None,
             ignored: Tuple[str, ...] = (),
             closed: Optional[Mapping[str, Accept]] = None) -> None:
        """Classify the keys that were not read as live."""
        fixed, closed = fixed or {}, closed or {}
        for key, value in self.tree.items():
            if key in self.seen or key in ignored:
                continue
            if key in closed:
                if not _accepts(closed[key], value):
                    self.problems.append(
                        '%s: closed-experiment key is set (%r); the recipe '
                        'does not implement it' % (self.where(key), value))
            elif key in fixed:
                if not _accepts(fixed[key], value):
                    self.problems.append(
                        '%s: %r is not supported by the recipe (supported: '
                        '%s)' % (self.where(key), value, _describe(
                            fixed[key])))
            else:
                self.problems.append('%s: unknown key' % self.where(key))


def _describe(accept: Accept) -> str:
    if callable(accept):
        return 'a positive number'
    return ' | '.join(repr(a) for a in accept if a is not None)


def _as_list(value: Any) -> Any:
    """``corpus: vox2`` and ``corpus: [vox2]`` mean the same."""
    return [value] if isinstance(value, str) else value


def _gpus(value: Any) -> Any:
    """``gpus`` is a list, or its text form in an unresolved config."""
    if isinstance(value, str):
        try:
            return yaml.safe_load(value)
        except yaml.YAMLError:
            return value
    return value


def _frames_to_sec(frames: Any) -> Any:
    if isinstance(frames, (int, float)) and not isinstance(frames, bool):
        return frames * FRAME_MS / 1000.0
    return frames


def _sampler(ds: _Node) -> Dict[str, Any]:
    es = ds.child('episodic_sampler')
    arms = {}
    arms_node = es.child('arms')
    for name in arms_node.tree:
        arm = arms_node.child(name)
        arms[name] = {'weight': arm.get('weight'),
                      'corpus': _as_list(arm.get('corpus'))}
        arm.done(closed=ARM_CLOSED)
    cells = {}
    cells_node = es.child('cells', required=False)
    for name in cells_node.tree:
        cell = cells_node.child(name)
        select = cell.child('select')
        cells[name] = {'arm': cell.get('arm'), 'share': cell.get('share'),
                       'select': {'spk2lang': select.get('spk2lang')}}
        select.done(closed=CELL_SELECT_CLOSED)
        cell.done(closed=CELL_CLOSED)
    cons = es.child('construction')
    policies = {}
    pol_node = cons.child('arms', required=False)
    for name, raw in pol_node.tree.items():
        pol_node.seen.add(name)
        if not raw:  # an arm listed without a policy has none
            continue
        pol = pol_node.child(name)
        policy = {'pair_policy': pol.get('pair_policy', 'xsession'),
                  'p_dialect': pol.get('p_dialect', 0.5),
                  'segment_first': pol.get('segment_first', True),
                  'w_bridge': pol.get('w_bridge', None) or 1.0}
        for key in ('subdialect_file', 'spk_weight_file', 'spk_weight_share'):
            if pol.has(key):
                policy[key] = pol.get(key)
        if pol.has('dialect_mass'):
            mass = pol.child('dialect_mass')
            policy['dialect_mass'] = {
                'boost': mass.get('boost', 3.0),
                'cap_x_median': mass.get('cap_x_median', 8)}
            mass.done()
        pol.done(closed=POLICY_CLOSED)
        policies[name] = policy
    out = {
        'P': es.get('P'), 'K': es.get('K', 2),
        'sample_num_per_epoch': ds.get('sample_num_per_epoch'),
        'vox2_lang': es.get('vox2_lang', 'en'),
        'stratify': es.get('stratify'),
        'log_every': es.get('log_every', 200),
        'arms': arms, 'cells': cells,
        'forbid_same_segment': cons.get('forbid_same_segment', False),
        'policies': policies,
    }
    cons.done(closed=CONSTRUCTION_CLOSED)
    lists = {'index': es.get('index'), 'spk2corpus': es.get('spk2corpus'),
             'spk2lang': es.get('spk2lang'),
             'index_root': es.get('root', None)}
    es.done(fixed=SAMPLER_FIXED, ignored=SAMPLER_IGNORED,
            closed=SAMPLER_CLOSED)
    return {'sampler': out, 'lists': lists}


def _crop(ds: _Node) -> Dict[str, Any]:
    vc = ds.child('var_crop')
    out = {
        'max_secs': vc.get('max_secs'),
        'min_sec': vc.get('min_sec', 0.5),
        'full_prob': vc.get('full_prob', 0.5),
        'onset_prob': vc.get('onset_prob', 0.25),
        'onset_max_sec': vc.get('onset_max_sec', 1.0),
        'pair_prob': vc.get('pair_prob', 0.0),
        'pair_anchor_sec': vc.get('pair_anchor_sec', 6.0),
        'pair_view_secs': vc.get('pair_view_secs'),
        'pair_view_onset_prob': vc.get('pair_view_onset_prob', None),
        'arm_caps': vc.get('arm_caps', None) or {},
    }
    if vc.has('energy_gate') and vc.tree['energy_gate']:
        gate = vc.child('energy_gate')
        out['energy_gate'] = {
            'gate_below_sec': gate.get('gate_below_sec', 1.5),
            'min_active_frac': gate.get('min_active_frac', 0.5),
            'max_redraws': gate.get('max_redraws', 4)}
        gate.done()
    vc.done(closed=CROP_CLOSED)
    return out


def _augment(top: _Node, ds: _Node) -> Dict[str, Any]:
    probs = ds.child('aug_probs')
    out_probs = {k: probs.get(k) for k in schema.AUG_TYPES if k in probs.tree}
    probs.done(closed=AUG_CLOSED)
    arm_probs = {}
    per_arm = ds.child('aug_arm_probs', required=False)
    for name in per_arm.tree:
        node = per_arm.child(name)
        arm_probs[name] = {k: node.get(k) for k in schema.AUG_TYPES
                           if k in node.tree}
        node.done(closed=AUG_CLOSED)
    return {
        'probs': out_probs, 'arm_probs': arm_probs,
        'reverberant_noise': ds.get('reverberant_noise', False),
        'reverb_sources': ds.get('reverb_data'),
        'reverb_weights': ds.get('reverb_weights'),
        'noise_lmdb': top.get('noise_data'),
        'reverb_lmdb': top.get('reverb_data'),
        'noise_exclude_list': ds.get('noise_exclude_list', None),
    }


def _channel(ds: _Node) -> Dict[str, Any]:
    ch = ds.child('asym_channel')
    out = {
        'arms': ch.get('arms'), 'prob': ch.get('prob'),
        'kpos': ch.get('kpos', [1]), 'distinct': ch.get('distinct', True),
        'codecs': ch.get('codecs', list(schema.CODECS)),
        'eq_prob': ch.get('eq_prob', 0.6),
        'eq_gain_db': ch.get('eq_gain_db', 6.0),
        'bandlimit_prob': ch.get('bandlimit_prob', 0.4),
        'clip_prob': ch.get('clip_prob', 0.3),
        'clip_level': ch.get('clip_level', [0.3, 0.7]),
        'loss_prob': ch.get('loss_prob', 0.2),
        'loss_rate': ch.get('loss_rate', [0.02, 0.08]),
        'isolate_ffmpeg': ch.get('isolate_ffmpeg', False),
    }
    ch.done(closed=CHANNEL_CLOSED)
    return out


def _data(top: _Node, ds: _Node) -> Dict[str, Any]:
    out = _sampler(ds)
    out['lists']['train_label'] = top.get('train_label')
    flt = ds.child('filter_args')
    out['filter'] = {'min_sec': _frames_to_sec(flt.get('min_num_frames')),
                     'max_sec': _frames_to_sec(flt.get('max_num_frames'))}
    flt.done()
    out['crop'] = _crop(ds)
    zr = ds.child('zero_run_dither')
    out['zero_run_dither'] = {
        'arms': zr.get('arms', None),
        'min_run_ms': zr.get('min_run_ms', 50.0),
        'lsb_level': zr.get('lsb_level', 4),
        'dither_lsb': zr.get('dither_lsb', 1.0)}
    zr.done()
    out['augment'] = _augment(top, ds)
    out['channel'] = _channel(ds)
    loader = top.child('dataloader_args')
    out['loader'] = {k: loader.get(k) for k in (
        'batch_size', 'num_workers', 'prefetch_factor', 'pin_memory',
        'drop_last')}
    loader.done(closed=LOADER_CLOSED)
    out['resample_rate'] = ds.get('resample_rate')
    out['num_frms'] = ds.get('num_frms')
    return out


def _model(top: _Node, ds: _Node) -> Dict[str, Any]:
    fe = ds.child(FRONTEND + '_args')
    upstream = fe.child('upstream_args')
    mask = fe.child('time_mask')
    frontend = {
        'assets': upstream.get('path_or_url'),
        'num_layers': ds.get('frontend_num_layers'),
        'gradient_checkpointing': fe.get('gradient_checkpointing', False),
        'time_mask': {'prob': mask.get('prob'),
                      'span': mask.get('span', 10)},
    }
    mask.done()
    upstream.done(ignored=('name',))
    fe.done(fixed=FRONTEND_FIXED, ignored=FRONTEND_IGNORED,
            closed=FRONTEND_CLOSED)
    margs = top.child('model_args')
    backend = {'name': top.get('model')}
    backend.update({k: margs.get(k) for k in (
        'head_nb', 'embed_dim', 'compression_dim', 'nb_layer', 'feat_dim')})
    backend['feature_grad_mult'] = margs.get('feature_grad_mult', 1.0)
    margs.done(closed=MODEL_CLOSED)
    proj = top.child('projection_args')
    margin = top.child('margin_update')
    dur = proj.child('duration_margin')
    head = {'scale': proj.get('scale'), 'K': proj.get('K'),
            'k_top': proj.get('k_top'), 'mp': proj.get('mp'),
            'margin': margin.get('initial_margin'),
            'duration_margin_ref_sec': dur.get('ref_sec')}
    dur.done()
    if margin.get('final_margin') != head['margin']:
        top.problems.append(
            'margin_update.final_margin: %r differs from initial_margin %r; '
            'the recipe trains with a constant margin' %
            (margin.tree.get('final_margin'), head['margin']))
    margin.done(fixed=MARGIN_FIXED, ignored=MARGIN_IGNORED)
    embed = proj.get('embed_dim', None)
    if embed is not None and embed != backend['embed_dim']:
        top.problems.append('projection_args.embed_dim: %r differs from '
                            'model_args.embed_dim %r' %
                            (embed, backend['embed_dim']))
    proj.seen.add('num_class')
    proj.done(fixed=PROJECTION_FIXED, closed=PROJECTION_CLOSED)
    init = {'checkpoint': top.get('init_checkpoint'),
            'head_check': top.get('head_spk_check', 'required')}
    return {'frontend': frontend, 'backend': backend, 'head': head,
            'init': init}


def _loss(top: _Node) -> Dict[str, Any]:
    pa = top.child('pair_auc')
    center = pa.child('center')
    out = {'weight': pa.get('weight'), 'scale': pa.get('scale', 16.0),
           'center': {'arm': center.get('arm'),
                      'momentum': center.get('momentum', 0.99),
                      'warmup': center.get('warmup', 20)}}
    center.done()
    pa.done(closed=PAIR_AUC_CLOSED)
    return {'consistency_weight': top.get('consistency_weight'),
            'pair_auc': out}


def _optim(top: _Node) -> Dict[str, Any]:
    oargs = top.child('optimizer_args')
    sargs = top.child('scheduler_args')
    out = {
        'optimizer': top.get('optimizer'),
        'weight_decay': oargs.get('weight_decay'),
        'fused': oargs.get('fused', False),
        'scheduler': top.get('scheduler'),
        'initial_lr': sargs.get('initial_lr'),
        'final_lr': sargs.get('final_lr'),
        'warm_up_epoch': sargs.get('warm_up_epoch'),
        'warm_from_zero': sargs.get('warm_from_zero', False),
        'frontend_lr_ratio': sargs.get('frontend_lr_ratio'),
        'frontend_modules': sargs.get('frontend_modules'),
        'grad_clip': top.get('grad_clip'),
    }
    # the trainer copies initial_lr / num_epochs into the resolved config
    for node, key, want in ((oargs, 'lr', out['initial_lr']),
                            (sargs, 'num_epochs', top.tree.get('num_epochs'))):
        value = node.get(key, None)
        if value is not None and value != want:
            top.problems.append('%s: %r differs from %r' %
                                (node.where(key), value, want))
    oargs.done()
    for key in ('epoch_iter', 'scale_ratio'):
        sargs.seen.add(key)
    sargs.done(closed=SCHEDULER_CLOSED)
    return out


def _derived(tree: Mapping[str, Any], cfg: Config,
             problems: List[str]) -> Optional[Dict[str, Any]]:
    """Derived values stored by the trainer; checked against the config."""
    sargs = tree.get('scheduler_args') or {}
    margin = tree.get('margin_update') or {}
    stored = (('scheduler_args.epoch_iter', sargs.get('epoch_iter'),
               schema.epoch_iter(cfg)),
              ('margin_update.epoch_iter', margin.get('epoch_iter'),
               schema.epoch_iter(cfg)),
              ('scheduler_args.scale_ratio', sargs.get('scale_ratio'),
               schema.scale_ratio(cfg)))
    for path, value, want in stored:
        if value is not None and value != want:
            problems.append('%s: %r does not match the value %r computed '
                            'from the config' % (path, value, want))
    num_class = (tree.get('projection_args') or {}).get('num_class')
    if num_class is None:
        return None
    return {'epoch_iter': schema.epoch_iter(cfg),
            'scale_ratio': schema.scale_ratio(cfg), 'num_class': num_class}


def _raise(problems: List[str]) -> None:
    if problems:
        raise ConfigError('legacy config: %d problem%s\n  %s' % (
            len(problems), '' if len(problems) == 1 else 's',
            '\n  '.join(problems)))


def from_legacy(tree: Mapping[str, Any]) -> Config:
    """Resolved wespeaker-schema training config -> :class:`Config`.

    ``derived`` is filled when the config carries the class count
    (``projection_args.num_class``), as a config saved by the trainer does.
    """
    if not isinstance(tree, Mapping):
        raise ConfigError('legacy config: must be a mapping, got %r' %
                          type(tree).__name__)
    problems: List[str] = []
    top = _Node(tree, '', problems)
    ddp = top.child('ddp_args')
    ds = top.child('dataset_args')
    new = {
        'run': {
            'exp_dir': top.get('exp_dir'), 'seed': top.get('seed'),
            'gpus': _gpus(top.get('gpus')),
            'num_epochs': top.get('num_epochs'),
            'save_epoch_interval': top.get('save_epoch_interval'),
            'log_batch_interval': top.get('log_batch_interval'),
            'num_avg': top.get('num_avg'),
        },
        'data': _data(top, ds),
        'model': _model(top, ds),
        'loss': _loss(top),
        'optim': _optim(top),
        'engine': {
            'amp_dtype': top.get('amp_dtype'), 'tf32': top.get('tf32'),
            'cudnn_benchmark': top.get('cudnn_benchmark'),
            'ddp': {k: ddp.get(k) for k in ('gradient_as_bucket_view',
                                            'static_graph')},
            'save_optimizer_state': top.get('save_optimizer_state'),
        },
    }
    ddp.done()
    ds.done(fixed=DATASET_FIXED, ignored=DATASET_IGNORED,
            closed=DATASET_CLOSED)
    top.done(fixed=TOP_FIXED, ignored=TOP_IGNORED, closed=TOP_CLOSED)
    _raise(problems)
    try:
        cfg = schema.build_config(new)
    except ConfigError as exc:
        raise ConfigError('legacy config: %s' % exc) from exc
    derived = _derived(tree, cfg, problems)
    _raise(problems)
    if derived is not None:
        new['derived'] = derived
        cfg = schema.build_config(new)
    return cfg
