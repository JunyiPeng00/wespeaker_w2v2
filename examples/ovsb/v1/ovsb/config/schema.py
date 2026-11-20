"""Typed schema of the recipe config.

One dataclass per config section. :func:`build_config` turns a plain mapping
(parsed YAML) into a :class:`Config` and raises :class:`ConfigError` naming
the dotted path of the first unknown key, missing key, wrong type or
out-of-range value. :func:`validate` adds the checks that span several
fields (arm names, batch layout, layer counts).

Fields that take a single value today (``stratify``, ``amp_dtype``, ...) are
enumerations with one member: the recipe implements that one behaviour.
"""

import dataclasses
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

AUG_TYPES = ('clean', 'noise', 'reverb', 'reverb_noise')
CODECS = ('mp3', 'opus', 'vorbis', 'mulaw8k', 'g722')
PAIR_POLICIES = ('xsession', 'contrast', 'ks_contrast')
SIMULATED_REVERB = 'simulated'
# batch size the base learning rate refers to (wespeaker convention)
LR_BASE_BATCH = 64


class ConfigError(ValueError):
    """The config has an unknown key, a wrong type or an invalid value."""


def _f(default: Any = dataclasses.MISSING, **checks: Any) -> Any:
    """A dataclass field with optional checks.

    checks: ge / gt / le / lt (numbers; applied to every element of a list),
    choices (allowed values), min_len / length (lists and mappings).
    """
    if default is dataclasses.MISSING:
        return field(metadata=checks)
    if isinstance(default, (list, dict)):
        return field(default_factory=lambda: type(default)(default),
                     metadata=checks)
    return field(default=default, metadata=checks)


# --------------------------------------------------------------------- run
@dataclass
class RunCfg:
    """Run identity and epoch bookkeeping."""
    exp_dir: str = _f()
    seed: int = _f(ge=0)
    gpus: List[int] = _f(min_len=1, ge=0)
    num_epochs: int = _f(ge=1)
    save_epoch_interval: int = _f(ge=1)
    log_batch_interval: int = _f(ge=1)
    num_avg: int = _f(ge=1)


# -------------------------------------------------------------------- data
@dataclass
class ListsCfg:
    """Training list and its side tables."""
    train_label: str = _f()  # utt2spk; its speakers define the classes
    index: str = _f()  # tar index directory
    spk2corpus: str = _f()
    spk2lang: str = _f()
    index_root: Optional[str] = _f(None)  # overrides the index's shard root


@dataclass
class ArmCfg:
    """A sampling arm: the speakers of the listed corpora."""
    weight: float = _f(gt=0.0)
    corpus: List[str] = _f(min_len=1)


@dataclass
class CellSelectCfg:
    """Speaker selection of a cell."""
    spk2lang: str = _f()


@dataclass
class CellCfg:
    """A fixed share of all speaker draws reserved inside an arm."""
    arm: str = _f()
    share: float = _f(gt=0.0, lt=1.0)
    select: CellSelectCfg = _f()


@dataclass
class DialectMassCfg:
    """Equal draw mass per dialect, capped at a multiple of the median."""
    boost: float = _f(gt=0.0)
    cap_x_median: float = _f(gt=0.0)


@dataclass
class PolicyCfg:
    """How the K utterances of a speaker are chosen in one arm."""
    pair_policy: str = _f(choices=PAIR_POLICIES)
    p_dialect: float = _f(0.5, ge=0.0, le=1.0)
    segment_first: bool = _f(True)
    dialect_mass: Optional[DialectMassCfg] = _f(None)
    w_bridge: float = _f(1.0, gt=0.0)  # weight of multi-genre speakers
    subdialect_file: Optional[str] = _f(None)  # ks_contrast
    spk_weight_file: Optional[str] = _f(None)  # '<spk> <multiplier>' lines
    spk_weight_share: Optional[float] = _f(None, gt=0.0, lt=1.0)


@dataclass
class SamplerCfg:
    """Episodic sampler: P speakers x K utterances per episode."""
    P: int = _f(ge=1)
    K: int = _f(ge=2)
    sample_num_per_epoch: int = _f(ge=1)
    vox2_lang: str = _f()
    stratify: str = _f(choices=('systematic',))
    log_every: int = _f(ge=1)
    arms: Dict[str, ArmCfg] = _f(min_len=1)
    forbid_same_segment: bool = _f()
    cells: Dict[str, CellCfg] = _f({})
    policies: Dict[str, PolicyCfg] = _f({})


@dataclass
class FilterCfg:
    """Utterance length filter after the sampler."""
    min_sec: float = _f(gt=0.0)  # shorter utterances are dropped
    max_sec: float = _f(gt=0.0)  # longer ones are cut to a random chunk


@dataclass
class EnergyGateCfg:
    """Redraw short crops that are mostly silence."""
    gate_below_sec: float = _f(gt=0.0)
    min_active_frac: float = _f(ge=0.0, le=1.0)
    max_redraws: int = _f(ge=0)


@dataclass
class CropCfg:
    """Variable-length crops; one length cap per batch.

    ``max_secs`` and ``arm_caps`` hold ``[seconds, weight]`` pairs.
    """
    max_secs: List[List[float]] = _f(min_len=1)
    min_sec: float = _f(gt=0.0)
    full_prob: float = _f(ge=0.0, le=1.0)
    onset_prob: float = _f(ge=0.0, le=1.0)
    onset_max_sec: float = _f(gt=0.0)
    pair_prob: float = _f(ge=0.0, le=1.0)
    pair_anchor_sec: float = _f(gt=0.0)
    pair_view_secs: List[float] = _f(length=2, gt=0.0)
    pair_view_onset_prob: Optional[float] = _f(None, ge=0.0, le=1.0)
    arm_caps: Dict[str, List[List[float]]] = _f({})
    energy_gate: Optional[EnergyGateCfg] = _f(None)


@dataclass
class ZeroRunCfg:
    """Dither over runs of exact zeros before augmentation."""
    min_run_ms: float = _f(gt=0.0)
    lsb_level: float = _f(gt=0.0)
    dither_lsb: float = _f(gt=0.0)
    arms: Optional[List[str]] = _f(None)  # None = every arm


@dataclass
class AugmentCfg:
    """Additive noise and reverberation; one type is drawn per utterance.

    ``probs`` maps a type of ``AUG_TYPES`` to its probability (the rest is
    clean); ``arm_probs`` replaces ``probs`` for the listed arms.
    """
    probs: Dict[str, float] = _f(ge=0.0, le=1.0)
    reverberant_noise: bool = _f()
    reverb_sources: List[str] = _f(min_len=1)  # 'simulated' or an LMDB dir
    reverb_weights: List[float] = _f(min_len=1, ge=0.0)
    noise_lmdb: str = _f()
    reverb_lmdb: str = _f()  # the 'simulated' source
    arm_probs: Dict[str, Dict[str, float]] = _f({})
    noise_exclude_list: Optional[str] = _f(None)


@dataclass
class ChannelCfg:
    """Codec / device chain on some rows of single-session arms."""
    arms: List[str] = _f(min_len=1)
    prob: float = _f(ge=0.0, le=1.0)
    kpos: List[int] = _f(min_len=1, ge=0)
    distinct: bool = _f()
    codecs: List[str] = _f(min_len=1, choices=CODECS)
    eq_prob: float = _f(ge=0.0, le=1.0)
    eq_gain_db: float = _f(ge=0.0)
    bandlimit_prob: float = _f(ge=0.0, le=1.0)
    clip_prob: float = _f(ge=0.0, le=1.0)
    clip_level: List[float] = _f(length=2, gt=0.0, le=1.0)
    loss_prob: float = _f(ge=0.0, le=1.0)
    loss_rate: List[float] = _f(length=2, ge=0.0, le=1.0)
    isolate_ffmpeg: bool = _f()


@dataclass
class LoaderCfg:
    """torch DataLoader arguments."""
    batch_size: int = _f(ge=1)
    num_workers: int = _f(ge=0)
    prefetch_factor: int = _f(ge=1)
    pin_memory: bool = _f()
    drop_last: bool = _f()


@dataclass
class DataCfg:
    """Everything between the files on disk and a batch."""
    lists: ListsCfg = _f()
    sampler: SamplerCfg = _f()
    filter: FilterCfg = _f()
    crop: CropCfg = _f()
    zero_run_dither: ZeroRunCfg = _f()
    augment: AugmentCfg = _f()
    channel: ChannelCfg = _f()
    loader: LoaderCfg = _f()
    resample_rate: int = _f(ge=1)
    # frames per crop of fixed-length batched extraction (not training)
    num_frms: int = _f(ge=1)


# ------------------------------------------------------------------- model
@dataclass
class TimeMaskCfg:
    """Zeroed spans of projected input frames in training."""
    prob: float = _f(ge=0.0, le=0.5)
    span: int = _f(ge=1)


@dataclass
class FrontendCfg:
    """W2V-BERT 2.0 encoder."""
    assets: str = _f()  # directory with config.json and the weights
    num_layers: int = _f(ge=1)  # encoder layers kept
    gradient_checkpointing: bool = _f()
    time_mask: TimeMaskCfg = _f()


@dataclass
class BackendCfg:
    """Pooling backend."""
    name: str = _f(choices=('SSL_BACKEND_MHFA_LayerNorm',))
    head_nb: int = _f(ge=1)
    embed_dim: int = _f(ge=1)
    compression_dim: int = _f(ge=1)
    feature_grad_mult: float = _f(ge=0.0)
    nb_layer: int = _f(ge=2)  # hidden states pooled: num_layers + 1
    feat_dim: int = _f(ge=1)  # encoder width


@dataclass
class HeadCfg:
    """Sub-centre additive-margin classifier with inter-top-k penalty."""
    scale: float = _f(gt=0.0)
    K: int = _f(ge=1)
    k_top: int = _f(ge=0)
    mp: float = _f(ge=0.0)
    margin: float = _f(ge=0.0)
    # crops shorter than this get a proportionally smaller margin
    duration_margin_ref_sec: float = _f(gt=0.0)


@dataclass
class InitCfg:
    """Initial checkpoint (encoder, backend and classifier head)."""
    checkpoint: str = _f()
    head_check: str = _f(choices=('required',))


@dataclass
class ModelCfg:
    """Speaker model."""
    frontend: FrontendCfg = _f()
    backend: BackendCfg = _f()
    head: HeadCfg = _f()
    init: InitCfg = _f()


# -------------------------------------------------------------------- loss
@dataclass
class CenterCfg:
    """Running mean embedding of one arm, subtracted before scoring."""
    arm: str = _f()
    momentum: float = _f(ge=0.0, lt=1.0)
    warmup: int = _f(ge=1)


@dataclass
class PairAucCfg:
    """Pairwise verification loss over the batch."""
    weight: float = _f(ge=0.0)
    scale: float = _f(gt=0.0)
    center: CenterCfg = _f()


@dataclass
class LossCfg:
    """Weights of the losses added to the classification loss."""
    consistency_weight: float = _f(ge=0.0)
    pair_auc: PairAucCfg = _f()


# ------------------------------------------------------------------- optim
@dataclass
class OptimCfg:
    """Optimiser and learning-rate schedule.

    Learning rates refer to a global batch of 64 and are multiplied by
    ``derived.scale_ratio``.
    """
    optimizer: str = _f(choices=('AdamW',))
    weight_decay: float = _f(ge=0.0)
    fused: bool = _f()
    scheduler: str = _f(choices=('ExponentialDecrease',))
    initial_lr: float = _f(gt=0.0)
    final_lr: float = _f(gt=0.0)
    warm_up_epoch: float = _f(ge=0.0)
    warm_from_zero: bool = _f()
    frontend_lr_ratio: float = _f(gt=0.0)
    frontend_modules: List[str] = _f(min_len=1)
    grad_clip: float = _f(gt=0.0)


# ------------------------------------------------------------------ engine
@dataclass
class DdpCfg:
    """DistributedDataParallel arguments."""
    gradient_as_bucket_view: bool = _f()
    static_graph: bool = _f()


@dataclass
class EngineCfg:
    """Numerics and bookkeeping of the training loop."""
    amp_dtype: str = _f(choices=('bfloat16',))
    tf32: bool = _f()
    cudnn_benchmark: bool = _f()
    ddp: DdpCfg = _f()
    save_optimizer_state: bool = _f()


# ----------------------------------------------------------------- derived
@dataclass
class DerivedCfg:
    """Values computed from the rest of the config (see ``load.resolve``)."""
    epoch_iter: int = _f(ge=1)
    scale_ratio: float = _f(gt=0.0)
    num_class: int = _f(ge=1)


@dataclass
class Config:
    """The recipe config."""
    run: RunCfg = _f()
    data: DataCfg = _f()
    model: ModelCfg = _f()
    loss: LossCfg = _f()
    optim: OptimCfg = _f()
    engine: EngineCfg = _f()
    derived: Optional[DerivedCfg] = _f(None)


# ------------------------------------------------------------ construction
def _type_name(hint: Any) -> str:
    if hint is float:
        return 'a number'
    if hint is int:
        return 'an integer'
    if hint is bool:
        return 'a boolean'
    if hint is str:
        return 'a string'
    origin = typing.get_origin(hint)
    if origin is list:
        return 'a list'
    return 'a mapping'


def _check_number(value: Any, checks: Mapping[str, Any], path: str) -> None:
    bounds = (('ge', '>=', lambda v, b: v >= b), ('gt', '>', lambda v, b: v > b),
              ('le', '<=', lambda v, b: v <= b), ('lt', '<', lambda v, b: v < b))
    for key, symbol, holds in bounds:
        if key in checks and not holds(value, checks[key]):
            raise ConfigError('%s: must be %s %r, got %r' %
                              (path, symbol, checks[key], value))


def _check_scalar(value: Any, checks: Mapping[str, Any], path: str) -> None:
    if 'choices' in checks and value not in checks['choices']:
        raise ConfigError('%s: must be one of %s, got %r' %
                          (path, list(checks['choices']), value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _check_number(value, checks, path)


def _check_size(value: Any, checks: Mapping[str, Any], path: str) -> None:
    if 'min_len' in checks and len(value) < checks['min_len']:
        raise ConfigError('%s: needs at least %d entr%s, got %d' % (
            path, checks['min_len'],
            'y' if checks['min_len'] == 1 else 'ies', len(value)))
    if 'length' in checks and len(value) != checks['length']:
        raise ConfigError('%s: needs exactly %d entries, got %d' %
                          (path, checks['length'], len(value)))


def _wrong_type(hint: Any, value: Any, path: str) -> ConfigError:
    return ConfigError('%s: must be %s, got %r (%s)' %
                       (path, _type_name(hint), value, type(value).__name__))


def _coerce(hint: Any, value: Any, path: str,
            checks: Mapping[str, Any]) -> Any:
    """Check ``value`` against the type hint; return the typed value."""
    origin = typing.get_origin(hint)
    if origin is typing.Union:  # Optional[X]
        inner = [a for a in typing.get_args(hint) if a is not type(None)][0]
        return None if value is None else _coerce(inner, value, path, checks)
    if value is None:
        raise ConfigError('%s: must be %s, got null' %
                          (path, _type_name(hint)))
    if dataclasses.is_dataclass(hint):
        return build_section(hint, value, path)
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise _wrong_type(hint, value, path)
        _check_size(value, checks, path)
        inner = typing.get_args(hint)[0]
        inner_checks = {k: v for k, v in checks.items()
                        if k not in ('min_len', 'length')}
        return [_coerce(inner, v, '%s[%d]' % (path, i), inner_checks)
                for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, Mapping):
            raise _wrong_type(hint, value, path)
        _check_size(value, checks, path)
        inner = typing.get_args(hint)[1]
        inner_checks = {k: v for k, v in checks.items()
                        if k not in ('min_len', 'length')}
        out = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ConfigError('%s: key %r must be a non-empty string' %
                                  (path, key))
            out[key] = _coerce(inner, item, '%s.%s' % (path, key),
                               inner_checks)
        return out
    if hint is bool:
        if not isinstance(value, bool):
            raise _wrong_type(hint, value, path)
    elif hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _wrong_type(hint, value, path)
    elif hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _wrong_type(hint, value, path)
        value = float(value)
        if value != value or value in (float('inf'), float('-inf')):
            raise ConfigError('%s: must be finite, got %r' % (path, value))
    elif hint is str:
        if not isinstance(value, str):
            raise _wrong_type(hint, value, path)
        if not value:
            raise ConfigError('%s: must not be empty' % path)
    else:  # a schema bug, not a config error
        raise TypeError('unsupported schema type %r at %s' % (hint, path))
    _check_scalar(value, checks, path)
    return value


def build_section(cls: type, value: Any, path: str) -> Any:
    """Build dataclass ``cls`` from a mapping; ``path`` is its dotted path."""
    if isinstance(value, cls):
        value = dataclasses.asdict(value)
    if not isinstance(value, Mapping):
        raise ConfigError('%s: must be a mapping with keys %s, got %r' % (
            path or '<config>', [f.name for f in dataclasses.fields(cls)],
            value))
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    prefix = path + '.' if path else ''
    for key in value:
        if key not in fields:
            raise ConfigError('%s%s: unknown key (known keys of %s: %s)' % (
                prefix, key, path or 'the config', sorted(fields)))
    kwargs = {}
    for name, fld in fields.items():
        if name not in value:
            if fld.default is dataclasses.MISSING and \
                    fld.default_factory is dataclasses.MISSING:
                raise ConfigError('%s%s: required key is missing' %
                                  (prefix, name))
            continue
        kwargs[name] = _coerce(hints[name], value[name], prefix + name,
                               fld.metadata)
    return cls(**kwargs)


def _need_arms(names: Any, arms: Mapping[str, Any], path: str) -> None:
    for name in names:
        if name not in arms:
            raise ConfigError('%s: %r is not an arm of data.sampler.arms %s' %
                              (path, name, list(arms)))


def _check_caps(caps: List[List[float]], path: str) -> None:
    for i, cap in enumerate(caps):
        if len(cap) != 2 or cap[0] <= 0 or cap[1] <= 0:
            raise ConfigError('%s[%d]: must be [seconds > 0, weight > 0], '
                              'got %r' % (path, i, cap))


def _check_probs(probs: Mapping[str, float], path: str) -> None:
    for name, prob in probs.items():
        if name not in AUG_TYPES:
            raise ConfigError('%s.%s: unknown augmentation type (known: %s)'
                              % (path, name, list(AUG_TYPES)))
        _check_number(prob, {'ge': 0.0, 'le': 1.0}, '%s.%s' % (path, name))
    if sum(probs.values()) > 1.0 + 1e-6:
        raise ConfigError('%s: probabilities sum to %g > 1' %
                          (path, sum(probs.values())))


def world_size(cfg: Config) -> int:
    """Number of training processes (one per GPU)."""
    return len(cfg.run.gpus)


def epoch_iter(cfg: Config) -> int:
    """Optimiser steps per epoch."""
    return (cfg.data.sampler.sample_num_per_epoch // world_size(cfg)
            // cfg.data.loader.batch_size)


def scale_ratio(cfg: Config) -> float:
    """Learning-rate multiplier of the global batch relative to 64."""
    return 1.0 * world_size(cfg) * cfg.data.loader.batch_size / LR_BASE_BATCH


def validate(cfg: Config) -> Config:
    """Checks that span several fields. Returns ``cfg``."""
    run, data, model = cfg.run, cfg.data, cfg.model
    sampler, arms = data.sampler, data.sampler.arms
    if len(set(run.gpus)) != len(run.gpus):
        raise ConfigError('run.gpus: repeated entries in %r' % (run.gpus,))
    if run.num_avg > run.num_epochs:
        raise ConfigError('run.num_avg: %d exceeds run.num_epochs %d' %
                          (run.num_avg, run.num_epochs))
    if sampler.P * sampler.K != data.loader.batch_size:
        raise ConfigError(
            'data.sampler.P: P * K (%d) must equal data.loader.batch_size '
            '(%d)' % (sampler.P * sampler.K, data.loader.batch_size))
    if epoch_iter(cfg) < 1:
        raise ConfigError('data.sampler.sample_num_per_epoch: %d gives no '
                          'step per epoch for %d processes x batch %d' % (
                              sampler.sample_num_per_epoch, world_size(cfg),
                              data.loader.batch_size))
    total = sum(a.weight for a in arms.values())
    for name, cell in sampler.cells.items():
        path = 'data.sampler.cells.%s' % name
        _need_arms([cell.arm], arms, path + '.arm')
        if not cell.share < arms[cell.arm].weight / total:
            raise ConfigError(
                '%s.share: %g must be below the normalised weight %g of arm '
                '%s' % (path, cell.share, arms[cell.arm].weight / total,
                        cell.arm))
    _need_arms(sampler.policies, arms, 'data.sampler.policies')
    for name, pol in sampler.policies.items():
        path = 'data.sampler.policies.%s' % name
        if (pol.pair_policy == 'ks_contrast') != \
                (pol.subdialect_file is not None):
            raise ConfigError('%s.subdialect_file: set exactly when '
                              'pair_policy is ks_contrast' % path)
        if pol.spk_weight_share is not None and pol.spk_weight_file is None:
            raise ConfigError('%s.spk_weight_share: needs spk_weight_file'
                              % path)
    if data.filter.min_sec >= data.filter.max_sec:
        raise ConfigError('data.filter.min_sec: %g must be below max_sec %g'
                          % (data.filter.min_sec, data.filter.max_sec))
    crop = data.crop
    _check_caps(crop.max_secs, 'data.crop.max_secs')
    _need_arms(crop.arm_caps, arms, 'data.crop.arm_caps')
    for name, caps in crop.arm_caps.items():
        if not caps:
            raise ConfigError('data.crop.arm_caps.%s: needs at least 1 entry'
                              % name)
        _check_caps(caps, 'data.crop.arm_caps.%s' % name)
    if crop.pair_view_secs[0] > crop.pair_view_secs[1]:
        raise ConfigError('data.crop.pair_view_secs: must be [low, high], '
                          'got %r' % (crop.pair_view_secs,))
    _need_arms(data.zero_run_dither.arms or (), arms,
               'data.zero_run_dither.arms')
    aug = data.augment
    _check_probs(aug.probs, 'data.augment.probs')
    _need_arms(aug.arm_probs, arms, 'data.augment.arm_probs')
    for name, probs in aug.arm_probs.items():
        _check_probs(probs, 'data.augment.arm_probs.%s' % name)
    if len(aug.reverb_weights) != len(aug.reverb_sources):
        raise ConfigError('data.augment.reverb_weights: needs one weight per '
                          'entry of reverb_sources (%d), got %d' % (
                              len(aug.reverb_sources),
                              len(aug.reverb_weights)))
    if sum(aug.reverb_weights) <= 0:
        raise ConfigError('data.augment.reverb_weights: all weights are 0')
    channel = data.channel
    _need_arms(channel.arms, arms, 'data.channel.arms')
    for name in ('clip_level', 'loss_rate'):
        low, high = getattr(channel, name)
        if low > high:
            raise ConfigError('data.channel.%s: must be [low, high], got %r'
                              % (name, [low, high]))
    if any(k >= sampler.K for k in channel.kpos):
        raise ConfigError('data.channel.kpos: positions must be below '
                          'data.sampler.K (%d), got %r' %
                          (sampler.K, channel.kpos))
    if model.backend.nb_layer != model.frontend.num_layers + 1:
        raise ConfigError(
            'model.backend.nb_layer: %d must be model.frontend.num_layers + '
            '1 (%d)' % (model.backend.nb_layer,
                        model.frontend.num_layers + 1))
    _need_arms([cfg.loss.pair_auc.center.arm], arms,
               'loss.pair_auc.center.arm')
    if cfg.optim.final_lr > cfg.optim.initial_lr:
        raise ConfigError('optim.final_lr: %g exceeds optim.initial_lr %g' %
                          (cfg.optim.final_lr, cfg.optim.initial_lr))
    if cfg.derived is not None:
        for name, want in (('epoch_iter', epoch_iter(cfg)),
                           ('scale_ratio', scale_ratio(cfg))):
            if getattr(cfg.derived, name) != want:
                raise ConfigError(
                    'derived.%s: %r does not match the value %r computed '
                    'from the config' % (name, getattr(cfg.derived, name),
                                         want))
    return cfg


def build_config(tree: Mapping[str, Any]) -> Config:
    """Mapping (parsed YAML) -> validated :class:`Config`."""
    return validate(build_section(Config, tree, ''))
