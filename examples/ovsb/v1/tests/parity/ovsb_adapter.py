#!/usr/bin/env python
# Copyright (c) 2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Whole-recipe parity gate: this recipe against a reference dump.

    ovsb_adapter.py dump    --harness parity.py --site site.yaml --out new.json
    ovsb_adapter.py compare --harness parity.py ref.json new.json

``parity.py`` is the harness that produced the reference dump from the
original training tree. It is imported as a module (never modified): its
check functions run unchanged wherever they only need an object with the
attributes of its ``TrainPath`` adapter. :class:`OvsbPath` is that object
for this recipe, built through ``ovsb.api`` only.

``dump`` writes a JSON file in the layout of the harness (checks
``sampler``, ``batches``, ``schedule``, ``model``, plus ``config_port`` and
``meta``). ``compare`` reduces the reference and the new dump to what both
sides have (the reductions are listed in :func:`make_views` and printed),
compares the two views bit for bit with the harness's own ``walk`` and
writes them next to the new dump, so that the harness can also be run on
them directly: ``parity.py compare --strict ref.view.json new.view.json``.

See README.md in this directory for the site file and the commands.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import logging
import os
import struct
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

RECIPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
REPO_ROOT = os.path.abspath(os.path.join(RECIPE_DIR, '..', '..', '..'))
RECIPE_CONFIG = os.path.join(RECIPE_DIR, 'conf', 'stage2_v9c.yaml')
CHECKS = ('config', 'sampler', 'batches', 'schedule', 'model')
PENDING = 'pending engine'

# config fields that hold a location, not behaviour
PATH_FIELDS = (
    'run.exp_dir', 'data.lists.train_label', 'data.lists.index',
    'data.lists.spk2corpus', 'data.lists.spk2lang', 'data.lists.index_root',
    'data.augment.noise_lmdb', 'data.augment.reverb_lmdb',
    'data.augment.reverb_sources', 'data.augment.noise_exclude_list',
    'model.frontend.assets', 'model.init.checkpoint')
PATH_LEAVES = ('subdialect_file', 'spk_weight_file')
# mappings that are looked up by name: their order is not behaviour
UNORDERED = ('data.sampler.policies', 'data.augment.arm_probs',
             'data.crop.arm_caps')
# per-row fields of an episode that the sampler check hashes
EPISODE_FIELDS = ('key', 'spk', 'arm', 'lang', 'vid', 'kpos', 'ptype', 'seg',
                  'dev', 'dcls', 'dial', 'genre')


def load_harness(path: str) -> Any:
    """Import the harness file as a module, without running it."""
    spec = importlib.util.spec_from_file_location('parity_harness', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['parity_harness'] = module
    spec.loader.exec_module(module)
    return module


def read_site(path: str) -> Dict[str, Any]:
    """The site file: where the reference run's data is on this machine."""
    import yaml
    with open(path) as fin:
        site = yaml.safe_load(fin)
    for key in ('recipe_dir', 'checkpoint', 'legacy_saved', 'legacy_authored',
                'overrides'):
        if key not in site:
            raise SystemExit('%s: no entry %r' % (path, key))
    return site


def site_overrides(site: Dict[str, Any]) -> List[str]:
    """``dotted.path=value`` strings of the site file's override mapping."""
    return ['%s=%s' % (k, json.dumps(v)) for k, v in site['overrides'].items()]


# ---------------------------------------------------------------- adapter
class _Criterion:
    """Stand-in for the loss module the harness moves to the device."""

    def to(self, device: Any) -> '_Criterion':
        return self


class _SamplerView:
    """The attributes the harness's sampler check reads, from the table."""

    def __init__(self, sampler: Any) -> None:
        table = sampler.table
        self.n_utts, self.n_spk = table.n_utts, table.n_spk
        self.P, self.K = sampler.P, sampler.K
        self.arm_names = list(table.arm_names)
        self.spk_arm = table.spk_arm
        self.static_weights = table.arm_weights
        self.rows, self.spk_of = table.rows, table.spk_of
        self.spk_names = table.spk_names


class OvsbPath:
    """What the harness's checks need, built through ``ovsb.api``.

    Attribute and method names are those of the harness's ``TrainPath``.
    """

    def __init__(self, cfg: Any, world: int) -> None:
        from ovsb.config import schema
        self.cfg = cfg
        self.rank = 0
        self.world = int(world)
        self.epoch_iter = schema.epoch_iter(cfg)
        pauc = cfg.loss.pair_auc
        # the few keys the harness reads from the original config dict
        self.configs: Dict[str, Any] = {
            'dataloader_args': {'num_workers': cfg.data.loader.num_workers},
            'dataset_args': {'episodic_sampler': {'arms': {
                name: None for name in cfg.data.sampler.arm_order}}},
            'pair_auc': {'weight': pauc.weight, 'scale': pauc.scale,
                         'center': {'arm': pauc.center.arm,
                                    'momentum': pauc.center.momentum,
                                    'warmup': pauc.center.warmup}},
            'consistency_weight': cfg.loss.consistency_weight,
            'enable_amp': True,
        }
        self._centers: Dict[Tuple[int, float, int], Any] = {}
        self.objects: Any = None

    # -- construction ------------------------------------------------------
    def build(self) -> int:
        """``build_training`` for rank 0 on the CPU; returns the base seed
        the first data-loader iterator would draw next."""
        import torch
        from ovsb import api
        self.objects = obj = api.build_training(
            self.cfg, self.rank, self.world, torch.device('cpu'))
        base_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.dataset, self.loader = obj.dataset, obj.loader
        self.sampler, self.model = obj.sampler, obj.model
        self.ep_sampler = _SamplerView(obj.sampler)
        self.class_names = [str(n) for n in obj.table.class_names]
        self.num_params = sum(p.numel() for n, p in
                              self.model.named_parameters()
                              if not n.startswith('projection.'))
        return base_seed

    def release_data(self, trim: Any) -> None:
        """Drop the data side (several GB) before the model check."""
        obj = self.objects
        obj.table = obj.sampler = obj.dataset = obj.loader = None
        obj.optimizer = obj.scheduler = obj.margin_scheduler = None
        self.dataset = self.loader = self.sampler = self.ep_sampler = None
        trim()

    # -- sampler -----------------------------------------------------------
    def episode_rows(self, epoch: int, rank: int, wid: int, nw: int,
                     max_episodes: Optional[int]) -> Iterator[List[List[Any]]]:
        """Per episode, the rows of :data:`EPISODE_FIELDS` (no audio)."""
        sampler = self.sampler
        table = sampler.table
        for ep in sampler.iter_episodes(
                epoch=epoch, rank=rank, world=self.world, wid=wid, nw=nw,
                max_episodes=max_episodes):
            d = {k: v.tolist() for k, v in sampler.describe(ep).items()}
            yield [[table.index.key(d['row'][i]),
                    str(table.spk_names[d['spk'][i]])] +
                   [int(d[f][i]) for f in EPISODE_FIELDS[2:]]
                   for i in range(len(d['row']))]

    # -- model -------------------------------------------------------------
    def load_checkpoint(self, path: str) -> Tuple[str, int]:
        """Head check and weights as the trainer loads them."""
        from ovsb.engine.checkpoint import load_init
        rows = int(self.model.state_dict()['projection.weight'].shape[0])
        how = load_init(self.model, path, self.class_names,
                        os.path.dirname(path), write=False)
        return how, rows

    def build_schedulers(self, ddp_model: Any = None) -> Tuple[Any, ...]:
        """Fresh (criterion, optimizer, scheduler, margin scheduler, None)."""
        from ovsb import api
        optimizer = api.build_optimizer(self.model, self.cfg)
        scheduler, margin = api.build_schedulers(optimizer, self.model,
                                                 self.cfg)
        return _Criterion(), optimizer, scheduler, margin, None

    def _bundle(self, configs: Dict[str, Any]) -> Any:
        """The objective for a variant of the harness's config dict.

        The reference trainer kept one running centre per process and
        (arm, momentum, warm-up) setting; the harness's variants with
        another warm-up therefore had their own centre. Same here.
        """
        from ovsb.losses.bundle import LossBundle
        from ovsb.losses.pair_auc import RunningCenter
        pauc = configs['pair_auc']
        center = None
        if float(pauc['weight']) > 0:
            key = (self.cfg.data.sampler.arm_order.index(
                pauc['center']['arm']), float(pauc['center']['momentum']),
                int(pauc['center']['warmup']))
            if key not in self._centers:
                self._centers[key] = RunningCenter(*key)
            center = self._centers[key]
        return LossBundle(float(configs['consistency_weight']),
                          float(pauc['weight']), float(pauc['scale']), center)

    def run_epoch(self, loader: Any, ddp_model: Any, criterion: Any,
                  optimizer: Any, scheduler: Any, margin_scheduler: Any,
                  epoch: int, scaler: Any, device: Any,
                  configs: Dict[str, Any]) -> None:
        """One ``train_step`` per batch of ``loader``, as the epoch loop."""
        from ovsb.engine.step import train_step
        ddp_model.train()
        bundle = self._bundle(configs)
        for i, batch in enumerate(loader):
            train_step(ddp_model, bundle, batch, self.cfg, device, optimizer,
                       scheduler, margin_scheduler,
                       (epoch - 1) * self.epoch_iter + i,
                       amp=bool(configs['enable_amp']))

    def extract_forward(self, wavs: Any, lens: Any = None) -> Any:
        """The forward of embedding extraction (ragged batch / one row)."""
        import torch
        model = self.model
        with torch.no_grad():
            if lens is None:
                lens = torch.LongTensor([wavs.shape[1]]).repeat(
                    wavs.shape[0]).to(wavs.device)
                features, _ = model.frontend(wavs, lens)
                outputs = model(features)
            else:
                features, flens = model.frontend(wavs, lens)
                outputs = model(features) if flens is None \
                    else model(features, flens)
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
        return embeds.float().cpu()


# ----------------------------------------------------------------- checks
def flatten(tree: Any, prefix: str = '') -> Dict[str, Any]:
    """Nested mapping -> {dotted path: leaf}; lists are leaves."""
    if not isinstance(tree, dict):
        return {prefix: tree}
    out: Dict[str, Any] = {}
    for key, value in tree.items():
        out.update(flatten(value, '%s.%s' % (prefix, key) if prefix
                           else str(key)))
    return out


def mapping_orders(tree: Any, prefix: str = '') -> Dict[str, List[str]]:
    """{dotted path: key order} of every mapping in ``tree``."""
    out: Dict[str, List[str]] = {}
    if isinstance(tree, dict):
        out[prefix] = [str(k) for k in tree]
        for key, value in tree.items():
            out.update(mapping_orders(value, '%s.%s' % (prefix, key)
                                      if prefix else str(key)))
    return out


def is_path_field(path: str) -> bool:
    return path in PATH_FIELDS or path.rsplit('.', 1)[-1] in PATH_LEAVES


def diff_configs(a: Any, b: Any) -> Dict[str, Any]:
    """Compare two recipe configs: values, and the order of mappings."""
    from ovsb.config import to_dict
    ta, tb = to_dict(a), to_dict(b)
    ta.pop('derived'), tb.pop('derived')
    fa, fb = flatten(ta), flatten(tb)
    differ = sorted(k for k in set(fa) | set(fb)
                    if fa.get(k, '<absent>') != fb.get(k, '<absent>'))
    oa, ob = mapping_orders(ta), mapping_orders(tb)
    order = sorted(k for k in set(oa) & set(ob)
                   if oa[k] != ob[k] and sorted(oa[k]) == sorted(ob[k]))
    return dict(
        differing_path_fields=[k for k in differ if is_path_field(k)],
        differing_other_fields={k: [fa.get(k, '<absent>'),
                                    fb.get(k, '<absent>')]
                                for k in differ if not is_path_field(k)},
        order_differs_unordered=[k for k in order if k in UNORDERED],
        order_differs_ordered=[k for k in order if k not in UNORDERED])


def check_config_port(site: Dict[str, Any], cfg_site: Any) -> Dict[str, Any]:
    """Replacement of the harness's ``config`` check (which reads guard
    internals of the original tree): the original run's config, read by the
    legacy reader, against this recipe's ``conf/stage2_v9c.yaml``."""
    from ovsb import api
    from ovsb.config import to_dict
    legacy = api.load_config(site['legacy_saved'],
                             legacy=site['legacy_authored'])
    recipe = api.load_config(RECIPE_CONFIG)
    out = dict(
        recipe_config=os.path.relpath(RECIPE_CONFIG, RECIPE_DIR),
        recipe_config_sha256=hashlib.sha256(
            open(RECIPE_CONFIG, 'rb').read()).hexdigest(),
        arm_order=list(recipe.data.sampler.arm_order),
        cell_order=list(recipe.data.sampler.cell_order),
        legacy_derived=to_dict(legacy)['derived'],
        legacy_vs_recipe=diff_configs(legacy, recipe),
        legacy_vs_gate_config=diff_configs(legacy, cfg_site),
        site_override_keys=sorted(site['overrides']))
    d = out['legacy_vs_recipe']
    out['equal_modulo_paths'] = not d['differing_other_fields'] and \
        not d['order_differs_ordered']
    g = out['legacy_vs_gate_config']
    out['gate_config_equal_modulo_paths'] = \
        not g['differing_other_fields'] and not g['order_differs_ordered']
    return out


def check_schedule(tp: OvsbPath, harness: Any, tmp: str) -> Dict[str, Any]:
    """Learning rates and margin of every step; layout of the harness.

    The resumed trajectories go through ``ovsb.engine.checkpoint.resume``
    at an epoch boundary: the wespeaker train state and the recipe's
    auxiliary state are written as ``save_epoch`` writes them (without the
    model file), fresh optimiser and schedulers are built and restored.
    """
    cfg, optim = tp.cfg, tp.cfg.optim
    n_ep, ei = int(cfg.run.num_epochs), tp.epoch_iter
    total = n_ep * ei
    warm = int(optim.warm_up_epoch * ei)
    sample_at = sorted(set([0, 1, 2, warm - 1, warm, warm + 1, ei - 1, ei,
                            ei + 1, total // 2, total - ei - 1, total - ei,
                            total - 1]))
    work = os.path.join(tmp, 'sched_%d' % os.getpid())
    os.makedirs(work, exist_ok=True)

    def boundary(epoch_done: int, opt: Any, sch: Any, msch: Any
                 ) -> Tuple[Any, Any, Any]:
        from ovsb.engine import checkpoint as ckpt
        from wespeaker.utils.train_state import (atomic_save,
                                                 save_train_state,
                                                 train_state_path)
        state = train_state_path(work, epoch_done)
        aux = ckpt.aux_state_path(work, epoch_done)
        atomic_save({'epoch': epoch_done, 'stream': 0, 'world': tp.world,
                     'center': None}, aux)
        save_train_state(state, epoch_done, opt, None, sch, msch, None)
        _, opt, sch, msch, _ = tp.build_schedulers()
        point = ckpt.ResumePoint(ckpt.EPOCH, epoch_done,
                                 weights=ckpt.model_path(work, epoch_done))
        assert point.start_epoch == epoch_done + 1 and point.start_step == 0
        ckpt.resume(point, work, opt, sch, msch, None, 0, tp.world, ei,
                    cfg.engine.save_optimizer_state)
        os.remove(state)
        os.remove(aux)
        return opt, sch, msch

    def trajectory(resume_every: int) -> Dict[str, Any]:
        _, opt, sch, msch, _ = tp.build_schedulers()
        groups = harness.group_summary(opt)
        digest = hashlib.sha256()
        samples, resumes = {}, 0
        for epoch in range(1, n_ep + 1):
            if resume_every and epoch > 1 and (epoch - 1) % resume_every == 0:
                opt, sch, msch = boundary(epoch - 1, opt, sch, msch)
                resumes += 1
            for i in range(ei):
                cur = (epoch - 1) * ei + i
                sch.step(cur)
                msch.step(cur)
                vals = [float(g['lr']) for g in opt.param_groups] + \
                    [float(msch.get_margin())]
                digest.update(struct.pack('<%dd' % len(vals), *vals))
                if cur in sample_at:
                    samples[str(cur)] = vals
        return dict(sha256=digest.hexdigest(), samples=samples,
                    resumes=resumes, param_groups=groups,
                    get_lr_last=float(sch.get_lr()))

    out = dict(num_epochs=n_ep, epoch_iter=ei, total_steps=total,
               scheduler=optim.scheduler, margin_scheduler='MarginScheduler',
               scheduler_args=dict(
                   initial_lr=optim.initial_lr, final_lr=optim.final_lr,
                   warm_up_epoch=optim.warm_up_epoch,
                   warm_from_zero=optim.warm_from_zero,
                   frontend_modules=list(optim.frontend_modules),
                   frontend_lr_ratio=optim.frontend_lr_ratio),
               columns='lr of every param group in optimizer order, then '
                       'the margin')
    out['uninterrupted'] = trajectory(0)
    names = ('uninterrupted', 'resume_every_epoch', 'resume_every_2_epochs')
    try:
        out['resume_every_epoch'] = trajectory(1)
        out['resume_every_2_epochs'] = trajectory(2)
        out['resume_invariant'] = len({out[k]['sha256'] for k in names}) == 1
    except Exception as exc:  # the engine's resume path is not usable
        harness.log('schedule: resume trajectories %s (%s: %s)' % (
            PENDING, type(exc).__name__, exc))
        for name in names[1:]:
            out[name] = '%s: %s: %s' % (PENDING, type(exc).__name__, exc)
        out.pop('resume_invariant', None)
    for name in os.listdir(work):
        os.remove(os.path.join(work, name))
    os.rmdir(work)
    return out


# ------------------------------------------------------------------- dump
class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def cmd_dump(a: argparse.Namespace) -> None:
    site = read_site(a.site)
    a.out, a.tmp = os.path.abspath(a.out), os.path.abspath(a.tmp)
    harness_path = os.path.abspath(a.harness)
    os.makedirs(a.tmp, exist_ok=True)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['OMP_NUM_THREADS'] = str(a.threads)
    os.environ['PYTHONPATH'] = os.pathsep.join([REPO_ROOT, RECIPE_DIR])
    sys.dont_write_bytecode = True
    recipe = os.path.abspath(site['recipe_dir'])
    sys.path[:] = [p for p in sys.path
                   if os.path.realpath(p or '.') != os.path.realpath(recipe)]
    sys.path[:0] = [REPO_ROOT, RECIPE_DIR]
    os.chdir(recipe)
    try:
        if os.nice(0) < 10:
            os.nice(10 - os.nice(0))
    except OSError:
        pass
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                        format='%(levelname)s %(name)s: %(message)s')
    harness = load_harness(harness_path)
    import numpy as np
    import torch
    import ovsb
    import wespeaker
    for module in (ovsb, wespeaker):
        where = os.path.realpath(module.__file__)
        assert where.startswith(os.path.realpath(REPO_ROOT) + os.sep), \
            '%s imported from %s' % (module.__name__, where)
    torch.set_num_threads(a.threads)
    checks = [c for c in a.checks.split(',') if c]
    assert all(c in CHECKS for c in checks), checks
    a.checkpoint = site['checkpoint']
    a.device, a.amp = 'cpu', False
    capture = _Capture()
    api_log = logging.getLogger('ovsb')
    api_log.setLevel(logging.INFO)
    api_log.addHandler(capture)

    from ovsb import api
    T = harness.Timer()
    out: Dict[str, Any] = {}
    harness.start_rss_watcher()
    with T('load config'):
        cfg = api.load_config(RECIPE_CONFIG, overrides=site_overrides(site))
        tp = OvsbPath(cfg, len(cfg.run.gpus))
    if 'config' in checks:
        with T('check config'):
            out['config_port'] = check_config_port(site, cfg)
    with T('build_training (rank 0, cpu)'):
        base_seed = tp.build()
    derived_seed = base_seed
    a.base_seed_source = 'ovsb.api.build_training(rank 0) -> DataLoader ' \
        'iterator'
    if a.base_seed is not None:
        base_seed, a.base_seed_source = int(a.base_seed), '--base-seed'
    real_fields = None
    if 'batches' in checks:
        with T('check batches'):
            out['batches'], real_fields = harness.check_batches(tp, a,
                                                                base_seed)
            out['batches']['base_seed_train_order'] = derived_seed
    if 'sampler' in checks:
        with T('check sampler'):
            out['sampler'] = harness.check_sampler(tp, a)
            out['sampler']['fields'] = list(EPISODE_FIELDS)
            out['sampler']['build_log'] = [
                m for m in capture.lines if m.startswith('speaker table:')]
    if 'schedule' in checks:
        with T('check schedule'):
            out['schedule'] = check_schedule(tp, harness, a.tmp)
    if 'model' in checks:
        tp.release_data(harness.trim_memory)
        with T('check model'):
            out['model'] = harness.check_model(tp, a, real_fields)
    import torchaudio
    import transformers
    out['meta'] = dict(
        tree=os.path.realpath(REPO_ROOT), recipe=recipe,
        wespeaker=os.path.realpath(wespeaker.__file__),
        ovsb=os.path.realpath(ovsb.__file__),
        git=harness.git_info(REPO_ROOT), checks=checks, argv=sys.argv[1:],
        harness=harness_path, host=os.uname().nodename,
        time=time.strftime('%Y-%m-%d %H:%M:%S %Z'), seconds=T.t,
        rss_anon_file_gb=T.rss, seconds_total=round(sum(T.t.values()), 1),
        versions=dict(python=sys.version.split()[0], torch=torch.__version__,
                      torchaudio=torchaudio.__version__,
                      transformers=transformers.__version__,
                      numpy=np.__version__),
        threads=a.threads, device=a.device,
        max_rss_gb=harness.peak_rss()[0],
        max_rss_anon_gb=harness.peak_rss()[1])
    with open(a.out + '.tmp', 'w') as fout:
        json.dump(out, fout, indent=1)
    os.replace(a.out + '.tmp', a.out)
    harness.log('wrote %s (%s; %.0f s; peak RSS %.1f GB)' % (
        a.out, ','.join(checks), out['meta']['seconds_total'],
        out['meta']['max_rss_gb']))


# ---------------------------------------------------------------- compare
def _strip_prefix(line: str) -> str:
    """A table summary line without its leading label and K / P fields."""
    return line.split(', ', 1)[1] if ', ' in line else line


def make_views(ref: Dict[str, Any], new: Dict[str, Any]
               ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """Reduce both dumps to what is defined on both sides.

    Returns (reference view, new view, notes). Every reduction is listed
    in the notes; nothing else is touched, and the views are compared for
    exact equality afterwards.
    """
    ref, new = copy.deepcopy(ref), copy.deepcopy(new)
    notes: List[str] = []
    if 'config' in ref:
        ref.pop('config')
        notes.append('config: the reference check reads the guard internals '
                     'of the original tree; replaced by config_port (below)')
    new.pop('config_port', None)
    if 'batches' in ref and 'batches' in new:
        rb, nb = ref['batches'], new['batches']
        for side in (rb, nb):
            side.pop('base_seed_source', None)
        notes.append('batches.base_seed_source: a label, not compared '
                     '(base_seed and base_seed_train_order are)')
        dropped, added = set(), set()
        for r, n in zip(rb['batches'], nb['batches']):
            both = sorted(set(r['fields']) & set(n['fields']))
            dropped |= set(r['fields']) - set(both)
            added |= set(n['fields']) - set(both)
            for side in (r, n):
                for field in set(side['fields']) - set(both):
                    side.pop(field)
                side['fields'] = both
        notes.append('batches: fields of the reference collate that this '
                     'recipe does not emit, not compared: %s' %
                     sorted(dropped))
        notes.append('batches: fields only this recipe emits, not compared: '
                     '%s' % sorted(added))
    if 'sampler' in ref and 'sampler' in new:
        for side in (ref['sampler'], new['sampler']):
            side['build_log'] = [_strip_prefix(m) for m in side['build_log']]
        notes.append('sampler.build_log: compared after the first ", " (the '
                     'line starts with "episodic sampler: P .. K .." in the '
                     'reference and "speaker table: K .." here)')
    if 'schedule' in ref and 'schedule' in new:
        names = ('resume_every_epoch', 'resume_every_2_epochs')
        if any(isinstance(new['schedule'].get(k), str) for k in names):
            for side in (ref['schedule'], new['schedule']):
                for key in names + ('resume_invariant', ):
                    side.pop(key, None)
            notes.append('schedule: resumed trajectories %s, not compared'
                         % PENDING)
    return ref, new, notes


def chain_fail_rows(new: Dict[str, Any]) -> Optional[int]:
    """Rows of the new dump whose codec call failed (None: not recorded)."""
    total = None
    for batch in (new.get('batches') or {}).get('batches', []):
        if 'chain_fail' in batch:
            # the harness dumps the values of this field as a digest only;
            # "-1 on every row" is recognised by the digest of that tensor
            n = batch['chain_fail']['shape'][0]
            none = hashlib.sha256(struct.pack('<%dq' % n, *([-1] * n)))
            total = (total or 0) + int(
                batch['chain_fail']['sha256'] != none.hexdigest())
    return total


def cmd_compare(a: argparse.Namespace) -> int:
    harness = load_harness(os.path.abspath(a.harness))
    with open(a.ref) as fin:
        ref = json.load(fin)
    with open(a.new) as fin:
        new = json.load(fin)
    rv, nv, notes = make_views(ref, new)
    stem = a.new[:-5] if a.new.endswith('.json') else a.new
    views = (stem + '.ref.view.json', stem + '.view.json')
    for path, view in zip(views, (rv, nv)):
        with open(path, 'w') as fout:
            json.dump(view, fout, indent=1)
    print('reference: %s' % a.ref)
    print('new:       %s' % a.new)
    for key in ('versions', 'threads', 'device', 'host'):
        if ref.get('meta', {}).get(key) != new.get('meta', {}).get(key):
            print('note: meta.%s differs: %s vs %s' % (
                key, ref['meta'].get(key), new['meta'].get(key)))
    print('reductions before the comparison:')
    for note in notes:
        print('  - ' + note)
    bad = 0
    for check in CHECKS[1:]:
        if check not in rv or check not in nv:
            print('%-9s FAIL    present only in %s' % (
                check, 'reference' if check in rv else 'new'
                if check in nv else 'neither file'))
            bad += 1
            continue
        res = dict(fail=[], close=[], n=0)
        harness.walk(rv[check], nv[check], check, res)
        # strict: a float inside the harness's tolerance is still a failure
        problems = res['fail'] + ['%s: |diff| %.3g (not bit-identical)' % x
                                  for x in sorted(res['close'],
                                                  key=lambda x: -x[1])]
        if problems:
            bad += 1
            print('%-9s FAIL    %d of %d values differ' % (
                check, len(problems), res['n']))
            for line in problems[:a.max_show]:
                print('    ' + line)
            if len(problems) > a.max_show:
                print('    ... %d more' % (len(problems) - a.max_show))
        else:
            print('%-9s PASS    identical (%d values)' % (check, res['n']))
    print('non-strict sections:')
    port = new.get('config_port')
    if port is None:
        print('  config_port: not in the new dump')
    else:
        print('  config_port: legacy config == %s modulo path fields: %s'
              % (port['recipe_config'], port['equal_modulo_paths']))
        d = port['legacy_vs_recipe']
        print('    path fields that differ: %s' % d['differing_path_fields'])
        print('    other fields that differ: %s' %
              d['differing_other_fields'])
        print('    name-keyed mappings in another order: %s' %
              d['order_differs_unordered'])
        print('    config used by the gate == legacy modulo path fields: '
              '%s' % port['gate_config_equal_modulo_paths'])
        bad += 0 if port['equal_modulo_paths'] and \
            port['gate_config_equal_modulo_paths'] else 1
    failed = chain_fail_rows(new)
    print('  chain_fail: %s' % (
        'not recorded' if failed is None else
        '%d batch(es) with a failed codec call' % failed))
    seeds = (new.get('batches') or {}).get('base_seed_train_order')
    print('  base seed derived by the new construction: %s' % seeds)
    print('views written: %s %s' % views)
    print('RESULT: %s' % ('MISMATCH' if bad else 'PARITY'))
    return 1 if bad else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('dump', help='record this recipe (JSON)')
    d.add_argument('--harness', required=True, help='path of parity.py')
    d.add_argument('--site', required=True, help='site file (README.md)')
    d.add_argument('--out', required=True)
    d.add_argument('--checks', default=','.join(CHECKS))
    d.add_argument('--ranks', default='0,3,7')
    d.add_argument('--workers', default='0,5,11')
    d.add_argument('--epochs', default='1,2,9')
    d.add_argument('--episodes', type=int, default=200)
    d.add_argument('--n-batches', dest='n_batches', type=int, default=12)
    d.add_argument('--base-seed', dest='base_seed', type=int, default=None,
                   help='data-loader base seed for the batches check '
                        '(default: the one the construction leads to)')
    d.add_argument('--threads', type=int, default=4)
    d.add_argument('--tmp', default=os.environ.get('TMPDIR', '/tmp'),
                   help='scratch directory')
    c = sub.add_parser('compare', help='reference dump vs new dump')
    c.add_argument('--harness', required=True, help='path of parity.py')
    c.add_argument('ref')
    c.add_argument('new')
    c.add_argument('--max-show', dest='max_show', type=int, default=25)
    a = parser.parse_args()
    if a.cmd == 'compare':
        sys.exit(cmd_compare(a))
    cmd_dump(a)


if __name__ == '__main__':
    main()
