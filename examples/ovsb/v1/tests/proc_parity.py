"""Parity harness of the audio processors (crop, zero-run, augment, channel).

One file, two roles:

* imported by the tests: synthetic sample streams, tiny LMDBs, and
  :func:`run_both`, which runs a list of case specs once per tree in a
  subprocess and returns both result lists;
* run as a script inside such a subprocess: ``--side new`` drives the
  ``ovsb.data`` processors, ``--side source`` the functions they were
  ported from (``PYTHONPATH`` = the source tree; the two trees'
  ``wespeaker`` packages are never imported into one process).

Both sides get the same seeded stream (dicts with the fields the episodic
sampler emits), seed ``random`` / numpy / torch the same way and report,
per output row, every metadata field plus a digest of the waveform bytes,
and digests of the three RNG states after the run.

Environment:
  OVSB_SOURCE_TREE     root of the source repository (parity tests skip
                       without it)
  OVSB_SOURCE_RECIPE   its recipe directory; enables the cases that read
                       the real noise / RIR LMDBs (data/musan/lmdb,
                       data/rirs/lmdb), read-only
"""

import argparse
import hashlib
import io
import os
import pickle
import random
import subprocess
import sys
import types
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

RECIPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.abspath(os.path.join(RECIPE_DIR, '..', '..', '..'))
SOURCE_TREE = os.environ.get('OVSB_SOURCE_TREE')
SOURCE_RECIPE = os.environ.get('OVSB_SOURCE_RECIPE')
SR = 16000
ARM_NAMES = ('vox2', 'vb2', '3dspeaker', 'cn2', 'kespeech', 'mandarin_t1',
             'tvx', 'cn1')
ARM_INDEX = {name: i for i, name in enumerate(ARM_NAMES)}

# the registered recipe's processor settings (schema field names)
CROP = dict(
    max_secs=[[2.0, 0.15], [3.0, 0.45], [4.0, 0.25], [6.0, 0.15]],
    min_sec=0.5, full_prob=0.5, onset_prob=0.5, onset_max_sec=1.0,
    pair_prob=0.25, pair_anchor_sec=6.0, pair_view_secs=[0.5, 2.0],
    pair_view_onset_prob=1.0,
    arm_caps={'cn1': [[2.0, 0.3], [3.0, 0.4], [4.0, 0.2], [6.0, 0.1]],
              'cn2': [[2.0, 0.3], [3.0, 0.4], [4.0, 0.2], [6.0, 0.1]]},
    energy_gate=dict(gate_below_sec=1.5, min_active_frac=0.5, max_redraws=4))
ZERO_RUN = dict(arms=None, min_run_ms=50.0, lsb_level=4.0, dither_lsb=1.0)
AUG_PROBS = dict(noise=0.2, reverb=0.2, reverb_noise=0.2)
AUG_ARM_PROBS = {n: dict(noise=0.2)
                 for n in ('3dspeaker', 'cn1', 'cn2', 'tvx')}
CHANNEL = dict(
    arms=['mandarin_t1'], prob=1.0, kpos=[1], distinct=True,
    codecs=['mp3', 'opus', 'vorbis', 'mulaw8k', 'g722'], eq_prob=0.6,
    eq_gain_db=6.0, bandlimit_prob=0.4, clip_prob=0.0,
    clip_level=[0.3, 0.7], loss_prob=0.0, loss_rate=[0.02, 0.08],
    isolate_ffmpeg=True)
FILTER = dict(min_num_frames=50, max_num_frames=400, frame_shift=20)


# ------------------------------------------------------------ test inputs
def _quantise(x: np.ndarray) -> np.ndarray:
    """float -> int16 grid in [-1, 1) units, as decoded 16-bit audio."""
    return (np.clip(np.round(x * 32768.0), -32768, 32767) /
            32768.0).astype(np.float32)


def _speech(rs: np.random.RandomState, n: int, sr: int) -> np.ndarray:
    """Noise bursts separated by near-silent pauses (energy onsets)."""
    x = rs.randn(n) * 0.1
    env = np.ones(n)
    pos = int(rs.uniform(0.0, 0.4) * sr)
    env[:pos] = 0.001
    while pos < n:
        pos += int(rs.uniform(0.3, 1.5) * sr)
        gap = int(rs.uniform(0.15, 0.6) * sr)
        env[pos:pos + gap] = 0.001
        pos += gap
    return x * env


def make_wave(rs: np.random.RandomState, kind: str, sr: int = SR
              ) -> np.ndarray:
    """One synthetic utterance (float32, int16 grid) of the given kind."""
    if kind == 'speech':
        x = _speech(rs, int(rs.uniform(1.0, 9.5) * sr), sr)
    elif kind == 'short':  # below the 1-s filter
        x = _speech(rs, int(rs.uniform(0.3, 0.99) * sr), sr)
    elif kind == 'tiny':  # fewer than 8 energy frames
        x = rs.randn(int(rs.uniform(0.05, 0.15) * sr)) * 0.1
    elif kind == 'sparse':  # one burst in near-silence: energy gate
        n = int(rs.uniform(3.0, 7.0) * sr)
        x = rs.randn(n) * 2e-4
        a = rs.randint(0, n - sr)
        b = a + int(rs.uniform(0.3, 0.6) * sr)
        x[a:b] = rs.randn(b - a) * 0.2
    elif kind == 'zrun':  # speech with exact-zero runs
        x = _speech(rs, int(rs.uniform(1.5, 7.0) * sr), sr)
        for _ in range(rs.randint(1, 4)):
            a = rs.randint(0, x.shape[0] - sr // 4)
            x[a:a + int(rs.uniform(0.03, 1.0) * sr)] = 0.0
    elif kind == 'silent':  # all zeros
        x = np.zeros(int(rs.uniform(1.2, 3.0) * sr))
    elif kind == 'faint':  # peak below 8 LSB, with a zero run
        n = int(rs.uniform(1.2, 3.0) * sr)
        x = rs.randint(-4, 5, n) / 32768.0
        x[n // 2:] = 0.0
    else:
        raise ValueError(kind)
    return _quantise(x)


PROFILES = {
    # kind -> probability
    'mixed': dict(speech=0.5, short=0.08, sparse=0.14, zrun=0.16,
                  silent=0.06, faint=0.06),
    'crop': dict(speech=0.45, short=0.1, tiny=0.05, sparse=0.2, zrun=0.1,
                 silent=0.05, faint=0.05),
    'zeros': dict(speech=0.2, zrun=0.4, silent=0.2, faint=0.2),
    'speech': dict(speech=0.8, zrun=0.2),
}


def make_stream(seed: int, n: int, profile: str = 'mixed',
                arms: Optional[Sequence[int]] = None, rate_8k: float = 0.0
                ) -> Iterator[Dict[str, Any]]:
    """Seeded stream shaped like the episodic sampler's output.

    Rows come in K = 2 groups of one speaker and arm. Fields: key, spk,
    wav (1, T) float32, sample_rate, arm, lang, vid and the construction
    fields kpos, ptype, seg, dev, dcls, dial, genre.
    """
    import torch
    rs = np.random.RandomState(seed)
    kinds = sorted(PROFILES[profile])
    probs = [PROFILES[profile][k] for k in kinds]
    arms = list(range(len(ARM_NAMES))) if arms is None else list(arms)
    spk, arm = '', 0
    for i in range(n):
        if i % 2 == 0:
            arm = int(arms[rs.randint(len(arms))])
            spk = 'spk{:05d}'.format(rs.randint(100000))
        sr = 8000 if rs.rand() < rate_8k else SR
        wav = make_wave(rs, kinds[rs.choice(len(kinds), p=probs)], sr)
        yield dict(key='{}_utt{:06d}'.format(spk, i), spk=spk,
                   wav=torch.from_numpy(wav).unsqueeze(0), sample_rate=sr,
                   arm=arm, lang=int(rs.randint(-1, 5)),
                   vid=int(rs.randint(0, 1000)), kpos=i % 2,
                   ptype=int(rs.randint(0, 4)), seg=int(rs.randint(-1, 9)),
                   dev=int(rs.randint(-1, 4)), dcls=int(rs.randint(-1, 3)),
                   dial=int(rs.randint(-1, 2)), genre=int(rs.randint(-1, 11)))


class FakeSource:
    """In-memory stand-in of ``LmdbData`` with the same use of ``random``."""

    def __init__(self, items):
        self.items = list(items)
        self.keys = [k for k, _ in self.items]
        store = dict(self.items)

        class _Txn:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get(self, key):
                return store.get(key.decode())

        class _Db:
            def begin(self, write=False):
                return _Txn()

        self.db = _Db()

    def random_one(self):
        return self.items[random.randint(0, len(self.keys) - 1)]


def wav_bytes(x: np.ndarray, sr: int = SR) -> bytes:
    """A wav file (int16 unless ``x`` is float) as bytes."""
    from scipy.io import wavfile
    buf = io.BytesIO()
    wavfile.write(buf, sr, x)
    return buf.getvalue()


def write_lmdb(path: str, items: Sequence[Any]) -> str:
    """An LMDB in the layout of ``LmdbData``: ``__keys__`` + key -> bytes."""
    import lmdb
    os.makedirs(path, exist_ok=True)
    env = lmdb.open(path, map_size=1 << 30)
    with env.begin(write=True) as txn:
        for key, value in items:
            txn.put(key.encode(), value)
        txn.put(b'__keys__', pickle.dumps([k for k, _ in items]))
    env.close()
    return path


def make_aug_lmdbs(root: str, seed: int = 0) -> Dict[str, Any]:
    """Tiny noise LMDB, two RIR LMDBs and a noise exclusion list.

    Noise: three categories, 16 kHz plus one 8-kHz item. RIRs: several
    rooms with a few positions each, one single-position room, one 8-kHz
    RIR; the second LMDB repeats one key of the first.
    """
    rs = np.random.RandomState(seed)
    os.makedirs(root, exist_ok=True)
    noise = []
    for cat in ('noise', 'music', 'speech'):
        for j in range(4):
            sr = 8000 if (cat, j) == ('noise', 3) else SR
            x = (rs.randn(int(sr * rs.uniform(2.0, 9.0))) * 3000).astype(
                np.int16)
            noise.append(('{}/set{}/{}-{:04d}.wav'.format(cat, j % 2, cat, j),
                          wav_bytes(x, sr)))

    def rir(sr: int = SR) -> bytes:
        n = int(0.25 * sr)
        h = rs.randn(n) * np.exp(-np.arange(n) / (0.04 * sr))
        h[int(0.01 * sr)] = 5.0
        return wav_bytes((h / np.abs(h).max() * 3e4).astype(np.int16), sr)

    sim, real = [], []
    for room, count in (('smallroom/Room001', 4), ('mediumroom/Room002', 3),
                        ('largeroom/Room003', 1)):
        for j in range(count):
            sim.append(('{}/{}-{:05d}.wav'.format(room, room[-7:], j), rir()))
    for room, count in (('hall/A', 3), ('office/B', 2)):
        for j in range(count):
            real.append(('{}/pos{:02d}.wav'.format(room, j), rir()))
    real.append(('office/B/pos8k.wav', rir(8000)))
    real.append(sim[1])  # a key both sources hold
    exclude = os.path.join(root, 'noise_exclude.txt')
    with open(exclude, 'w') as f:
        f.write('# keys held out of training\n')
        f.write('\n'.join([noise[1][0], noise[6][0], 'not/a/key.wav']) + '\n')
    return dict(noise_lmdb=write_lmdb(os.path.join(root, 'noise'), noise),
                reverb_lmdb=write_lmdb(os.path.join(root, 'rir_sim'), sim),
                reverb_real=write_lmdb(os.path.join(root, 'rir_real'), real),
                noise_exclude_list=exclude)


def augment_cfg(lmdbs: Dict[str, Any], **over: Any) -> Dict[str, Any]:
    """An ``AugmentCfg`` mapping over the LMDBs of :func:`make_aug_lmdbs`."""
    cfg = dict(probs=dict(AUG_PROBS), arm_probs=dict(AUG_ARM_PROBS),
               reverberant_noise=True,
               reverb_sources=['simulated', lmdbs['reverb_real']],
               reverb_weights=[0.5, 0.5], noise_lmdb=lmdbs['noise_lmdb'],
               reverb_lmdb=lmdbs['reverb_lmdb'],
               noise_exclude_list=lmdbs['noise_exclude_list'])
    cfg.update(over)
    return cfg


# ------------------------------------------------------------ result coding
def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: Any) -> Any:
    """A picklable, exactly comparable form of a sample field."""
    import torch
    if isinstance(value, torch.Tensor):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return ('array', str(value.dtype), tuple(value.shape),
                _digest(np.ascontiguousarray(value).tobytes()))
    if isinstance(value, (float, np.floating)):
        return ('float', repr(float(value)))
    if isinstance(value, (np.integer, )):
        return int(value)
    return value


def rng_digests() -> Dict[str, str]:
    """Digests of the python / numpy / torch global RNG states."""
    import torch
    state = np.random.get_state()
    return dict(
        python=_digest(repr(random.getstate()).encode()),
        numpy=_digest(state[1].tobytes() + repr(state[2:]).encode()),
        torch=_digest(torch.get_rng_state().numpy().tobytes()))


def seed_all(seed: int) -> None:
    """Seed the three global RNGs."""
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


# ----------------------------------------------------------- stage builders
def _sec(x: float) -> int:
    return int(round(float(x) * SR))


def _cached(stages: Any, aug: Dict[str, Any], opener: Any) -> Any:
    """(reverb, noise) sources, opened once per process and LMDB set.

    An LMDB environment cannot be opened twice in one process.
    """
    cache = stages.__dict__.setdefault('_sources', {})
    key = repr([aug[k] for k in ('reverb_lmdb', 'reverb_sources',
                                 'reverb_weights', 'noise_lmdb',
                                 'noise_exclude_list')])
    if key not in cache:
        cache[key] = opener(aug)
    return cache[key]


class _SourceStages:
    """The source functions, called the way the source ``Dataset`` does."""

    def __init__(self) -> None:
        from wespeaker.dataset import processor
        from wespeaker.dataset.lmdb_data import LmdbData, MultiLmdbData
        self.p, self.lmdb, self.multi = processor, LmdbData, MultiLmdbData

    def filter(self, data: Any, spec: Dict[str, Any]) -> Any:
        return self.p.filter(data, data_type='shard', **spec['filter'])

    def spk_to_id(self, data: Any, spec: Dict[str, Any]) -> Any:
        return self.p.spk_to_id(data, spec['spk2id'], None)

    def resample(self, data: Any, spec: Dict[str, Any]) -> Any:
        return self.p.resample(data, SR)

    def crop(self, data: Any, spec: Dict[str, Any]) -> Any:
        vc = spec['crop']
        caps = vc['max_secs']
        arm_caps = {ARM_INDEX[a]: ([_sec(c[0]) for c in v],
                                   [float(c[1]) for c in v])
                    for a, v in (vc.get('arm_caps') or {}).items()} or None
        gate = vc.get('energy_gate')
        if gate:
            gate = dict(gate_below=_sec(gate['gate_below_sec']),
                        min_active_frac=float(gate['min_active_frac']),
                        max_redraws=int(gate['max_redraws']))
        return self.p.random_chunk_var(
            data, [_sec(c[0]) for c in caps], [float(c[1]) for c in caps],
            spec['batch_size'], _sec(vc['min_sec']),
            full_prob=vc['full_prob'], onset_prob=vc['onset_prob'],
            onset_max_len=_sec(vc['onset_max_sec']),
            pair_prob=vc['pair_prob'],
            pair_anchor_len=_sec(vc['pair_anchor_sec']),
            pair_view_lens=[_sec(v) for v in vc['pair_view_secs']],
            sample_rate=SR, seed=spec['crop_seed'], sync_ranks=True,
            arm_caps=arm_caps,
            pair_view_onset_prob=vc.get('pair_view_onset_prob'),
            par_jitter=None, energy_gate=gate or None, pair_law=None,
            epoch=spec['epoch'])

    def zero_run(self, data: Any, spec: Dict[str, Any]) -> Any:
        zr = spec['zero_run']
        arms = zr['arms']
        if arms is not None:
            arms = [ARM_INDEX[a] for a in arms]
        kw = {} if spec.get('rank') is None else dict(rank=spec['rank'])
        return self.p.zero_run_guard(
            data, arms=arms, dither=spec.get('dither', True),
            min_run_ms=float(zr['min_run_ms']),
            lsb_level=float(zr['lsb_level']),
            dither_lsb=float(zr['dither_lsb']), sample_rate=SR,
            stats=spec.get('stats', True), seed=spec['crop_seed'],
            epoch=spec['epoch'], **kw)

    def sources(self, aug: Dict[str, Any]) -> Any:
        return _cached(self, aug, self._open)

    def _open(self, aug: Dict[str, Any]) -> Any:
        reverb = self.multi(
            [self.lmdb(aug['reverb_lmdb'] if s == 'simulated' else s)
             for s in aug['reverb_sources']], aug['reverb_weights'])
        noise = self.lmdb(aug['noise_lmdb'],
                          exclude_keys=aug.get('noise_exclude_list'))
        return reverb, noise

    def draw(self, data: Any, spec: Dict[str, Any]) -> Any:
        aug = spec['augment']
        arm_probs = {ARM_INDEX[a]: p
                     for a, p in aug['arm_probs'].items()} or None
        return self.p.draw_aug_type(data, aug['probs'], arm_probs)

    def reverb_noise(self, data: Any, spec: Dict[str, Any]) -> Any:
        aug = spec['augment']
        reverb, noise = self.sources(aug)
        kw = dict(reverberant_noise=True) if aug['reverberant_noise'] else {}
        return self.p.add_reverb_noise(
            data, reverb, noise, SR, 0.6, aug_mode='mixed', snr_ranges={},
            overlap_noise_prob=0.5, farfield_args=None, **kw)

    def channel(self, data: Any, spec: Dict[str, Any]) -> Any:
        asym = dict(spec['channel'])
        arms = [ARM_INDEX[a] for a in asym.pop('arms')]
        return self.p.add_channel_asym(
            data, arms, kpos=tuple(asym.pop('kpos')), resample_rate=SR,
            **asym)

    def apply_channel(self, x: np.ndarray, **kw: Any) -> np.ndarray:
        return self.p.apply_channel(x, SR, **kw)


class _NewStages:
    """The ported processors, driven through the typed config sections."""

    def __init__(self) -> None:
        from ovsb.config import schema
        from ovsb.data import augment, channel, crop, lmdb_multi, zero_run
        from wespeaker.dataset import processor
        self.schema, self.core = schema, processor
        self.augment, self.chan, self.crop_mod = augment, channel, crop
        self.lmdb_multi, self.zr = lmdb_multi, zero_run

    def _cfg(self, cls: str, tree: Dict[str, Any]) -> Any:
        return self.schema.build_section(getattr(self.schema, cls), tree,
                                         cls)

    def filter(self, data: Any, spec: Dict[str, Any]) -> Any:
        return self.core.filter(data, data_type='shard', **spec['filter'])

    def spk_to_id(self, data: Any, spec: Dict[str, Any]) -> Any:
        return self.core.spk_to_id(data, spec['spk2id'])

    def resample(self, data: Any, spec: Dict[str, Any]) -> Any:
        return self.core.resample(data, SR)

    def crop(self, data: Any, spec: Dict[str, Any]) -> Any:
        kw = self.crop_mod.crop_args(
            self._cfg('CropCfg', spec['crop']), ARM_INDEX,
            spec['batch_size'], SR, spec['crop_seed'])
        return self.crop_mod.random_chunk_var(data, epoch=spec['epoch'],
                                              **kw)

    def zero_run(self, data: Any, spec: Dict[str, Any]) -> Any:
        kw = self.zr.zero_run_args(
            self._cfg('ZeroRunCfg', spec['zero_run']), ARM_INDEX, SR,
            spec['crop_seed'])
        kw.update(dither=spec.get('dither', True),
                  stats=spec.get('stats', True))
        return self.zr.zero_run_guard(data, epoch=spec['epoch'],
                                      rank=spec.get('rank'), **kw)

    def sources(self, aug: Dict[str, Any]) -> Any:
        return _cached(self, aug, self._open)

    def _open(self, aug: Dict[str, Any]) -> Any:
        cfg = self._cfg('AugmentCfg', aug)
        return (self.lmdb_multi.open_reverb_source(cfg),
                self.lmdb_multi.open_noise_source(cfg))

    def draw(self, data: Any, spec: Dict[str, Any]) -> Any:
        cfg = self._cfg('AugmentCfg', spec['augment'])
        return self.augment.draw_aug_type(
            data, **self.augment.draw_args(cfg, ARM_INDEX))

    def reverb_noise(self, data: Any, spec: Dict[str, Any]) -> Any:
        aug = spec['augment']
        reverb, noise = self.sources(aug)
        return self.augment.add_reverb_noise(
            data, reverb, noise, SR,
            reverberant_noise=aug['reverberant_noise'])

    def channel(self, data: Any, spec: Dict[str, Any]) -> Any:
        cfg = self._cfg('ChannelCfg', spec['channel'])
        self.chan.require_ffmpeg(cfg)
        return self.chan.add_channel_asym(
            data, **self.chan.channel_args(cfg, ARM_INDEX, SR))

    def apply_channel(self, x: np.ndarray, **kw: Any) -> np.ndarray:
        return self.chan.apply_channel(x, SR, **kw)


# ------------------------------------------------------------------ cases
def _run_chain(stages: Any, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Stream -> the listed stages -> rows (all fields) + RNG digests."""
    import torch
    st = spec['stream']
    data = make_stream(st['seed'], st['n'], st.get('profile', 'mixed'),
                       st.get('arms'), st.get('rate_8k', 0.0))
    if 'spk_to_id' in spec['stages']:
        names = sorted({d['spk'] for d in make_stream(
            st['seed'], st['n'], st.get('profile', 'mixed'), st.get('arms'),
            st.get('rate_8k', 0.0))})
        # every 9th speaker is not a class: its rows are dropped
        spec = dict(spec, spk2id={s: i for i, s in enumerate(names)
                                  if i % 9 != 4})
    orig = torch.utils.data.get_worker_info
    if spec.get('worker_id') is not None:
        fake = types.SimpleNamespace(id=spec['worker_id'], num_workers=12)
        torch.utils.data.get_worker_info = lambda: fake
    try:
        seed_all(spec['seed'])
        for name in spec['stages']:
            data = getattr(stages, name)(data, spec)
        rows = [{k: encode(v) for k, v in row.items()} for row in data]
    finally:
        torch.utils.data.get_worker_info = orig
    return dict(rows=rows, rng=rng_digests())


def channel_inputs(seed: int, n: int) -> List[np.ndarray]:
    """Fixed crops (0.5 - 6 s) for the codec chain, plus a square wave."""
    rs = np.random.RandomState(seed)
    out = []
    for i in range(n):
        if i % 5 == 4:
            t = np.arange(int(rs.uniform(0.5, 3.0) * SR)) / SR
            out.append(np.sign(np.sin(2 * np.pi * 440 * t)).astype(
                np.float32) * 0.5)
            continue
        w = make_wave(rs, 'speech' if i % 2 else 'zrun')
        m = min(w.shape[0], int(rs.uniform(0.5, 6.0) * SR))
        a = rs.randint(0, w.shape[0] - m + 1)
        out.append(w[a:a + m].copy())
    return out


def _run_apply_channel(stages: Any, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Each (codec, input) twice per mode: output arrays + RNG digests."""
    inputs = channel_inputs(spec['input_seed'], spec['n_inputs'])
    calls = []
    for codec in spec['codecs']:
        for i, x in enumerate(inputs):
            for isolate in spec['isolate']:
                outs, rngs = [], []
                for _ in range(spec.get('repeats', 2)):
                    seed_all(spec['seed'] + i)
                    kw = dict(spec['channel_args'], codec_prob=1.0,
                              codecs=(codec, ))
                    if isolate:
                        kw['isolate_ffmpeg'] = True
                    outs.append(stages.apply_channel(x.copy(), **kw))
                    rngs.append(rng_digests())
                calls.append(dict(codec=codec, input=i, isolate=isolate,
                                  out=outs, rng=rngs))
    return dict(calls=calls)


def _run_multi_lmdb(stages: Any, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Draws of the RIR / noise sources: keys, value digests, RNG."""
    reverb, noise = stages.sources(spec['augment'])
    seed_all(spec['seed'])
    draws = []
    for _ in range(spec['n']):
        key, value = reverb.random_one()
        draws.append((key, _digest(value)))
        key, value = noise.random_one()
        draws.append((key, _digest(value)))
    lookups = []
    with reverb.db.begin(write=False) as txn:
        for key in list(reverb.keys) + ['no/such/key.wav']:
            value = txn.get(key.encode())
            lookups.append((key, None if value is None else _digest(value)))
    return dict(draws=draws, lookups=lookups, keys=list(reverb.keys),
                cum=[repr(c) for c in reverb.cum],
                noise_keys=list(noise.keys), num_excluded=noise.num_excluded,
                rng=rng_digests())


_KINDS = {'chain': _run_chain, 'apply_channel': _run_apply_channel,
          'multi_lmdb': _run_multi_lmdb}


def run_specs(side: str, specs: Sequence[Dict[str, Any]]
              ) -> List[Dict[str, Any]]:
    """Run the case specs in this process with the given tree's code."""
    stages = _SourceStages() if side == 'source' else _NewStages()
    return [_KINDS[spec.get('kind', 'chain')](stages, spec) for spec in specs]


# ------------------------------------------------------------- test driver
def have_source() -> bool:
    """Whether the source tree is configured."""
    return bool(SOURCE_TREE) and os.path.isdir(SOURCE_TREE)


def run_side(side: str, specs: Sequence[Dict[str, Any]], workdir: str
             ) -> List[Dict[str, Any]]:
    """Run ``specs`` with one tree's code in a fresh interpreter."""
    spec_path = os.path.join(workdir, 'specs_{}.pkl'.format(side))
    out_path = os.path.join(workdir, 'out_{}.pkl'.format(side))
    with open(spec_path, 'wb') as f:
        pickle.dump(list(specs), f)
    env = dict(os.environ)
    tree = [SOURCE_TREE] if side == 'source' else [RECIPE_DIR, REPO_ROOT]
    env['PYTHONPATH'] = os.pathsep.join(tree)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.setdefault('OMP_NUM_THREADS', '2')
    cmd = [sys.executable, os.path.abspath(__file__), '--side', side,
           '--specs', spec_path, '--out', out_path]
    proc = subprocess.run(cmd, env=env, cwd=workdir, capture_output=True,
                          text=True)
    if proc.returncode != 0:
        raise RuntimeError('{} side failed:\n{}\n{}'.format(
            side, proc.stdout[-2000:], proc.stderr[-6000:]))
    with open(out_path, 'rb') as f:
        return pickle.load(f)


def run_both(specs: Sequence[Dict[str, Any]], workdir: str) -> Any:
    """(source results, new results) of the same specs."""
    return (run_side('source', specs, workdir),
            run_side('new', specs, workdir))


def diff_rows(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]],
              ignore: Sequence[str] = ()) -> List[str]:
    """Human-readable differences of two row lists (empty = equal)."""
    out = []
    if len(a) != len(b):
        out.append('row count {} != {}'.format(len(a), len(b)))
    for i, (x, y) in enumerate(zip(a, b)):
        keys = (set(x) | set(y)) - set(ignore)
        for k in sorted(keys):
            if x.get(k, '<missing>') != y.get(k, '<missing>'):
                out.append('row {} field {}: {!r} != {!r}'.format(
                    i, k, x.get(k, '<missing>'), y.get(k, '<missing>')))
        if len(out) > 20:
            break
    return out


def main() -> None:
    """Subprocess entry: run pickled specs with one tree, pickle results."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--side', choices=('source', 'new'), required=True)
    ap.add_argument('--specs', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    import wespeaker
    root = os.path.realpath(
        SOURCE_TREE if args.side == 'source' else REPO_ROOT)
    assert os.path.realpath(wespeaker.__file__).startswith(root + os.sep), \
        (args.side, wespeaker.__file__)
    with open(args.specs, 'rb') as f:
        specs = pickle.load(f)
    results = run_specs(args.side, specs)
    with open(args.out, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    # the script's directory is sys.path[0]; the trees come from PYTHONPATH
    main()
