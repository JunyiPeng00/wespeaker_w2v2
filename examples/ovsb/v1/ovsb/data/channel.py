"""Device / transmission chain with a lossy codec on rows of listed arms.

:func:`add_channel_asym` is the last waveform stage (after the additive
noise / reverberation stage), 1:1 and order-preserving. In a speaker's
K-group only the rows at the listed positions get the chain, so the two
utterances of a single-session speaker differ in channel.

The codec round-trip needs the FFmpeg shared libraries through
``torchaudio.io.AudioEffector``; :func:`require_ffmpeg` checks that at
startup. With ``isolate_ffmpeg`` the call runs in a helper process
(:mod:`ovsb.data.ffmpeg_isolation`): same code, same output, same use of
the random stream.

Every random draw of the chain comes from the global ``random`` module, in
a fixed order, whether or not the drawn step is applied: probabilities of
0.0 still consume their draw.

One deliberate difference from the code this was ported from: when the
codec call fails, the row is not tagged with ``sample['chain']`` (it is
tagged ``sample['chain_fail']`` instead) and the failure is counted and
logged at WARNING level at a bounded rate. The waveform and the random
stream of a failed call are unchanged.
"""

import logging
import math
import os
import random
import time
from functools import partial
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
from scipy.signal import resample_poly

from ovsb.config import schema
from ovsb.config.schema import ChannelCfg
from ovsb.data import ffmpeg_isolation

Sample = Dict[str, Any]

_CODECS = {
    # name: (format, encoder, bit-rate range in bps or None, sample rate)
    'mp3': ('mp3', None, (16000, 64000), 16000),
    'opus': ('ogg', 'opus', (8000, 24000), 16000),
    'vorbis': ('ogg', 'vorbis', (24000, 48000), 16000),
    'mulaw8k': ('wav', 'pcm_mulaw', None, 8000),
    'g722': ('g722', 'g722', None, 16000),
}
# sample['chain'] -> integer code of the batch (index in this tuple)
CHAIN_CODECS = ('mp3', 'opus', 'vorbis', 'mulaw8k', 'g722')
assert CHAIN_CODECS == tuple(_CODECS) == tuple(schema.CODECS)

_FFMPEG_OK: Optional[bool] = None
_MAX_FAIL_LINES = 20  # first N codec failures per process logged in full
_FAIL_LOG_EVERY = 300.0  # s, then at most one line per interval
_STAT_KEYS = ('rows', 'effector_calls', 'effector_failures', 'codec_failures')
_STATS: Dict[str, Any] = {'pid': None}
_FAIL_LOG = {'n': 0, 'last': 0.0, 'suppressed': 0}


def _ffmpeg_ok() -> bool:
    """Whether torchaudio can reach FFmpeg (probed once per process)."""
    global _FFMPEG_OK
    if _FFMPEG_OK is None:
        try:
            from torchaudio.io import AudioEffector
            AudioEffector(format='wav', encoder='pcm_mulaw').apply(
                torch.zeros(1600, 1), 16000)
            _FFMPEG_OK = True
        except Exception as e:  # noqa
            logging.warning('channel chain: FFmpeg unavailable ({}), codecs '
                            'fall back to numpy band-limit + mu-law'.format(
                                str(e)[:80]))
            _FFMPEG_OK = False
    return _FFMPEG_OK


def require_ffmpeg(cfg: ChannelCfg) -> None:
    """Fail at startup when the configured chain cannot run its codecs.

    Without FFmpeg the chain would silently degrade to the numpy fallback.
    No probe is made when the chain is off (``prob`` 0, no isolation).
    """
    if cfg.isolate_ffmpeg or float(cfg.prob) > 0.0:
        if not _ffmpeg_ok():
            raise RuntimeError(
                'FFmpeg is unavailable to torchaudio but the config sets '
                'data.channel: the codec chain would silently degrade to '
                'the numpy fallback. Make the FFmpeg 6 shared libraries '
                'loadable (e.g. load the FFmpeg module) or set '
                'data.channel.prob to 0.')


def _stats() -> Dict[str, Any]:
    if _STATS['pid'] != os.getpid():  # new (forked) process: own counters
        _STATS.clear()
        _STATS.update({k: 0 for k in _STAT_KEYS})
        _STATS['pid'] = os.getpid()
        _FAIL_LOG.update(n=0, last=0.0, suppressed=0)
    return _STATS


def get_stats() -> Dict[str, int]:
    """Chain counters of this process (= of one DataLoader worker).

    rows: rows sent through :func:`apply_channel`; effector_calls /
    effector_failures: FFmpeg calls and those that raised;
    codec_failures: rows whose drawn codec was not applied. The counters
    of the helper process are included with the prefix ``ffx_``.
    """
    out = {k: v for k, v in _stats().items() if k != 'pid'}
    out.update(('ffx_' + k, v)
               for k, v in ffmpeg_isolation.get_stats().items())
    return out


def _effector_failed(codec: Optional[str], error: BaseException) -> None:
    """Count a failed FFmpeg call; WARNING at a bounded rate."""
    st = _stats()
    st['effector_failures'] += 1
    if codec:
        st['codec_failures'] += 1
    now = time.time()
    _FAIL_LOG['n'] += 1
    if _FAIL_LOG['n'] <= _MAX_FAIL_LINES or \
            now - _FAIL_LOG['last'] >= _FAIL_LOG_EVERY:
        sup = _FAIL_LOG['suppressed']
        _FAIL_LOG.update(last=now, suppressed=0)
        logging.warning(
            'channel chain[pid %d]: %s failed, codec not applied: %s%s | %s',
            os.getpid(), codec or 'effect', str(error)[:200],
            ' (+%d failures not logged)' % sup if sup else '',
            ' '.join('%s=%d' % kv for kv in get_stats().items()))
    else:
        _FAIL_LOG['suppressed'] += 1


def _mulaw(x: np.ndarray, bits: int = 8) -> np.ndarray:
    """mu-law companding round-trip (numpy fallback of the mulaw codec)."""
    mu = 2**bits - 1
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round((y + 1) / 2 * mu) / mu * 2 - 1
    return np.sign(y) * ((1 + mu)**np.abs(y) - 1) / mu


def apply_channel(audio: np.ndarray,
                  sample_rate: int = 16000,
                  eq_prob: float = 0.6,
                  eq_gain_db: float = 6.0,
                  bandlimit_prob: float = 0.4,
                  codec_prob: float = 0.8,
                  codecs: Sequence[str] = CHAIN_CODECS,
                  clip_prob: float = 0.3,
                  clip_level: Sequence[float] = (0.3, 0.7),
                  loss_prob: float = 0.2,
                  loss_rate: Sequence[float] = (0.02, 0.08),
                  isolate_ffmpeg: bool = False,
                  isolate_timeout: Optional[float] = None,
                  isolate_dump_dir: Optional[str] = None,
                  info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Device / transmission chain on a 1-D float waveform (same length out).

    EQ (3 random peaking bands or a spectral tilt, +-``eq_gain_db``) ->
    8-kHz band-limit -> lossy codec round-trip -> clipping at a fraction of
    the peak -> 20-ms packet loss (zeroed frames) -> peak normalisation.

    isolate_ffmpeg: run the FFmpeg call in the per-process helper; a native
        crash or hang then costs a helper restart and one retry instead of
        the DataLoader worker. ``isolate_timeout`` (s per call) and
        ``isolate_dump_dir`` (inputs that killed the helper) go with it.
    info: if a dict, receives ``codec`` (the drawn codec or None) and
        ``codec_applied`` (True only if FFmpeg ran that codec).

    If the FFmpeg call raises, the EQ and the codec are skipped (a drawn
    band-limit is done by polyphase resampling) and the failure is counted.

    Draws, in order (``random``): EQ decision; if EQ: kind, then 3 x
    (frequency, gain) or one tilt gain; band-limit decision; codec
    decision; codec choice (if decided and ``codecs`` is not empty); bit
    rate (codecs with a rate range, only when FFmpeg is used); clip
    decision (+ level); loss decision (+ rate and one draw per frame).
    """
    n = audio.shape[0]
    x = audio.astype(np.float32)
    peak = float(np.max(np.abs(x))) + 1e-6
    x = x / peak * 0.9
    filters = []
    if eq_prob > random.random():
        if random.random() < 0.5:
            for _ in range(3):
                f = math.exp(random.uniform(math.log(150), math.log(6000)))
                filters.append('equalizer=f={:.0f}:t=o:w=1:g={:.1f}'.format(
                    f, random.uniform(-eq_gain_db, eq_gain_db)))
        else:  # tilt: opposite bass / treble shelves
            g = random.uniform(-eq_gain_db, eq_gain_db)
            filters.append('bass=g={:.1f}:f=300'.format(-g))
            filters.append('treble=g={:.1f}:f=2000'.format(g))
    narrow = bandlimit_prob > random.random()
    codec = random.choice(list(codecs)) if (
        codecs and codec_prob > random.random()) else None
    if codec == 'mulaw8k':
        narrow = True
    st = _stats()
    st['rows'] += 1
    applied = False
    if (filters or codec) and _ffmpeg_ok():
        from torchaudio.io import AudioEffector, CodecConfig
        effector = AudioEffector
        if isolate_ffmpeg:
            effector = partial(
                ffmpeg_isolation.IsolatedEffector, timeout=isolate_timeout,
                dump_dir=isolate_dump_dir)
        fmt, enc, rates, _ = _CODECS[codec] if codec else (None, None, None,
                                                           None)
        if narrow:
            filters.append('aresample=8000')
        cfg = CodecConfig(bit_rate=random.randint(*rates)) if rates else None
        st['effector_calls'] += 1
        try:
            ae = effector(effect=','.join(filters) or None, format=fmt,
                          encoder=enc, codec_config=cfg)
            y = ae.apply(torch.from_numpy(x).unsqueeze(1),
                         sample_rate)[:, 0].numpy()
            x = y[:n] if y.shape[0] >= n else np.pad(y, (0, n - y.shape[0]))
            narrow = False  # done by aresample
            applied = codec is not None
        except Exception as e:  # noqa
            _effector_failed(codec, e)
    elif codec:  # numpy fallback: band-limit + mu-law
        narrow = True
        if codec == 'mulaw8k':
            x = _mulaw(x)
    if info is not None:
        info.update(codec=codec, codec_applied=applied)
    if narrow:
        x = resample_poly(resample_poly(x, 1, 2), 2, 1)[:n]
        if x.shape[0] < n:
            x = np.pad(x, (0, n - x.shape[0]))
    if clip_prob > random.random():
        c = random.uniform(*clip_level) * float(np.max(np.abs(x)) + 1e-6)
        x = np.clip(x, -c, c)
    if loss_prob > random.random():
        hop = int(0.02 * sample_rate)
        rate = random.uniform(*loss_rate)
        for k in range(n // hop):
            if random.random() < rate:
                x[k * hop:(k + 1) * hop] = 0.0
    x = x / (np.max(np.abs(x)) + 1e-4)
    return np.ascontiguousarray(x, dtype=np.float32)


def add_channel_asym(data: Iterable[Sample],
                     arms: Iterable[int],
                     prob: float = 1.0,
                     kpos: Sequence[int] = (1,),
                     distinct: bool = True,
                     resample_rate: int = 16000,
                     codecs: Sequence[str] = CHAIN_CODECS,
                     **channel_args: Any) -> Iterator[Sample]:
    """Channel chain with a forced codec on some rows of the listed arms.

    A row gets the chain, with probability ``prob``, when its
    ``sample['arm']`` is listed, its position in the speaker's K-group
    (``sample['kpos']``) is in ``kpos`` and it is not an anchor of a pair
    batch (``pair_role`` 1). Rows flagged ``zr_bypass`` (silent crops) are
    skipped before any draw.

    distinct: the codecs of one speaker group (consecutive rows of the same
        ``sample['spk']``) are all different.
    channel_args: keyword arguments of :func:`apply_channel`.

    Sets ``sample['chain']`` to the codec that was applied; if the codec
    call failed, ``sample['chain_fail']`` is set to the drawn codec instead
    and ``chain`` stays unset.

    Draws per chained row (``random``): the ``prob`` decision, the codec,
    then the draws of :func:`apply_channel` (whose own codec decision and
    choice are still consumed).
    """
    arm_set = set(arms)
    codec_list = list(codecs)
    used, last = set(), None
    for sample in data:
        spk = sample.get('spk')
        if spk != last:
            used, last = set(), spk
        if sample.get('zr_bypass'):
            yield sample
            continue
        if sample.get('arm') in arm_set and sample.get('kpos', 0) in kpos \
                and sample.get('pair_role', 0) != 1 and prob > random.random():
            avail = [c for c in codec_list if c not in used] if distinct \
                else codec_list
            codec = random.choice(avail or codec_list)
            used.add(codec)
            audio = sample['wav'].numpy()[0]
            info: Dict[str, Any] = {}
            sample['wav'] = torch.from_numpy(apply_channel(
                audio, resample_rate, codec_prob=1.0, codecs=(codec,),
                info=info, **channel_args)).unsqueeze(0)
            if info.get('codec_applied'):
                sample['chain'] = codec
            else:
                sample['chain_fail'] = codec
        yield sample


def channel_args(cfg: ChannelCfg, arm_index: Mapping[str, int],
                 sample_rate: int) -> Dict[str, Any]:
    """Keyword arguments of :func:`add_channel_asym` for a channel config.

    ``arm_index`` maps an arm name to its integer in ``sample['arm']``.
    """
    return dict(arms=[arm_index[name] for name in cfg.arms],
                prob=cfg.prob,
                kpos=tuple(cfg.kpos),
                distinct=cfg.distinct,
                resample_rate=sample_rate,
                codecs=tuple(cfg.codecs),
                eq_prob=cfg.eq_prob,
                eq_gain_db=cfg.eq_gain_db,
                bandlimit_prob=cfg.bandlimit_prob,
                clip_prob=cfg.clip_prob,
                clip_level=tuple(cfg.clip_level),
                loss_prob=cfg.loss_prob,
                loss_rate=tuple(cfg.loss_rate),
                isolate_ffmpeg=cfg.isolate_ffmpeg)
