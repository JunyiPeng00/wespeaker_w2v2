"""Exact digital-zero runs in a crop: measure, dither, flag silent rows.

:func:`zero_run_guard` runs right after the crop and before every
augmentation / channel stage; it is 1:1 and order-preserving. Long runs of
exact zeros are replaced by +-1 LSB white noise so that no codec or peak
normaliser downstream sees them. The dither has its own random stream,
``numpy.random.default_rng((seed, epoch, worker, 31[, rank]))`` (the rank
is appended for ranks > 0 only), so every other draw of the pipeline is the
same with or without it.
"""

from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from ovsb.config.schema import ZeroRunCfg

Sample = Dict[str, Any]
# no dither when the crop's peak is below this many LSB (int16)
DITHER_MIN_PEAK_LSB = 8.0
_STREAM_TAG = 31


def zero_run_stats(x: np.ndarray, min_run: int, lsb_level: float = 4
                   ) -> Tuple[float, List[Tuple[int, int]]]:
    """Low-level statistics of a 1-D float waveform in int16 / 32768 units.

    Returns (fraction of samples with ``|x| <= lsb_level`` LSB, list of
    ``(start, end)`` runs of exact zeros of at least ``min_run`` samples).
    """
    if x.shape[0] == 0:
        return 0.0, []
    lsb = float(np.mean(np.abs(x) * 32768.0 <= lsb_level + 1e-3))
    z = np.concatenate(([0], (x == 0).astype(np.int8), [0]))
    d = np.diff(z)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (ends - starts) >= min_run
    return lsb, list(zip(starts[keep].tolist(), ends[keep].tolist()))


def _dither_rng(seed: int, epoch: int, worker_id: int, rank: int,
                stream: int) -> np.random.Generator:
    if stream:
        return np.random.default_rng(
            (int(seed), int(epoch), worker_id, _STREAM_TAG, int(rank),
             int(stream)))
    if int(rank) == 0:
        return np.random.default_rng(
            (int(seed), int(epoch), worker_id, _STREAM_TAG))
    return np.random.default_rng(
        (int(seed), int(epoch), worker_id, _STREAM_TAG, int(rank)))


def zero_run_guard(data: Iterable[Sample],
                   arms: Optional[Iterable[int]] = None,
                   dither: bool = True,
                   min_run_ms: float = 50.0,
                   lsb_level: float = 4,
                   dither_lsb: float = 1.0,
                   sample_rate: int = 16000,
                   stats: bool = True,
                   seed: int = 0,
                   epoch: int = 0,
                   rank: Optional[int] = None,
                   stream: int = 0) -> Iterator[Sample]:
    """Dither exact-zero runs of every crop and record its statistics.

    dither: every exact-zero run of at least ``min_run_ms`` in a row of the
        listed arm indices (None = every row) is replaced by white noise
        uniform in +-``dither_lsb`` LSB. A crop that is silence (peak below
        ``DITHER_MIN_PEAK_LSB``) or one zero run from end to end is left
        untouched, consumes no random number and gets
        ``sample['zr_bypass'] = 1``: the channel stage skips such rows.
    stats: record ``zr_lsb`` (fraction of samples at or below ``lsb_level``
        LSB), ``zr_run`` (1 if a run exists), ``zr_dith`` (1 if dithered)
        and ``zr_silent`` (1 if bypassed), all measured before dithering.
    rank: None = the distributed rank of this process (0 without DDP).
    stream: 0 except after a mid-epoch resume (fresh dither sequence).
    """
    min_run = max(1, int(round(min_run_ms / 1000.0 * sample_rate)))
    arm_set = None if arms is None else set(arms)
    info = torch.utils.data.get_worker_info()
    worker_id = 0 if info is None else info.id
    if rank is None:
        rank = dist.get_rank() if dist.is_available() and \
            dist.is_initialized() else 0
    rng = _dither_rng(seed, epoch, worker_id, rank, stream)
    amp = float(dither_lsb) / 32768.0
    for sample in data:
        if dither or stats:
            x = sample['wav'][0].numpy()
            lsb, runs = zero_run_stats(x, min_run, lsb_level)
            hit = False
            silent = False
            listed = arm_set is None or sample.get('arm') in arm_set
            if dither and listed:
                peak_lsb = float(np.abs(x).max()) * 32768.0
                silent = peak_lsb < DITHER_MIN_PEAK_LSB or any(
                    e - s >= x.shape[0] for s, e in runs)
                if silent:
                    sample['zr_bypass'] = 1
            if dither and runs and not silent and listed:
                y = x.copy()
                for s, e in runs:
                    y[s:e] = rng.uniform(-amp, amp, e - s)
                sample['wav'] = torch.from_numpy(
                    y.astype(np.float32)).unsqueeze(0)
                hit = True
            if stats:
                sample['zr_lsb'] = lsb
                sample['zr_run'] = int(bool(runs))
                sample['zr_dith'] = int(hit)
                sample['zr_silent'] = int(silent)
        yield sample


def zero_run_args(cfg: ZeroRunCfg, arm_index: Mapping[str, int],
                  sample_rate: int, seed: int) -> Dict[str, Any]:
    """Keyword arguments of :func:`zero_run_guard` for a dither config.

    ``arm_index`` maps an arm name to its integer in ``sample['arm']``.
    ``epoch``, ``rank`` and ``stream`` are left to the caller.
    """
    arms = None if cfg.arms is None else [arm_index[n] for n in cfg.arms]
    return dict(arms=arms,
                dither=True,
                min_run_ms=float(cfg.min_run_ms),
                lsb_level=float(cfg.lsb_level),
                dither_lsb=float(cfg.dither_lsb),
                sample_rate=sample_rate,
                stats=True,
                seed=seed)
