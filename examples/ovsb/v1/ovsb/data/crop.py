"""Variable-length crops inside a batch, and the pair batches.

:func:`random_chunk_var` is a 1:N generator stage (N = 1 for plain batches,
N = 2 for pair batches) placed after the length filter and the resampler.
It owns the batch boundaries: every ``batch_size`` yielded rows it draws a
new batch length and decides whether the next batch is a pair batch. The
data loader must therefore batch exactly ``batch_size`` consecutive rows.

Random streams (kept draw for draw):

* per batch, from ``random.Random('osvb-varcrop-<seed>-<epoch>-<worker>')``,
  identical on every rank: the batch cap, then (if ``pair_prob > 0``) the
  pair decision;
* per row, from the global ``random`` module: view length (pair batches),
  row cap of a capped arm, full / uniform length, onset decision, onset
  frame and pre-roll, random start, energy-gate redraws.
"""

import random
from typing import (Any, Dict, Iterable, Iterator, List, Mapping, NamedTuple,
                    Optional, Sequence, Tuple)

import numpy as np
import torch

from ovsb.config.schema import CropCfg

Sample = Dict[str, Any]
# rows of a pair batch: sample['pair_role']
ROLE_PLAIN, ROLE_ANCHOR, ROLE_VIEW = 0, 1, 2


class EnergyGate(NamedTuple):
    """Redraw rule of short plain crops (lengths in samples)."""
    gate_below: int
    min_active_frac: float
    max_redraws: int


def _frame_active(wav: np.ndarray, sample_rate: int
                  ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """20-ms frame energies (dB) of a 1-D waveform and the active mask.

    A frame is active when its energy exceeds the utterance maximum minus
    30 dB. Returns ``(None, None, hop)`` for fewer than 8 frames.
    """
    hop = int(0.02 * sample_rate)
    n = wav.shape[0] // hop
    if n < 8:
        return None, None, hop
    e = wav[:n * hop].reshape(n, hop)
    e = 10 * np.log10(np.mean(e.astype(np.float64)**2, axis=1) + 1e-10)
    return e, e > e.max() - 30, hop


def _active_frac(active: np.ndarray, hop: int, start: int,
                 length: int) -> float:
    """Fraction of active frames inside the crop ``[start, start + length)``.

    1.0 when the crop covers no full frame.
    """
    f0 = start // hop
    f1 = min(active.shape[0], (start + length) // hop)
    if f1 <= f0:
        return 1.0
    return float(np.mean(active[f0:f1]))


def _onset_start(wav: np.ndarray, crop_len: int, sample_rate: int,
                 pre_roll: float = 0.1) -> Optional[int]:
    """Start of a crop that begins just before a random energy onset.

    An onset is a frame that turns active after at least 100 ms of inactive
    frames. Returns None if the utterance has no usable onset (then no
    random number is consumed).
    """
    e, active, hop = _frame_active(wav, sample_rate)
    if e is None:
        return None
    n = e.shape[0]
    quiet = np.convolve(~active, np.ones(5), mode='full')[:n] >= 5
    onsets = np.nonzero(active[1:] & quiet[:-1])[0] + 1
    onsets = onsets[onsets * hop + crop_len // 2 <= wav.shape[0]]
    if onsets.shape[0] == 0:
        return None
    start = int(random.choice(onsets)) * hop - \
        int(random.uniform(0, pre_roll) * sample_rate)
    return max(0, min(start, wav.shape[0] - crop_len))


def _worker_id() -> int:
    info = torch.utils.data.get_worker_info()
    return 0 if info is None else info.id


def random_chunk_var(data: Iterable[Sample],
                     max_lens: Sequence[int],
                     max_weights: Sequence[float],
                     batch_size: int,
                     min_len: int,
                     full_prob: float = 0.5,
                     onset_prob: float = 0.25,
                     onset_max_len: Optional[int] = None,
                     pair_prob: float = 0.0,
                     pair_anchor_len: Optional[int] = None,
                     pair_view_lens: Optional[Sequence[int]] = None,
                     sample_rate: int = 16000,
                     seed: int = 0,
                     epoch: int = 0,
                     arm_caps: Optional[Mapping[
                         int, Tuple[Sequence[int], Sequence[float]]]] = None,
                     pair_view_onset_prob: Optional[float] = None,
                     energy_gate: Optional[EnergyGate] = None,
                     stream: int = 0) -> Iterator[Sample]:
    """Crop every utterance; one length cap per batch. Lengths in samples.

    Plain batch: the cap L is drawn from ``max_lens`` / ``max_weights``; a
    row is cropped to its row cap with probability ``full_prob``, else to
    U(``min_len``, row cap). Utterances shorter than the drawn length are
    kept whole. Crops of at most ``onset_max_len`` start at an energy onset
    with probability ``onset_prob``.

    Pair batch (probability ``pair_prob`` per batch): every input utterance
    yields two consecutive rows, an anchor of ``pair_anchor_len`` (role 1)
    and a view of U(``pair_view_lens``) (role 2), so the batch consumes
    ``batch_size / 2`` input rows. Views start at an onset with probability
    ``pair_view_onset_prob`` whatever their length (None = the plain rule).

    ``arm_caps``: {arm index: (lens, weights)} - the row cap of plain rows
    of that arm follows this distribution, coupled to the batch cap by
    quantiles (never above the batch cap).

    ``energy_gate``: a plain crop shorter than ``gate_below`` that is not the
    whole utterance must cover at least ``min_active_frac`` active frames;
    its start is redrawn up to ``max_redraws`` times, then kept.

    ``stream``: 0 except after a mid-epoch resume, where it selects a fresh
    per-batch sequence for the rest of the epoch.

    Each output row is a shallow copy of the input with ``wav`` replaced
    (unpadded) and ``pad_to`` (batch length the collate pads to),
    ``pair_role`` and, with a gate, ``eg_redraw`` (redraws, -1 = not
    subject to the gate) and ``eg_fail`` (1 = kept below the threshold).
    """
    max_lens = list(max_lens)
    assert len(max_lens) == len(max_weights) and batch_size > 0
    assert pair_prob == 0.0 or (batch_size % 2 == 0 and pair_anchor_len
                                and pair_view_lens)
    key = 'osvb-varcrop-{}-{}-{}'.format(seed, epoch, _worker_id())
    if stream:
        key += '-{}'.format(stream)
    rng = random.Random(key)
    onset_max_len = onset_max_len or 0
    acaps: Dict[int, Tuple[List[int], np.ndarray]] = {}
    if arm_caps:
        gw = np.asarray(max_weights, np.float64)
        gcdf = np.cumsum(gw / gw.sum())
        for arm, (lens, weights) in arm_caps.items():
            w = np.asarray(weights, np.float64)
            acaps[arm] = (list(lens), np.cumsum(w / w.sum()))

    def row_cap(sample: Sample, cap: int) -> int:
        arm = sample.get('arm')
        if arm not in acaps:
            return cap
        lens, acdf = acaps[arm]
        i = max_lens.index(cap)
        lo = gcdf[i - 1] if i > 0 else 0.0
        u = lo + (gcdf[i] - lo) * random.random()
        j = min(int(np.searchsorted(acdf, u, side='right')), len(lens) - 1)
        return min(lens[j], cap)

    def draw_start(wav: torch.Tensor, length: int, role: int) -> int:
        start = None
        if role == ROLE_VIEW and pair_view_onset_prob is not None:
            if pair_view_onset_prob > random.random():
                start = _onset_start(wav.numpy(), length, sample_rate)
        elif length <= onset_max_len and onset_prob > random.random():
            start = _onset_start(wav.numpy(), length, sample_rate)
        if start is None:
            start = random.randint(0, wav.shape[0] - length)
        return start

    def crop(sample: Sample, length: int, role: int, pad_to: int) -> Sample:
        wav = sample['wav'][0]
        length = min(length, wav.shape[0])
        start = draw_start(wav, length, role)
        n_redraw, failed = -1, 0
        if energy_gate and role == ROLE_PLAIN and \
                length < energy_gate.gate_below and wav.shape[0] > length:
            _, active, hop = _frame_active(wav.numpy(), sample_rate)
            if active is not None:
                n_redraw = 0
                while n_redraw < energy_gate.max_redraws and _active_frac(
                        active, hop, start,
                        length) < energy_gate.min_active_frac:
                    start = draw_start(wav, length, role)
                    n_redraw += 1
                failed = int(_active_frac(active, hop, start, length) <
                             energy_gate.min_active_frac)
        out = dict(sample)
        out['wav'] = wav[start:start + length].clone().unsqueeze(0)
        out['pad_to'] = pad_to
        out['pair_role'] = role
        if energy_gate:
            out['eg_redraw'] = n_redraw
            out['eg_fail'] = failed
        return out

    n = 0
    cap, paired = 0, False
    for sample in data:
        assert 'key' in sample
        if n % batch_size == 0:
            cap = rng.choices(max_lens, weights=max_weights)[0]
            paired = pair_prob > 0 and rng.random() < pair_prob
        if paired:
            view = random.randint(*pair_view_lens)
            yield crop(sample, pair_anchor_len, ROLE_ANCHOR, pair_anchor_len)
            yield crop(sample, view, ROLE_VIEW, pair_anchor_len)
            n += 2
            continue
        rcap = row_cap(sample, cap)
        if full_prob > random.random() or rcap <= min_len:
            length = rcap
        else:
            length = random.randint(min_len, rcap)
        yield crop(sample, length, ROLE_PLAIN, cap)
        n += 1


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Length in samples of a duration in seconds (rounded)."""
    return int(round(float(seconds) * sample_rate))


def crop_args(cfg: CropCfg, arm_index: Mapping[str, int], batch_size: int,
              sample_rate: int, seed: int) -> Dict[str, Any]:
    """Keyword arguments of :func:`random_chunk_var` for a crop config.

    ``arm_index`` maps an arm name to the integer the sampler writes into
    ``sample['arm']`` (the position of the arm in the sampler config).
    ``epoch`` (and ``stream``) are left to the caller.
    """
    def sec(x: float) -> int:
        return seconds_to_samples(x, sample_rate)

    arm_caps = {
        arm_index[name]: ([sec(c[0]) for c in caps],
                          [float(c[1]) for c in caps])
        for name, caps in cfg.arm_caps.items()}
    gate = cfg.energy_gate
    return dict(
        max_lens=[sec(c[0]) for c in cfg.max_secs],
        max_weights=[float(c[1]) for c in cfg.max_secs],
        batch_size=batch_size,
        min_len=sec(cfg.min_sec),
        full_prob=cfg.full_prob,
        onset_prob=cfg.onset_prob,
        onset_max_len=sec(cfg.onset_max_sec),
        pair_prob=cfg.pair_prob,
        pair_anchor_len=sec(cfg.pair_anchor_sec),
        pair_view_lens=[sec(v) for v in cfg.pair_view_secs],
        sample_rate=sample_rate,
        seed=seed,
        arm_caps=arm_caps or None,
        pair_view_onset_prob=cfg.pair_view_onset_prob,
        energy_gate=None if gate is None else EnergyGate(
            gate_below=sec(gate.gate_below_sec),
            min_active_frac=float(gate.min_active_frac),
            max_redraws=int(gate.max_redraws)))
