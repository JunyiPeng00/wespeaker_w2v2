"""Additive noise and reverberation: one type per utterance.

Two 1:1, order-preserving generator stages after the zero-run guard:

1. :func:`draw_aug_type` draws ``sample['aug_type']`` (one
   ``random.random()`` per row; anchors of pair batches stay clean);
2. :func:`add_reverb_noise` applies it: ``clean`` | ``noise`` (dry MUSAN
   noise) | ``reverb`` | ``reverb_noise`` (RIR, then noise; with
   ``reverberant_noise`` the noise passes through another RIR of the same
   room first and the SNR is set after that reverberation).

Reused unchanged from ``wespeaker.dataset.processor``: ``apply_reverb``,
``_decode_rir``, ``_rir_same_room``, ``get_random_chunk``,
``_snr_range_for_key``. Only :func:`apply_noise` is a copy, because the
core function has no ``noise_rir`` argument; with ``noise_rir=None`` the
two are equal (pinned by a test).
"""

import io
import random
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import torch
from scipy import signal
from scipy.io import wavfile

from ovsb.config.schema import AUG_TYPES, AugmentCfg
from wespeaker.dataset.processor import (_decode_rir, _rir_same_room,
                                         _snr_range_for_key, apply_reverb,
                                         get_random_chunk)

Sample = Dict[str, Any]


def _aug_cum(aug_probs: Mapping[str, float]) -> List[float]:
    """Cumulative probabilities over ``AUG_TYPES``; the rest goes to clean."""
    unknown = set(aug_probs) - set(AUG_TYPES)
    assert not unknown, 'unknown aug types {}'.format(unknown)
    weights = [float(aug_probs.get(t, 0.0)) for t in AUG_TYPES]
    assert min(weights) >= 0 and sum(weights) <= 1.0 + 1e-6, aug_probs
    weights[0] += max(0.0, 1.0 - sum(weights))  # remainder -> clean
    cum, acc = [], 0.0
    for w in weights:
        acc += w
        cum.append(acc)
    cum[-1] = 1.0
    return cum


def draw_aug_type(data: Iterable[Sample],
                  aug_probs: Mapping[str, float],
                  arm_probs: Optional[Mapping[int, Mapping[str, float]]] = None
                  ) -> Iterator[Sample]:
    """Draw one augmentation type per row into ``sample['aug_type']``.

    aug_probs: {type: probability}; missing types are 0, the probabilities
        sum to at most 1 and the remainder is ``clean``.
    arm_probs: {arm index: aug_probs} replacing ``aug_probs`` for rows whose
        ``sample['arm']`` is listed. One uniform draw per row either way.

    Anchors of pair batches (``pair_role`` 1) are always ``clean`` (their
    draw is still consumed).
    """
    cum = _aug_cum(aug_probs)
    arm_cum = {a: _aug_cum(p) for a, p in (arm_probs or {}).items()}
    for sample in data:
        u = random.random()
        row_cum = arm_cum.get(sample.get('arm'), cum) if arm_cum else cum
        for aug_type, c in zip(AUG_TYPES, row_cum):
            if u < c:
                sample['aug_type'] = aug_type
                break
        if sample.get('pair_role', 0) == 1:
            sample['aug_type'] = 'clean'
        yield sample


def apply_noise(audio: np.ndarray, noise_source: Any,
                resample_rate: int = 16000,
                snr_ranges: Optional[Mapping[str, List[float]]] = None,
                noise_rir: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, str, float]:
    """Add a random noise item at a random SNR of its category's range.

    Copy of ``wespeaker.dataset.processor.apply_noise`` plus ``noise_rir``:
    a unit-energy RIR the noise chunk is convolved with before the SNR is
    set, so the SNR is that of the reverberant noise against ``audio``.
    Returns (noisy audio, noise key, snr in dB).
    """
    snr_ranges = snr_ranges or {}
    audio_len = audio.shape[0]
    audio_db = 10 * np.log10(np.mean(audio**2) + 1e-4)
    key, noise_data = noise_source.random_one()
    snr_range = _snr_range_for_key(key, snr_ranges)
    noise_sr, noise_audio = wavfile.read(io.BytesIO(noise_data))
    # chunk before the float conversion (noise files are minutes long)
    if noise_sr != resample_rate:
        noise_audio = get_random_chunk(
            noise_audio, int(audio_len / resample_rate * noise_sr))
        noise_audio = noise_audio.astype(np.float32) / (1 << 15)
        noise_audio = signal.resample(noise_audio, audio_len)
    else:
        noise_audio = get_random_chunk(noise_audio, audio_len)
        noise_audio = noise_audio.astype(np.float32) / (1 << 15)
    if noise_rir is not None:
        noise_audio = signal.convolve(noise_audio, noise_rir,
                                      mode='full')[:audio_len]
        noise_audio = noise_audio.astype(np.float32)
    noise_snr = random.uniform(snr_range[0], snr_range[1])
    noise_db = 10 * np.log10(np.mean(noise_audio**2) + 1e-4)
    noise_audio = np.sqrt(10**(
        (audio_db - noise_db - noise_snr) / 10)) * noise_audio
    return audio + noise_audio, key, noise_snr


def add_reverb_noise(data: Iterable[Sample],
                     reverb_source: Any,
                     noise_source: Any,
                     resample_rate: int = 16000,
                     reverberant_noise: bool = False) -> Iterator[Sample]:
    """Apply ``sample['aug_type']`` (see :func:`draw_aug_type`).

    reverb_source / noise_source: ``LmdbData``-like objects (``random_one``,
        ``keys``, ``db``); the reverb source may be a ``MultiLmdbData``.
    reverberant_noise: the noise of a ``reverb_noise`` row is convolved with
        another RIR of the room the speech went through before the SNR is
        set. The plain ``noise`` type stays dry.

    Random draws per non-clean row, in order: speech RIR (reverb types),
    noise RIR (reverberant noise), noise item, noise chunk start, SNR.
    Augmented rows are peak-normalised; clean rows pass through untouched.
    """
    for sample in data:
        assert 'wav' in sample
        aug_type = sample.get('aug_type', 'clean')
        if aug_type == 'clean':
            yield sample
            continue
        audio = sample['wav'].numpy()[0]
        rir_key = None
        if aug_type in ('reverb', 'reverb_noise'):
            audio, rir_key = apply_reverb(audio, reverb_source, resample_rate)
        if aug_type in ('noise', 'reverb_noise'):
            noise_rir = None
            if reverberant_noise and aug_type != 'noise':
                _, rir_data = _rir_same_room(reverb_source, rir_key)
                noise_rir = _decode_rir(rir_data, resample_rate)
            audio, _, _ = apply_noise(audio, noise_source, resample_rate,
                                      noise_rir=noise_rir)
        # normalize into [-1, 1]
        audio = audio / (np.max(np.abs(audio)) + 1e-4)
        sample['wav'] = torch.from_numpy(
            np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        yield sample


def draw_args(cfg: AugmentCfg, arm_index: Mapping[str, int]
              ) -> Dict[str, Any]:
    """Keyword arguments of :func:`draw_aug_type` for an augment config.

    ``arm_index`` maps an arm name to its integer in ``sample['arm']``.
    """
    arm_probs = {arm_index[name]: dict(probs)
                 for name, probs in cfg.arm_probs.items()}
    return dict(aug_probs=dict(cfg.probs), arm_probs=arm_probs or None)
