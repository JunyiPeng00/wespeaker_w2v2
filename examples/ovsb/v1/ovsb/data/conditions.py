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
"""Per-utterance recording conditions, parsed from the utterance key.

The corpus converters encode the conditions in the keys:

3D-Speaker
    ``3ds_<spk>/Device<XX>_Distance<YY>_Dialect<ZZ>/<seg>_Device..._Dialect<ZZ>``.
    ``<seg>`` is the official segment id; a segment is one recording session
    captured in parallel by several devices, so the files of a segment are
    content duplicates. Distance class: 0 close (close-talk or <= 0.3 m),
    1 mid, 2 far (>= 1 m). Dialect 0 is Standard Mandarin.
CN-Celeb
    ``cn2_<id>/<genre>-<SS>-<UUU>`` -> genre.
KeSpeech
    ``ks_<spk>/phase<N>/<utt>`` -> phase; the sub-dialect of an utterance
    comes from a ``<key> <sub-dialect>`` table, ``Mandarin`` = Standard = 0.

Integer arrays use -1 for "not applicable" (keys of other corpora).
"""

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Union

import numpy as np

# 3D-Speaker distance code -> metres; None = close-talk microphone
DISTANCE_M = {0: None, 1: 3.5, 2: 0.2, 3: 0.1, 4: 1.0, 5: 1.5, 6: 3.0, 7: 2.5,
              8: 0.3, 9: 0.9, 10: 0.8, 11: 1.2, 12: 4.0, 13: 2.0}
CLOSE_MAX_M = 0.3
FAR_MIN_M = 1.0
STANDARD_SUBDIALECT = b'Mandarin'

_COND = re.compile(rb'Device(\d+)_Distance(\d+)_Dialect(\d+)$')

Key = Union[bytes, str]


class Cond3ds(NamedTuple):
    """Conditions of one 3D-Speaker utterance."""
    segment: bytes  # '<spk>/<segment id>'
    device: int
    distance_class: int
    dialect: int


@dataclass(frozen=True)
class ConditionArrays:
    """Key-parsed conditions of a list of utterances (-1 = not applicable).

    Attributes:
        seg: (N,) int64 index into ``seg_names`` (3D-Speaker segment).
        dev: (N,) int64 3D-Speaker device code.
        dcls: (N,) int64 3D-Speaker distance class (0 close, 1 mid, 2 far).
        dial: (N,) int64 3D-Speaker dialect code (0 = Standard).
        genre: (N,) int64 index into ``genre_names`` (CN-Celeb genre).
        seg_names: segment names in order of first appearance.
        genre_names: genre names in order of first appearance.
    """
    seg: np.ndarray
    dev: np.ndarray
    dcls: np.ndarray
    dial: np.ndarray
    genre: np.ndarray
    seg_names: List[bytes]
    genre_names: List[bytes]


@dataclass(frozen=True)
class KsArrays:
    """KeSpeech phase and sub-dialect of a list of utterances.

    Attributes:
        phase: (N,) int64 recording phase (1 or 2), -1 = not a KeSpeech key.
        dial: (N,) int64 index into ``dial_names`` (0 = Standard), -1 = not
            a KeSpeech key or no table entry.
        dial_names: sub-dialect names; Standard first, then first appearance.
        n_utts: number of KeSpeech keys.
        n_without_subdialect: KeSpeech keys without a table entry.
    """
    phase: np.ndarray
    dial: np.ndarray
    dial_names: List[bytes]
    n_utts: int
    n_without_subdialect: int


def _as_bytes(key: Key) -> bytes:
    return key.encode() if isinstance(key, str) else key


def distance_class(code: int) -> int:
    """3D-Speaker distance code -> 0 close / 1 mid / 2 far."""
    m = DISTANCE_M.get(int(code))
    if m is None or m <= CLOSE_MAX_M:
        return 0
    return 2 if m >= FAR_MIN_M else 1


def parse_3ds(key: Key) -> Optional[Cond3ds]:
    """Conditions of a 3D-Speaker key, None for any other key."""
    key = _as_bytes(key)
    if not key.startswith(b'3ds_'):
        return None
    p = key.split(b'/')
    if len(p) != 3:
        return None
    m = _COND.search(p[1])
    if m is None:
        return None
    suffix = b'_' + p[1]
    utt = p[2][:-4] if p[2].endswith(b'.wav') else p[2]
    seg = utt[:-len(suffix)] if utt.endswith(suffix) else utt
    return Cond3ds(p[0] + b'/' + seg, int(m.group(1)),
                   distance_class(int(m.group(2))), int(m.group(3)))


def parse_genre(key: Key) -> Optional[bytes]:
    """Genre of a CN-Celeb key ``cn2_<id>/<genre>-<SS>-<UUU>``, else None."""
    key = _as_bytes(key)
    if not key.startswith(b'cn2_'):
        return None
    p = key.rsplit(b'/', 1)[-1].split(b'-')
    return p[0] if len(p) >= 3 else None


def parse_ks_phase(key: Key) -> Optional[int]:
    """Phase of a KeSpeech key ``ks_<spk>/phase<N>/<utt>``, else None."""
    key = _as_bytes(key)
    if not key.startswith(b'ks_'):
        return None
    p = key.split(b'/')
    if len(p) == 3 and p[1].startswith(b'phase') and p[1][5:].isdigit():
        return int(p[1][5:])
    return None


def _starts_with(karr: np.ndarray, prefix: bytes) -> np.ndarray:
    if not len(karr):
        return np.zeros(0, bool)
    return np.char.startswith(karr, prefix)


def condition_arrays(keys: Sequence[bytes]) -> ConditionArrays:
    """3D-Speaker and CN-Celeb conditions of ``keys`` (see ConditionArrays).

    Segment and genre indices are assigned in order of first appearance in
    ``keys``.
    """
    n = len(keys)
    seg = np.full(n, -1, np.int64)
    dev = np.full(n, -1, np.int64)
    dcls = np.full(n, -1, np.int64)
    dial = np.full(n, -1, np.int64)
    genre = np.full(n, -1, np.int64)
    karr = np.asarray(keys, dtype='S')
    seg_ids, seg_names = {}, []
    for i in np.flatnonzero(_starts_with(karr, b'3ds_')):
        c = parse_3ds(karr[i])
        if c is None:
            continue
        s = seg_ids.get(c.segment)
        if s is None:
            s = seg_ids[c.segment] = len(seg_names)
            seg_names.append(c.segment)
        seg[i], dev[i], dcls[i], dial[i] = \
            s, c.device, c.distance_class, c.dialect
    genre_ids, genre_names = {}, []
    for i in np.flatnonzero(_starts_with(karr, b'cn2_')):
        g = parse_genre(karr[i])
        if g is None:
            continue
        j = genre_ids.get(g)
        if j is None:
            j = genre_ids[g] = len(genre_names)
            genre_names.append(g)
        genre[i] = j
    return ConditionArrays(seg, dev, dcls, dial, genre, seg_names,
                           genre_names)


def read_subdialect(paths: Iterable[str]) -> Dict[bytes, bytes]:
    """Read ``<key> <sub-dialect>`` tables (later files override earlier)."""
    table = {}
    for path in paths:
        with open(path, 'rb') as f:
            for line in f:
                t = line.split()
                if len(t) >= 2:
                    table[t[0]] = t[1]
    return table


def ks_arrays(keys: Sequence[bytes],
              subdialect: Dict[bytes, bytes]) -> KsArrays:
    """KeSpeech phase and sub-dialect of ``keys`` (see KsArrays).

    A key is looked up in ``subdialect`` as written and, when it ends in
    ``.wav``, without that suffix.
    """
    n = len(keys)
    phase = np.full(n, -1, np.int64)
    dial = np.full(n, -1, np.int64)
    karr = np.asarray(keys, dtype='S')
    is_ks = _starts_with(karr, b'ks_')
    names = [STANDARD_SUBDIALECT]
    ids = {STANDARD_SUBDIALECT: 0}
    n_without = 0
    for i in np.flatnonzero(is_ks):
        key = bytes(karr[i])
        ph = parse_ks_phase(key)
        if ph is not None:
            phase[i] = ph
        d = subdialect.get(key)
        if d is None and key.endswith(b'.wav'):
            d = subdialect.get(key[:-4])
        if d is None:
            n_without += 1
            continue
        j = ids.get(d)
        if j is None:
            j = ids[d] = len(names)
            names.append(d)
        dial[i] = j
    return KsArrays(phase, dial, names, int(is_ks.sum()), n_without)
