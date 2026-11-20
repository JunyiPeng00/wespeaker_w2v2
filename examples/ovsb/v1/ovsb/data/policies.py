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
"""Positive-pair policies: the K utterances drawn for one speaker.

An episode of the sampler holds K utterances ("rows") of each of its P
speakers; the K rows of a speaker are the positives of each other. This
module decides WHICH K rows a drawn speaker contributes. It only reads a
:class:`~ovsb.data.speakers.SpeakerTable` and consumes the random generator
it is given; choosing the speakers is the sampler's job.

Rows are "positions": indices into the table's sorted utterance order
(speaker s owns ``spk_ptr[s]:spk_ptr[s + 1]``).

The policy of a speaker is the ``pair_policy`` of its arm:

``xsession`` (also the rule of an arm without a policy)
    K rows from K different sessions (videos, recordings), uniform over the
    sessions and then over the files of a session. A speaker with fewer
    than K sessions takes one row per session and fills up with other rows
    (:func:`draw_cross_session`).

``contrast``
    * 3D-Speaker speakers (rows with a segment id; a segment is one
      utterance recorded in parallel on several devices): K rows from
      different segments whose device AND distance class also differ; the
      requirement is relaxed to "different segment and device", then
      "different segment". With probability ``p_dialect`` a speaker that has
      Standard-Mandarin and dialect segments gets one Standard row and one
      dialect row (:func:`draw_3ds_contrast`).
    * speakers whose keys carry a genre (CN-Celeb): K different genres,
      uniform over the speaker's genres, then over the sessions of the
      genre, then over the files; speakers with a single genre fall back to
      ``xsession`` (:func:`draw_genre_contrast`).

``ks_contrast`` (KeSpeech)
    a speaker with phase-1 Standard and phase-1 sub-dialect rows draws, with
    probability ``p_dialect``, rows alternating between the two; otherwise a
    speaker recorded in both phases draws rows alternating between the
    phases; otherwise ``xsession`` (:func:`draw_ks_contrast`).

After an ``xsession`` draw for a 3D-Speaker speaker the rows are checked for
parallel copies of one segment (:func:`draw_plain`): with
``forbid_same_segment`` the later row is redrawn from the speaker's other
segments, else the group is only labelled ``same_segment``.

Every draw reports a positive type (an index into :data:`PTYPES`) that says
which rule produced the rows. The order and number of random draws of every
function is part of the training data stream and must not change.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ovsb.data.speakers import SpeakerTable

# Positive types. The index of a name is the integer code stored with every
# row and logged per arm, so the order is fixed; 'cn2_hardpair', '3ds_farfar'
# and 'same_session' are never produced here and only keep their codes.
PTYPES = ('xsession', 'xsession_segfix', '3ds_contrast', '3ds_xdev',
          '3ds_xseg', 'dialect', 'cn2_xgenre', 'same_segment',
          'cn2_hardpair', '3ds_farfar', 'ks_xphase', 'ks_dialect',
          'same_session')
PT = {n: i for i, n in enumerate(PTYPES)}

Draw = Tuple[List[int], int]


@dataclass
class DrawCounters:
    """Counts of the fill-up cases of :func:`draw_cross_session`.

    Attributes:
        same_video_fill: speakers with fewer than K sessions (some rows
            share a session).
        repeat_fill: of those, speakers with fewer than K utterances (some
            row is repeated).
    """
    same_video_fill: int = 0
    repeat_fill: int = 0


def speaker_rows(table: SpeakerTable, spk: int) -> np.ndarray:
    """All positions of speaker ``spk``."""
    return np.arange(int(table.spk_ptr[spk]), int(table.spk_ptr[spk + 1]))


def is_3ds(table: SpeakerTable, spk: int) -> bool:
    """True for a 3D-Speaker speaker (its first row has a segment id)."""
    return bool(table.seg_of[int(table.spk_ptr[spk])] >= 0)


# ----------------------------------------------------------------------------
# cross-session draw


def draw_cross_session(table: SpeakerTable, spk: int,
                       rng: np.random.Generator,
                       counters: Optional[DrawCounters] = None) -> List[int]:
    """K positions of ``spk`` from K different sessions when possible.

    With at least K sessions: K sessions without replacement, one uniform
    file of each. With fewer: one file per session, then distinct other
    files of the speaker; a speaker with fewer than K files repeats files.
    """
    lo, hi = table.spk_ptr[spk], table.spk_ptr[spk + 1]
    K = table.K
    vids = table.vid_of[lo:hi]
    uv, first, counts = np.unique(vids, return_index=True,
                                  return_counts=True)
    if len(uv) >= K:
        pick = rng.choice(len(uv), size=K, replace=False)
        return [int(lo + first[v] + rng.integers(counts[v])) for v in pick]
    out = [int(lo + first[v] + rng.integers(counts[v]))
           for v in range(len(uv))]
    rest = [i for i in range(lo, hi) if i not in out]
    need = K - len(out)
    if counters is not None:
        counters.same_video_fill += 1
    if len(rest) >= need:
        out += [int(x) for x in rng.choice(rest, size=need, replace=False)]
    else:
        if counters is not None:
            counters.repeat_fill += 1
        out += rest
        out += [int(x) for x in rng.choice(np.arange(lo, hi),
                                           size=K - len(out))]
    return out


# ----------------------------------------------------------------------------
# single-row picks


def pick(table: SpeakerTable, cand: np.ndarray, rng: np.random.Generator,
         segment_first: bool) -> int:
    """One position of the candidates ``cand`` (non-empty array).

    ``segment_first``: for 3D-Speaker candidates spanning several segments,
    a uniform segment and then a uniform file of it, so every segment is
    equally exposed whatever its number of parallel copies; otherwise
    uniform over the files.
    """
    if segment_first:
        sg = table.seg_of[cand]
        if sg[0] >= 0:
            us, inv = np.unique(sg, return_inverse=True)
            if len(us) > 1:
                cand = cand[inv == rng.integers(len(us))]
    return int(cand[rng.integers(len(cand))])


def pick_session(table: SpeakerTable, cand: np.ndarray,
                 rng: np.random.Generator) -> int:
    """One position of ``cand``: a uniform session, then a uniform file."""
    uv, inv = np.unique(table.vid_of[cand], return_inverse=True)
    if len(uv) > 1:
        cand = cand[inv == rng.integers(len(uv))]
    return int(cand[rng.integers(len(cand))])


def next_contrast(table: SpeakerTable, out: Sequence[int], idx: np.ndarray,
                  rng: np.random.Generator, segment_first: bool,
                  restrict: Optional[np.ndarray] = None) -> Tuple[int, int]:
    """The next row of a 3D-Speaker contrast group.

    Among the speaker's positions ``idx`` the row must differ from every row
    in ``out`` by segment, device and distance class (level 0); when no such
    row exists, by segment and device (level 1); then by segment only
    (level 2); when every row shares a segment with ``out`` any row is taken
    (level 3). ``restrict`` (bool mask over ``idx``) limits levels 0-2, and
    level 3 when it selects any row.

    Returns ``(position, level)``.
    """
    sg, dv, dc = table.seg_of[idx], table.dev_of[idx], table.dcls_of[idx]
    o = np.asarray(out, np.int64)
    m_seg = ~np.isin(sg, table.seg_of[o])
    m_dev = ~np.isin(dv, table.dev_of[o])
    m_dc = ~np.isin(dc, table.dcls_of[o])
    if restrict is not None:
        m_seg = m_seg & restrict
    for lev, m in enumerate((m_seg & m_dev & m_dc, m_seg & m_dev, m_seg)):
        if m.any():
            return pick(table, idx[m], rng, segment_first), lev
    cand = idx[restrict] if restrict is not None and restrict.any() else idx
    return pick(table, cand, rng, segment_first), 3


def alternate(idx: np.ndarray, masks: Sequence[np.ndarray], K: int,
              rng: np.random.Generator) -> List[int]:
    """K positions alternating over the row groups ``idx[mask]``.

    Row k comes from group ``k % len(masks)`` (each group non-empty),
    uniform over the group's rows that were not taken yet (over the whole
    group when all were).
    """
    out = []
    for k in range(K):
        cand = idx[masks[k % len(masks)]]
        rest = cand[~np.isin(cand, out)]
        c = rest if len(rest) else cand
        out.append(int(c[rng.integers(len(c))]))
    return out


# ----------------------------------------------------------------------------
# policies


def draw_ks_contrast(table: SpeakerTable, spk: int, p_dialect: float,
                     rng: np.random.Generator) -> Optional[Draw]:
    """KeSpeech contrast rows of ``spk``, or None when neither rule applies.

    * the speaker has phase-1 Standard and phase-1 sub-dialect rows: with
      probability ``p_dialect`` rows alternating Standard / sub-dialect
      (type ``ks_dialect``);
    * otherwise, the speaker has rows in both phases: rows alternating
      phase 1 / phase 2 (type ``ks_xphase``).

    The probability is only drawn for speakers that qualify for the first
    rule.
    """
    idx = speaker_rows(table, spk)
    ph = table.ks_phase[idx]
    dl = table.ks_dial[idx]
    std1 = (ph == 1) & (dl == 0)
    dia1 = (ph == 1) & (dl > 0)
    if std1.any() and dia1.any() and rng.random() < float(p_dialect):
        return alternate(idx, (std1, dia1), table.K, rng), PT['ks_dialect']
    p1, p2 = ph == 1, ph == 2
    if p1.any() and p2.any():
        return alternate(idx, (p1, p2), table.K, rng), PT['ks_xphase']
    return None


def draw_3ds_contrast(table: SpeakerTable, spk: int, p_dialect: float,
                      segment_first: bool, rng: np.random.Generator) -> Draw:
    """3D-Speaker contrast rows of ``spk`` (see :func:`next_contrast`).

    A speaker with Standard rows and a dialect draws the probability
    ``p_dialect``; on success the first row is a Standard row, the second a
    dialect row and further rows are free (type ``dialect``). Otherwise the
    first row is any row. The type of a plain group is its worst level:
    ``3ds_contrast`` (0), ``3ds_xdev`` (1), ``3ds_xseg`` (2); a group that
    had to reuse a segment is ``same_segment`` in both cases.
    """
    idx = speaker_rows(table, spk)
    K = table.K
    has_both = table.spk_has_std[spk] and table.spk_dial[spk] > 0
    if has_both and rng.random() < float(p_dialect):
        dl = table.dial_of[idx]
        out = [pick(table, idx[dl == 0], rng, segment_first)]
        lev = 0
        while len(out) < K:
            restrict = dl > 0 if len(out) == 1 else None
            u, lv = next_contrast(table, out, idx, rng, segment_first,
                                  restrict)
            out.append(u)
            lev = max(lev, lv)
        return out, PT['dialect'] if lev < 3 else PT['same_segment']
    out = [pick(table, idx, rng, segment_first)]
    lev = 0
    while len(out) < K:
        u, lv = next_contrast(table, out, idx, rng, segment_first)
        out.append(u)
        lev = max(lev, lv)
    return out, (PT['3ds_contrast'], PT['3ds_xdev'], PT['3ds_xseg'],
                 PT['same_segment'])[lev]


def draw_genre_contrast(table: SpeakerTable, spk: int,
                        rng: np.random.Generator) -> Optional[Draw]:
    """Rows of ``spk`` from different genres, or None for < 2 genres.

    min(K, genres) genres without replacement, one row of each (uniform
    session of the genre, then file). When K exceeds the number of genres
    the remaining rows come from sessions not used yet (any session when
    all are used). Type ``cn2_xgenre``.
    """
    idx = speaker_rows(table, spk)
    K = table.K
    g = table.genre_of[idx]
    ug = np.unique(g[g >= 0])
    if len(ug) < 2:
        return None
    chosen = rng.choice(len(ug), size=min(K, len(ug)), replace=False)
    out = [pick_session(table, idx[g == ug[j]], rng) for j in chosen]
    while len(out) < K:
        rest = idx[~np.isin(table.vid_of[idx], table.vid_of[out])]
        out.append(pick_session(table, rest if len(rest) else idx, rng))
    return out, PT['cn2_xgenre']


def draw_plain(table: SpeakerTable, spk: int, rng: np.random.Generator,
               counters: Optional[DrawCounters] = None) -> Draw:
    """Cross-session rows of ``spk`` plus the same-segment check.

    For speakers other than 3D-Speaker this is :func:`draw_cross_session`
    (type ``xsession``). The sessions of a 3D-Speaker speaker are its
    recording conditions, so two cross-session rows can be parallel copies
    of one segment. Such a group is labelled ``same_segment``, or, with
    ``table.forbid_same_segment``, repaired: every later row that repeats a
    segment is redrawn from the rows of the other segments, preferring a
    session that the other rows do not use (type ``xsession_segfix``).
    """
    out = draw_cross_session(table, spk, rng, counters)
    if not is_3ds(table, spk):
        return out, PT['xsession']
    sg = table.seg_of[out]
    if len(np.unique(sg)) == len(sg):
        return out, PT['xsession']
    if not table.forbid_same_segment:
        return out, PT['same_segment']
    idx = speaker_rows(table, spk)
    for k in range(1, table.K):
        if table.seg_of[out[k]] not in table.seg_of[out[:k]]:
            continue
        others = out[:k] + out[k + 1:]
        m = ~np.isin(table.seg_of[idx], table.seg_of[others])
        mv = m & ~np.isin(table.vid_of[idx], table.vid_of[others])
        cand = idx[mv] if mv.any() else idx[m]
        out[k] = pick_session(table, cand, rng)
    return out, PT['xsession_segfix']


def draw_rows(table: SpeakerTable, spk: int, arm: int,
              rng: np.random.Generator,
              counters: Optional[DrawCounters] = None) -> Draw:
    """The K positions of speaker ``spk`` under the policy of arm ``arm``.

    Args:
        table: the speaker table (``table.K`` rows are drawn).
        spk: speaker id.
        arm: arm index whose policy applies (the speaker's arm).
        rng: generator of the sampler cell; its draws are consumed in a
            fixed order.
        counters: optional fill-up counters, updated in place.

    Returns:
        ``(positions, type)`` with ``type`` an index into :data:`PTYPES`.
    """
    pol = table.policy[arm]
    if pol is not None and pol.pair_policy == 'ks_contrast':
        res = draw_ks_contrast(table, spk, pol.p_dialect, rng)
        if res is not None:
            return res
    if pol is not None and pol.pair_policy == 'contrast':
        if is_3ds(table, spk):
            return draw_3ds_contrast(table, spk, pol.p_dialect,
                                     bool(pol.segment_first), rng)
        res = draw_genre_contrast(table, spk, rng)
        if res is not None:
            return res
    return draw_plain(table, spk, rng, counters)
