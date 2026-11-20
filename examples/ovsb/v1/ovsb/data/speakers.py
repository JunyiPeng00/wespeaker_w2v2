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
"""Static speaker and utterance tables of the episodic sampler.

:class:`SpeakerTable` joins the training label file to a tar index and holds
everything the sampler needs that does not depend on a random draw: the
utterance order, speaker and session ids, the arm of every speaker, its
language, the recording conditions of every utterance and per-speaker
summaries of them. Nothing here samples or uses a random generator.

Utterance order. Utterances are sorted by (speaker, session) with a stable
sort, so the order of two utterances of one speaker and session is their
order in the label file. Sampler draws index into this order: a label file
with the same lines in another order gives other draws.

Arms. An arm is a named group of speakers selected by corpus; a speaker
belongs to the first arm (in the given order) that lists its corpus. The arm
order also fixes the integer arm codes, so it must be the configured order.
"""

import logging
from dataclasses import dataclass, field
from typing import (Any, Dict, List, Mapping, Optional, Sequence, Tuple,
                    Union)

import numpy as np

from ovsb.data.conditions import (condition_arrays, ks_arrays,
                                  read_subdialect)
from ovsb.data.index import TarIndex

logger = logging.getLogger(__name__)

PAIR_POLICIES = ('xsession', 'contrast', 'ks_contrast')


# ----------------------------------------------------------------------------
# typed inputs (field names follow the recipe config)


@dataclass(frozen=True)
class ArmSpec:
    """One sampling arm.

    Attributes:
        name: arm name.
        weight: share of the speakers of an episode drawn from this arm
            (weights are normalised over the arms).
        corpus: corpus tags whose speakers belong to the arm (a single tag
            may be given as a string).
    """
    name: str
    weight: float
    corpus: Tuple[str, ...]

    def __post_init__(self) -> None:
        corpus = self.corpus
        corpus = (corpus, ) if isinstance(corpus, str) else tuple(corpus)
        object.__setattr__(self, 'corpus', corpus)
        object.__setattr__(self, 'weight', float(self.weight))


@dataclass(frozen=True)
class DialectMass:
    """Speaker-weight rule for 3D-Speaker dialect speakers.

    The dialect speakers of the arm together get ``boost`` times their
    number as mass, split equally over the dialects present and equally over
    the speakers of a dialect; weights are then capped at ``cap_x_median``
    times the median weight of the arm.
    """
    boost: float = 3.0
    cap_x_median: float = 8.0


@dataclass(frozen=True)
class PolicySpec:
    """Positive-pair policy and speaker-weight rules of one arm.

    Attributes:
        pair_policy: 'xsession', 'contrast' or 'ks_contrast'.
        p_dialect: probability of a Standard + dialect pair for speakers that
            have both ('contrast' on 3D-Speaker, 'ks_contrast').
        segment_first: 3D-Speaker draws pick a segment, then a file.
        dialect_mass: 3D-Speaker dialect speaker weights (None = off).
        w_bridge: weight of speakers with two or more genres (None = off).
        subdialect_file: ``<key> <sub-dialect>`` table ('ks_contrast').
        spk_weight_file: ``<speaker> <multiplier>`` table (None = off).
        spk_weight_share: with ``spk_weight_file``, the share of the arm's
            mass the listed speakers get in every sampler cell.
    """
    pair_policy: str = 'xsession'
    p_dialect: float = 0.5
    segment_first: bool = True
    dialect_mass: Optional[DialectMass] = None
    w_bridge: Optional[float] = None
    subdialect_file: Optional[str] = None
    spk_weight_file: Optional[str] = None
    spk_weight_share: Optional[float] = None

    def __post_init__(self) -> None:
        if self.pair_policy not in PAIR_POLICIES:
            raise ValueError('pair_policy %r is not one of %s' % (
                self.pair_policy, PAIR_POLICIES))
        if self.spk_weight_share is not None:
            if not self.spk_weight_file:
                raise ValueError('spk_weight_share needs spk_weight_file')
            if not 0.0 < float(self.spk_weight_share) < 1.0:
                raise ValueError('spk_weight_share %r is not in (0, 1)' %
                                 self.spk_weight_share)
        if isinstance(self.dialect_mass, Mapping):
            object.__setattr__(self, 'dialect_mass',
                               DialectMass(**self.dialect_mass))


@dataclass(frozen=True)
class CellSpec:
    """A share cell: speakers of one arm that get a fixed share of all draws.

    Attributes:
        name: cell name.
        arm: name of the arm the speakers are taken from.
        share: share of all speaker draws (must be below the arm's
            normalised weight; the arm's other speakers keep the rest).
        spk2lang: language label selecting the speakers (config key
            ``select.spk2lang``).
    """
    name: str
    arm: str
    share: float
    spk2lang: str


@dataclass(frozen=True, eq=False)
class ShareCell:
    """A resolved :class:`CellSpec`: arm index and speaker mask."""
    name: str
    arm: int
    share: float
    mask: np.ndarray  # (n_spk,) bool


# ----------------------------------------------------------------------------
# key and speaker-id rules


def read_two_col(path: Optional[str]) -> Dict[str, str]:
    """Read a whitespace-separated two-column file ({} for no path)."""
    out = {}
    if not path:
        return out
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                out[p[0]] = p[1]
    return out


def read_label(path: str) -> List[Tuple[str, str]]:
    """Read ``<utterance key> <speaker>`` lines in file order."""
    out = []
    with open(path, encoding='utf8') as f:
        for n, line in enumerate(f, 1):
            p = line.split()
            if len(p) != 2:
                raise ValueError('%s:%d: expected 2 columns, got %d' % (
                    path, n, len(p)))
            out.append((p[0], p[1]))
    return out


def corpus_of(spk: str, spk2corpus: Optional[Mapping[str, str]] = None) -> str:
    """Corpus tag of a training speaker id.

    The ``spk2corpus`` entry when there is one, else the id namespace:
    ``vb2_*`` -> 'vb2', ``id*`` or no underscore -> 'vox2', ``<tag>_*`` ->
    '<tag>'.
    """
    if spk2corpus and spk in spk2corpus:
        return spk2corpus[spk]
    if spk.startswith('vb2_'):
        return 'vb2'
    if spk.startswith('id'):
        return 'vox2'
    return spk.split('_', 1)[0] if '_' in spk else 'vox2'


def session_of(key: bytes) -> bytes:
    """Session (video) id of an utterance key.

    * ``<spk>/<session>/<utt>`` -> ``<spk>/<session>``
    * ``<spk>/<recording>_<start>_<end>`` (segment keys) ->
      ``<spk>/<recording>``
    * ``<spk>/<genre>-<SS>-<UUU>[...]`` -> ``<spk>/<genre>-<SS>``
    * any other two-field key -> ``<spk>``
    """
    head, tail = key.rsplit(b'/', 1)
    if b'/' in head:
        return head
    if tail.endswith(b'.wav'):
        tail = tail[:-4]
    q = tail.rsplit(b'_', 2)
    if len(q) == 3 and q[1].isdigit() and q[2].isdigit():
        return head + b'/' + q[0]
    p = tail.split(b'-')
    return head + b'/' + b'-'.join(p[:2]) if len(p) >= 3 else head


def arm_matches(arm: ArmSpec, corpus: str) -> bool:
    """True when a speaker of ``corpus`` satisfies the rule of ``arm``."""
    return corpus in arm.corpus


def assign_arms(spks: Sequence[str], arms: Sequence[ArmSpec],
                spk2lang: Mapping[str, str], vox2_lang: str = 'en',
                spk2corpus: Optional[Mapping[str, str]] = None
                ) -> Tuple[np.ndarray, List[str]]:
    """Arm and language of every speaker.

    Returns the index of the first matching arm per speaker (-1 = none) and
    the language label per speaker: ``vox2_lang`` for VoxCeleb2 speakers,
    else the ``spk2lang`` entry, else 'unk'.
    """
    arm = np.full(len(spks), -1, np.int64)
    langs = []
    for i, s in enumerate(spks):
        c = corpus_of(s, spk2corpus)
        langs.append(vox2_lang if c == 'vox2' else spk2lang.get(s, 'unk'))
        for a, spec in enumerate(arms):
            if arm_matches(spec, c):
                arm[i] = a
                break
    return arm, langs


# ----------------------------------------------------------------------------
# the table


@dataclass(frozen=True, eq=False)
class SpeakerTable:
    """Static tables of the episodic sampler (see the module doc).

    Sizes: N utterances (after dropping label keys that are missing from the
    index or shorter than ``min_len``), S speakers, A arms. "Position" means
    an index into the N sorted utterances.

    Join and order
        index: the tar index the rows point into.
        rows: (N,) int64 index row of every position.
        spk_of: (N,) int64 speaker id of every position (non-decreasing).
        vid_of: (N,) int64 session id of every position (rank of the
            session name among all session names).
        spk_ptr: (S + 1,) int64; speaker s owns positions
            ``spk_ptr[s]:spk_ptr[s + 1]``.
        spk_nvid: (S,) int64 number of sessions of the speaker.
        spk_names: (S,) str speaker labels, sorted.
        class_names: (C,) str every label of the label file, sorted; the
            classifier class of a label is its index here.
        spk_class: (S,) int64 class id of every speaker.
        n_missing, n_short: label keys dropped (not in the index / too short).

    Arms and languages
        arm_names: arm names in configured order.
        arm_weights: (A,) float64 configured arm weights (not normalised).
        spk_arm: (S,) int64 arm of the speaker, -1 = never drawn.
        lang_names: sorted language labels; spk_lang: (S,) int64 index.
        policy: per arm, the :class:`PolicySpec` or None (no policy).
        share_cells: resolved share cells.
        K: utterances per speaker in an episode.
        forbid_same_segment: the K rows of a 3D-Speaker speaker never share
            a segment; such speakers with fewer than K segments have
            ``spk_arm`` -1.

    Conditions (-1 = not applicable; see ``ovsb.data.conditions``)
        seg_of, dev_of, dcls_of, dial_of, genre_of: (N,) int64.
        n_segments: number of distinct 3D-Speaker segments.
        genre_names: genre names indexed by ``genre_of``.
        spk_nseg: (S,) int64 distinct segments of the speaker.
        spk_dial: (S,) int64 most frequent non-Standard dialect, 0 = none.
        spk_has_std: (S,) bool the speaker has Standard rows.
        spk_ngenre: (S,) int64 distinct genres of the speaker.
        ks_phase, ks_dial: (N,) int64 KeSpeech phase / sub-dialect index.
        ks_dial_names: sub-dialect names indexed by ``ks_dial``.
        spk_ks_both: (S,) bool the speaker has rows in phases 1 and 2.
        spk_ks_stddial: (S,) bool the speaker has phase-1 Standard and
            phase-1 sub-dialect rows.

    info: counts for the log (same entries as the sampler's construction
        summary); cells_info: per share cell.
    """
    index: TarIndex
    rows: np.ndarray
    spk_of: np.ndarray
    vid_of: np.ndarray
    spk_ptr: np.ndarray
    spk_nvid: np.ndarray
    spk_names: np.ndarray
    class_names: np.ndarray
    spk_class: np.ndarray
    n_missing: int
    n_short: int
    arm_names: Tuple[str, ...]
    arm_weights: np.ndarray
    spk_arm: np.ndarray
    lang_names: np.ndarray
    spk_lang: np.ndarray
    policy: Tuple[Optional[PolicySpec], ...]
    share_cells: Tuple[ShareCell, ...]
    K: int
    forbid_same_segment: bool
    seg_of: np.ndarray
    dev_of: np.ndarray
    dcls_of: np.ndarray
    dial_of: np.ndarray
    genre_of: np.ndarray
    n_segments: int
    genre_names: Tuple[bytes, ...]
    spk_nseg: np.ndarray
    spk_dial: np.ndarray
    spk_has_std: np.ndarray
    spk_ngenre: np.ndarray
    ks_phase: np.ndarray
    ks_dial: np.ndarray
    ks_dial_names: Tuple[bytes, ...]
    spk_ks_both: np.ndarray
    spk_ks_stddial: np.ndarray
    info: Dict[str, Any] = field(default_factory=dict)
    cells_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_utts(self) -> int:
        """Number of utterances N."""
        return len(self.rows)

    @property
    def n_spk(self) -> int:
        """Number of speakers S."""
        return len(self.spk_names)

    @property
    def n_class(self) -> int:
        """Number of classifier classes (labels of the label file)."""
        return len(self.class_names)

    def arm_speaker_counts(self) -> np.ndarray:
        """(A,) number of speakers per arm."""
        return np.bincount(self.spk_arm[self.spk_arm >= 0],
                           minlength=len(self.arm_names))

    def arm_speakers(self, arm: int) -> np.ndarray:
        """Sorted speaker ids of arm index ``arm``."""
        return np.flatnonzero(self.spk_arm == arm)

    def summary(self) -> str:
        """One log line with the sizes of the table."""
        return ('speaker table: K %d, %d utts / %d spk (%d keys not in the '
                'index, %d shorter than min_len), arms %s spk %s unassigned '
                '%d, speakers with 1 video %d' % (
                    self.K, self.n_utts, self.n_spk, self.n_missing,
                    self.n_short, list(self.arm_names),
                    self.arm_speaker_counts().tolist(),
                    int((self.spk_arm < 0).sum()),
                    int((self.spk_nvid < 2).sum())))

    # ------------------------------------------------------------------
    @classmethod
    def from_files(cls, train_label: str, index: Union[TarIndex, str],
                   spk2corpus: Optional[str], spk2lang: Optional[str], *,
                   arms: Sequence[ArmSpec],
                   cells: Sequence[CellSpec] = (),
                   policies: Optional[Mapping[str, PolicySpec]] = None,
                   vox2_lang: str = 'en', K: int = 2,
                   forbid_same_segment: bool = False, min_len: int = 0,
                   index_root: Optional[str] = None) -> 'SpeakerTable':
        """Build the table from the training lists.

        Args:
            train_label: ``<utterance key> <speaker>`` file; its line order
                defines the utterance order inside a speaker session.
            index: tar index (object or directory) covering the keys.
            spk2corpus: ``<speaker> <corpus tag>`` file (None: corpus from
                the speaker-id namespace, see :func:`corpus_of`).
            spk2lang: ``<speaker> <language>`` file (None: 'unk').
            arms: the arms in configured order.
            cells: share cells.
            policies: arm name -> policy; arms without an entry have none.
            vox2_lang: language label of VoxCeleb2 speakers.
            K: utterances per speaker in an episode.
            forbid_same_segment: see the class doc.
            min_len: utterances with fewer samples (index ``nsamples``) are
                dropped.
            index_root: directory for relative shard paths when ``index``
                is a directory (see :class:`TarIndex`).
        """
        if not isinstance(index, TarIndex):
            index = TarIndex(index, root=index_root)
        return cls.build(read_label(train_label), index,
                         read_two_col(spk2corpus), read_two_col(spk2lang),
                         arms=arms, cells=cells, policies=policies,
                         vox2_lang=vox2_lang, K=K,
                         forbid_same_segment=forbid_same_segment,
                         min_len=min_len)

    @classmethod
    def build(cls, utt_spk: Sequence[Tuple[str, str]], index: TarIndex,
              spk2corpus: Mapping[str, str], spk2lang: Mapping[str, str], *,
              arms: Sequence[ArmSpec], cells: Sequence[CellSpec] = (),
              policies: Optional[Mapping[str, PolicySpec]] = None,
              vox2_lang: str = 'en', K: int = 2,
              forbid_same_segment: bool = False,
              min_len: int = 0) -> 'SpeakerTable':
        """Build the table from in-memory lists (see :meth:`from_files`)."""
        arms = list(arms)
        arm_names = tuple(a.name for a in arms)
        if len(set(arm_names)) != len(arm_names):
            raise ValueError('duplicate arm names: %s' % (arm_names, ))
        arm_weights = np.array([a.weight for a in arms], np.float64)
        if not ((arm_weights >= 0).all() and arm_weights.sum() > 0):
            raise ValueError('arm weights must be >= 0 with a positive sum')
        policies = dict(policies or {})
        unknown = set(policies) - set(arm_names)
        if unknown:
            raise ValueError('policies for unknown arms: %s' % sorted(unknown))
        policy = tuple(policies.get(n) for n in arm_names)

        join = _join_label_to_index(utt_spk, index, min_len)
        rows, spk_of, vid_of = join['rows'], join['spk_of'], join['vid_of']
        spk_names = join['spk_names']
        n_spk = len(spk_names)
        spk_ptr = np.zeros(n_spk + 1, np.int64)
        np.add.at(spk_ptr, spk_of + 1, 1)
        spk_ptr = np.cumsum(spk_ptr)
        new_vid = np.ones(len(rows), bool)
        new_vid[1:] = (spk_of[1:] != spk_of[:-1]) | (vid_of[1:] != vid_of[:-1])
        spk_nvid = np.bincount(spk_of[new_vid], minlength=n_spk)

        spk_arm, langs = assign_arms(list(spk_names), arms, spk2lang,
                                     vox2_lang, spk2corpus)
        lang_names, spk_lang = np.unique(np.array(langs), return_inverse=True)

        # per-utterance conditions and per-speaker summaries
        keys = join['keys']
        cond = condition_arrays(keys)
        lo, hi = spk_ptr[:-1], spk_ptr[1:]
        spk_nseg = np.zeros(n_spk, np.int64)
        spk_dial = np.zeros(n_spk, np.int64)
        spk_has_std = np.zeros(n_spk, bool)
        spk_ngenre = np.zeros(n_spk, np.int64)
        is_3ds = cond.seg[lo] >= 0  # by the speaker's first position
        for s in np.flatnonzero(is_3ds):
            sg = cond.seg[lo[s]:hi[s]]
            dl = cond.dial[lo[s]:hi[s]]
            spk_nseg[s] = len(np.unique(sg[sg >= 0]))
            spk_has_std[s] = bool((dl == 0).any())
            d = dl[dl > 0]
            if len(d):
                spk_dial[s] = int(np.bincount(d).argmax())
        for s in np.flatnonzero(cond.genre[lo] >= 0):
            g = cond.genre[lo[s]:hi[s]]
            spk_ngenre[s] = len(np.unique(g[g >= 0]))
        info = dict(n_3ds_utts=int((cond.seg >= 0).sum()),
                    n_3ds_segments=len(cond.seg_names),
                    n_cn2_genre_utts=int((cond.genre >= 0).sum()))

        ks = _ks_tables(keys, policy, spk_ptr, info)

        if forbid_same_segment:
            # K rows from K different segments need >= K segments
            bad = is_3ds & (spk_nseg < K) & (spk_arm >= 0)
            spk_arm[bad] = -1
            info['excluded_lt_K_segments'] = int(bad.sum())

        share_cells, cells_info = _share_cells(
            cells, arm_names, arm_weights, spk_arm, lang_names, spk_lang)

        class_names = join['class_names']
        table = cls(
            index=index, rows=rows, spk_of=spk_of, vid_of=vid_of,
            spk_ptr=spk_ptr, spk_nvid=spk_nvid, spk_names=spk_names,
            class_names=class_names,
            spk_class=np.searchsorted(class_names, spk_names).astype(np.int64),
            n_missing=join['n_missing'], n_short=join['n_short'],
            arm_names=arm_names, arm_weights=arm_weights, spk_arm=spk_arm,
            lang_names=lang_names, spk_lang=spk_lang, policy=policy,
            share_cells=share_cells, K=int(K),
            forbid_same_segment=bool(forbid_same_segment),
            seg_of=cond.seg, dev_of=cond.dev, dcls_of=cond.dcls,
            dial_of=cond.dial, genre_of=cond.genre,
            n_segments=len(cond.seg_names),
            genre_names=tuple(cond.genre_names), spk_nseg=spk_nseg,
            spk_dial=spk_dial, spk_has_std=spk_has_std, spk_ngenre=spk_ngenre,
            ks_phase=ks['phase'], ks_dial=ks['dial'],
            ks_dial_names=ks['dial_names'], spk_ks_both=ks['both'],
            spk_ks_stddial=ks['stddial'], info=info, cells_info=cells_info)
        logger.info(table.summary())
        return table


# ----------------------------------------------------------------------------
# construction steps


def _join_label_to_index(utt_spk: Sequence[Tuple[str, str]], index: TarIndex,
                         min_len: int) -> Dict[str, Any]:
    """Label keys -> index rows, sorted by (speaker, session), stable."""
    keys = np.array([u for u, _ in utt_spk], dtype='S')
    spks = np.array([s for _, s in utt_spk])
    class_names = np.unique(spks)
    ikeys = np.asarray(index.keys)
    order = np.argsort(ikeys, kind='stable')
    sorted_keys = ikeys[order]
    pos = np.searchsorted(sorted_keys, keys)
    pos = np.minimum(pos, len(order) - 1)
    found = sorted_keys[pos] == keys
    rows = order[pos]
    ok = found.copy()
    ns = np.asarray(index.nsamples)[rows]
    n_short = int((ok & (ns < min_len)).sum())
    ok &= ns >= min_len
    rows, keys, spks = rows[ok], keys[ok], spks[ok]
    spk_names, spk_id = np.unique(spks, return_inverse=True)
    # the session name starts with the speaker field of the key, so the
    # sessions of two speakers are distinct
    vid_names = np.array([session_of(k) for k in keys])
    _, vid_id = np.unique(vid_names, return_inverse=True)
    o = np.lexsort((vid_id, spk_id))
    return dict(rows=rows[o].astype(np.int64), keys=keys[o],
                spk_of=spk_id[o].astype(np.int64),
                vid_of=vid_id[o].astype(np.int64), spk_names=spk_names,
                class_names=class_names, n_missing=int((~found).sum()),
                n_short=n_short)


def _ks_tables(keys: np.ndarray, policy: Sequence[Optional[PolicySpec]],
               spk_ptr: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
    """KeSpeech arrays; all "not applicable" without a 'ks_contrast' arm."""
    n, n_spk = len(keys), len(spk_ptr) - 1
    both = np.zeros(n_spk, bool)
    stddial = np.zeros(n_spk, bool)
    ks_pols = [p for p in policy if p and p.pair_policy == 'ks_contrast']
    if not ks_pols:
        none = np.full(n, -1, np.int64)
        return dict(phase=none, dial=none.copy(), dial_names=(), both=both,
                    stddial=stddial)
    files = sorted({p.subdialect_file for p in ks_pols if p.subdialect_file})
    ks = ks_arrays(keys, read_subdialect(files))
    lo, hi = spk_ptr[:-1], spk_ptr[1:]
    is_ks = ks.phase[lo] >= 0  # by the speaker's first position
    for s in np.flatnonzero(is_ks):
        ph = ks.phase[lo[s]:hi[s]]
        dl = ks.dial[lo[s]:hi[s]]
        both[s] = bool((ph == 1).any() and (ph == 2).any())
        stddial[s] = bool(((ph == 1) & (dl == 0)).any() and
                          ((ph == 1) & (dl > 0)).any())
    info['ks'] = dict(
        utts=ks.n_utts, utts_without_subdialect=ks.n_without_subdialect,
        subdialects=[x.decode() for x in ks.dial_names],
        speakers=int(is_ks.sum()), speakers_both_phases=int(both.sum()),
        speakers_std_and_dialect=int(stddial.sum()))
    return dict(phase=ks.phase, dial=ks.dial,
                dial_names=tuple(ks.dial_names), both=both, stddial=stddial)


def _share_cells(cells: Sequence[CellSpec], arm_names: Sequence[str],
                 arm_weights: np.ndarray, spk_arm: np.ndarray,
                 lang_names: np.ndarray, spk_lang: np.ndarray
                 ) -> Tuple[Tuple[ShareCell, ...], Dict[str, Any]]:
    """Resolve the share cells to speaker masks (in the given order)."""
    out, info = [], {}
    wn = arm_weights / arm_weights.sum()
    for c in cells:
        if c.arm not in arm_names:
            raise ValueError('cells.%s.arm %r is not one of %s' % (
                c.name, c.arm, list(arm_names)))
        a = list(arm_names).index(c.arm)
        li = np.flatnonzero(lang_names == str(c.spk2lang))
        mask = (spk_arm == a) & np.isin(spk_lang, li)
        share = float(c.share)
        if not 0.0 < share < wn[a]:
            raise ValueError('cells.%s.share %r must lie in (0, %g) (arm '
                             'weight)' % (c.name, share, wn[a]))
        if not mask.any():
            raise ValueError('cells.%s selects no speaker' % c.name)
        if any(x.arm == a and (x.mask & mask).any() for x in out):
            raise ValueError('cells.%s overlaps another share cell' % c.name)
        out.append(ShareCell(c.name, a, share, mask))
        info[c.name] = dict(arm=c.arm, share=share, speakers=int(mask.sum()),
                            arm_speakers=int((spk_arm == a).sum()))
    return tuple(out), info
