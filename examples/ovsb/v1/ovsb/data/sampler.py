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
"""Speaker-aware P x K episodic sampler over tar shards.

An *episode* is P speakers with K utterances each, emitted as P * K
consecutive samples grouped by speaker (K consecutive rows per speaker). The
sampler is an ``IterableDataset``: every DataLoader worker of every DDP rank
runs its own endless stream of episodes.

Cells
    In epoch e all speakers are shuffled by one permutation that is the same
    on every rank and worker (seed key ``(seed, e, 7)``) and dealt round-robin
    into ``world_size * num_workers`` disjoint *cells*. Worker w of rank r
    draws only from cell ``r * num_workers + w``. Speakers are therefore
    exclusive across the ranks of a step and across the workers of a rank,
    and the split changes every epoch. All draws of a cell come from one
    generator keyed ``(seed, e, r, w)``.

    Rank-split arms (``rank_split_arms``, default none): the speakers of a
    listed arm are split over the ranks only. Every worker of rank r draws
    them from the union of the cells of rank r, each worker with its own
    cycle. They stay exclusive across the ranks of a step; a batch comes
    from one worker, so it never holds a speaker of such an arm twice as
    long as the arm has enough speakers per rank. Meant for a small arm
    with a large weight (see the known limitation below).

Slots and quotas
    The speakers of a cell are grouped by arm. A *share cell* (a named
    subset of one arm, e.g. the speakers of one language) is split off into
    its own group. Arms and share cells together are the *slots* of the
    stratification: slot weights are the normalised arm weights, a share
    cell takes its ``share`` from its arm. Per episode the P speakers are
    divided over the slots (:func:`stratified_counts`): ``floor(P * w)`` per
    slot, the remaining speakers by systematic sampling over the fractional
    parts, so the expected count of every slot is exactly ``P * w``. A slot
    that has no speaker in the cell gets weight 0 for that cell and its
    quota is redistributed over the other slots in proportion to their
    weights (counted, see :class:`SamplerStats`). Rows drawn through a share
    cell are tagged with the arm of the cell.

Speakers of a slot
    A slot hands out its speakers from a cycled random permutation: every
    speaker of the slot (in this cell) is drawn once before any is drawn a
    second time. In a weighted arm (see ``ovsb.data.weights``) the cycle is a
    *pool* in which a speaker of weight w appears ``floor(w)`` or
    ``floor(w) + 1`` times (stochastic rounding, redrawn every cycle); with
    a ``spk_weight_share`` rule the weights are first rescaled inside the
    cell so that the listed speakers hold the target share of the mass.
    A speaker that is already in the episode is drawn again, up to 8 times;
    in a weighted arm the rejected pool entries are *deferred* and served
    first in later episodes, so no mass is lost.

    Known limitation, kept on purpose: when a slot of a cell has fewer
    speakers than its quota for an episode (a small arm with a large weight
    split over many cells), the retries cannot succeed and the same speaker
    enters the episode more than once. :class:`SamplerStats` counts these
    repeats per arm.

Utterances of a speaker
    The K rows of every drawn speaker come from the pair policy of its arm
    (``ovsb.data.policies``).

Audio
    ``__iter__`` reads every row with a ``TarReader``. A row that cannot be
    read is replaced by another utterance of the same speaker (up to 5
    attempts), drawn from a separate generator keyed ``(seed, e, r, w, 1)``,
    so a read error never shifts the episode stream.

The seed keys and the order and number of random draws define the training
data stream; they must not change. Logging never draws.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import (Any, Dict, Iterator, List, NamedTuple, Optional,
                    Sequence, Tuple)

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from ovsb.data.index import TarReader
from ovsb.data.policies import PTYPES, DrawCounters, draw_rows
from ovsb.data.speakers import ShareCell, SpeakerTable
from ovsb.data.weights import SpeakerWeights, share_weights

logger = logging.getLogger(__name__)

# attempts to replace a speaker that is already in the episode
MAX_SPEAKER_RETRIES = 8
# attempts to read an utterance of a speaker before giving up
MAX_READ_ATTEMPTS = 5
# fields of an emitted sample, in order
SAMPLE_FIELDS = ('key', 'spk', 'wav', 'sample_rate', 'arm', 'lang', 'vid',
                 'kpos', 'ptype', 'seg', 'dev', 'dcls', 'dial', 'genre')


class Episode(NamedTuple):
    """One drawn episode, index only (no audio).

    All arrays have P * K entries, K consecutive entries per speaker.

    Attributes:
        pos: utterance positions (indices into the table's sorted
            utterance order).
        arm: arm index of every row.
        kpos: index of the row inside its K-group (0 .. K - 1).
        ptype: positive type of the row's group (index into
            ``ovsb.data.policies.PTYPES``).
    """
    pos: np.ndarray
    arm: np.ndarray
    kpos: np.ndarray
    ptype: np.ndarray


@dataclass
class SamplerStats:
    """Counters of one cell's episode stream (reset when a stream starts).

    Sizes: A arms, S slots (the arms, then the share cells).

    Attributes:
        episodes: episodes drawn.
        spk: speakers drawn (``episodes * P``).
        fill: fill-up cases of the cross-session draw.
        arm_draws: (A,) speakers drawn per arm.
        ptype: (A, len(PTYPES)) K-groups per arm and positive type.
        cell_draws: speakers drawn through every share cell.
        repeat_speakers: (A,) draws that put a speaker into an episode that
            already held it (the slot had fewer speakers in the cell than
            its quota, or 8 redraws all collided).
        empty_slot_episodes: (S,) episodes in which the slot had a positive
            weight but no speaker in the cell.
        redistributed: (S,) expected speaker draws (``P * weight`` per such
            episode) that went to the other slots instead.
    """
    n_arms: int
    n_slots: int
    episodes: int = 0
    spk: int = 0
    fill: DrawCounters = field(default_factory=DrawCounters)
    arm_draws: np.ndarray = None
    ptype: np.ndarray = None
    cell_draws: np.ndarray = None
    repeat_speakers: np.ndarray = None
    empty_slot_episodes: np.ndarray = None
    redistributed: np.ndarray = None

    def __post_init__(self) -> None:
        self.arm_draws = np.zeros(self.n_arms, np.int64)
        self.ptype = np.zeros((self.n_arms, len(PTYPES)), np.int64)
        self.cell_draws = np.zeros(self.n_slots - self.n_arms, np.int64)
        self.repeat_speakers = np.zeros(self.n_arms, np.int64)
        self.empty_slot_episodes = np.zeros(self.n_slots, np.int64)
        self.redistributed = np.zeros(self.n_slots, np.float64)


def stratified_counts(P: int, weights: Sequence[float],
                      rng: np.random.Generator) -> np.ndarray:
    """Split P draws over slots with the given (unnormalised) weights.

    Every slot gets ``floor(P * w)``; the remaining ``rem`` draws go to
    slots chosen by systematic sampling over the fractional parts: the
    fractional parts are laid end to end on a line of length ``rem`` and
    the slots at ``u, u + 1, ..`` (one uniform ``u``) get one more draw.
    The inclusion probability of a slot equals its fractional part, so
    ``E[count] = P * w`` exactly. One random number is consumed when
    ``rem > 0``, none otherwise.

    Returns the (len(weights),) int64 counts, summing to P.
    """
    w = np.asarray(weights, np.float64)
    w = w / w.sum()
    exp = P * w
    cnt = np.floor(exp).astype(np.int64)
    rem = int(P - cnt.sum())
    if rem > 0:
        frac = exp - cnt
        c = np.cumsum(frac)
        c *= rem / c[-1]
        pts = rng.random() + np.arange(rem)
        idx = np.searchsorted(c, pts, side='right')
        cnt[np.minimum(idx, len(w) - 1)] += 1
    return cnt


def slot_weights(arm_weights: Sequence[float],
                 share_cells: Sequence[ShareCell]) -> np.ndarray:
    """Arm weights -> normalised slot weights (arms, then share cells).

    The share of every cell is taken from its arm.
    """
    w = np.asarray(arm_weights, np.float64)
    w = w / w.sum()
    out = list(w)
    for c in share_cells:
        out[c.arm] -= c.share
        out.append(c.share)
    out = np.array(out)
    if not (out >= -1e-12).all():
        raise ValueError('share cells exceed their arm weight: %s' % out)
    return np.maximum(out, 0.0)


class EpisodicSampler(IterableDataset):
    """Stream of P x K episodes (see the module doc).

    Built in the main process; it holds numpy arrays only, shared
    copy-on-write with forked DataLoader workers.

    Args:
        table: static speaker and utterance tables.
        weights: speaker weights of ``table`` (``build_weights(table)``).
        P: speakers per episode.
        K: utterances per speaker; must equal ``table.K``.
        sample_num_per_epoch: utterances per epoch over all ranks (0 = the
            number of lines of the label file). Only used for
            :meth:`episodes_per_worker`.
        seed: base seed of every generator.
        stratify: remainder rule of the per-episode quotas; only
            'systematic' is implemented.
        log_every: episodes between stat lines of worker 0 of every rank
            (0 = no stat lines at all).
        rank_split_arms: names of the arms whose speakers are split over
            the ranks only (module doc); empty = every arm is split over
            the cells.
        log_at_end: also emit the stat lines once when a worker's stream is
            closed (the end of its epoch), for every worker. With
            ``log_every`` larger than the episodes a worker delivers per
            epoch this is the only line.

    Attributes:
        epoch, stream: set by :meth:`set_epoch` before the workers start.
        stats: counters of the most recently started stream of this
            process (:class:`SamplerStats`).
    """

    def __init__(self, table: SpeakerTable, weights: SpeakerWeights, P: int,
                 K: int, sample_num_per_epoch: int = 0, seed: int = 0,
                 stratify: str = 'systematic', log_every: int = 200,
                 log_at_end: bool = True,
                 rank_split_arms: Sequence[str] = ()) -> None:
        if int(K) != table.K:
            raise ValueError('K %d differs from the table K %d' % (K, table.K))
        if stratify != 'systematic':
            raise ValueError("stratify %r: only 'systematic' is implemented"
                             % (stratify, ))
        if int(P) < 1:
            raise ValueError('P must be >= 1, got %r' % (P, ))
        unknown = [a for a in rank_split_arms if a not in table.arm_names]
        if unknown:
            raise ValueError('rank_split_arms %s are not arms %s' % (
                unknown, list(table.arm_names)))
        self.rank_split_arms = tuple(rank_split_arms)
        self._rank_split = np.array(
            [table.arm_names.index(a) for a in self.rank_split_arms],
            np.int64)
        self.table = table
        self.weights = weights
        self.P, self.K = int(P), int(K)
        self.seed = int(seed)
        self.epoch = 0
        self.stream = 0
        self.log_every = int(log_every or 0)
        self.log_at_end = bool(log_at_end)
        self.sample_num_per_epoch = int(sample_num_per_epoch or 0) or (
            table.n_utts + table.n_missing + table.n_short)
        self.reader = TarReader(table.index)
        n_arms = len(table.arm_names)
        self.slot_arm = list(range(n_arms)) + [c.arm
                                               for c in table.share_cells]
        self.slot_names = list(table.arm_names) + [c.name
                                                   for c in table.share_cells]
        # slot weights as the episode loop uses them (normalised only with
        # share cells; stratified_counts normalises in any case)
        self._slot_w = slot_weights(table.arm_weights, table.share_cells) \
            if table.share_cells else table.arm_weights
        self.stats = SamplerStats(n_arms, len(self.slot_arm))

    # ------------------------------------------------------------------
    # epoch, cells, seeds

    def set_epoch(self, epoch: int, stream: int = 0) -> None:
        """Set the epoch (and the stream id) of the following iterations.

        ``stream`` is 0 for a run that goes through the epoch once. A
        restart inside an epoch passes a larger id so that the restarted
        epoch does not replay the draws already trained on; stream 0 uses
        exactly the seed keys of the module doc.
        """
        self.epoch = int(epoch)
        self.stream = int(stream)

    def _key(self, epoch: int, *tail: int) -> Tuple[int, ...]:
        """Seed key ``(seed, epoch) + tail``; for stream > 0 a 6-entry key
        ``(seed, epoch, rank, worker, purpose, stream)`` that cannot equal
        a stream-0 key (purpose: 0 draws, 1 read retries, 7 cell split)"""
        if not self.stream:
            return (self.seed, epoch) + tail
        if tail == (7, ):
            tail = (0, 0, 7)
        elif len(tail) == 2:
            tail = tail + (0, )
        return (self.seed, epoch) + tail + (self.stream, )

    @staticmethod
    def dist_info() -> Tuple[int, int, int, int]:
        """``(rank, world_size, worker id, num_workers)`` of this process."""
        rank, world = 0, 1
        if dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
        wi = torch.utils.data.get_worker_info()
        wid, nw = (0, 1) if wi is None else (wi.id, wi.num_workers)
        return rank, world, wid, nw

    def cell_speakers(self, epoch: int, rank: int, world: int, wid: int,
                      nw: int) -> np.ndarray:
        """Sorted speaker ids of cell (rank, worker) in ``epoch``.

        Includes speakers without an arm; the episode stream drops them.
        The speakers of a rank-split arm are those of all cells of the
        rank.
        """
        n_cells = world * nw
        perm = np.random.default_rng(self._key(epoch, 7)).permutation(
            self.table.n_spk)
        cell = perm[(rank * nw + wid)::n_cells]
        if len(self._rank_split):
            spk_arm = self.table.spk_arm
            cell = cell[~np.isin(spk_arm[cell], self._rank_split)]
            # positions of the cells of this rank in the round-robin deal
            of_rank = (np.arange(len(perm)) % n_cells) // nw == rank
            mine = perm[of_rank]
            cell = np.concatenate(
                [cell, mine[np.isin(spk_arm[mine], self._rank_split)]])
        return np.sort(cell)

    def episodes_per_worker(self, world: int, nw: int) -> int:
        """Upper bound of the episodes one worker delivers per epoch.

        An epoch is ``sample_num_per_epoch // world // (P * K)`` batches
        per rank, served round-robin by the ``nw`` workers; a batch
        consumes at most one episode.
        """
        batches = self.sample_num_per_epoch // world // (self.P * self.K)
        return int(math.ceil(batches / float(nw)))

    # ------------------------------------------------------------------
    # episodes

    def iter_episodes(self, epoch: Optional[int] = None,
                      rank: Optional[int] = None, world: Optional[int] = None,
                      wid: Optional[int] = None, nw: Optional[int] = None,
                      max_episodes: Optional[int] = None
                      ) -> Iterator[Episode]:
        """Yield the episodes of one cell without reading audio.

        Arguments left None default to the current epoch and to
        :meth:`dist_info` of this process. The stream is endless unless
        ``max_episodes`` is given. ``self.stats`` is replaced by the
        counters of this stream.
        """
        r0, w0, i0, n0 = self.dist_info()
        epoch = self.epoch if epoch is None else epoch
        rank = r0 if rank is None else rank
        world = w0 if world is None else world
        wid = i0 if wid is None else wid
        nw = n0 if nw is None else nw
        self.stats = SamplerStats(len(self.table.arm_names),
                                  len(self.slot_arm))
        return self._episodes(epoch, rank, world, wid, nw, max_episodes,
                              self.stats)

    def _episodes(self, epoch: int, rank: int, world: int, wid: int, nw: int,
                  max_episodes: Optional[int],
                  stats: SamplerStats) -> Iterator[Episode]:
        """The episode stream of one cell (see the module doc)."""
        t = self.table
        wts = self.weights
        rng = np.random.default_rng(self._key(epoch, rank, wid))
        cell = self.cell_speakers(epoch, rank, world, wid, nw)
        cell = cell[t.spk_arm[cell] >= 0]
        n_arms = len(t.arm_names)
        slot_arm = self.slot_arm
        n_slots = len(slot_arm)
        # speakers of every slot: arms first, then the share cells, whose
        # speakers leave the pool of their arm
        slot_spk = [cell[t.spk_arm[cell] == a] for a in range(n_arms)]
        for c in t.share_cells:
            pool = slot_spk[c.arm]
            m = c.mask[pool]
            slot_spk[c.arm] = pool[~m]
            slot_spk.append(pool[m])
        slot_weighted = [wts.spk_w is not None and
                         bool(wts.arm_weighted[a]) for a in slot_arm]
        perms: List[Optional[np.ndarray]] = [None] * n_slots
        ppos = [0] * n_slots
        # weighted slots: a pool entry that collides with a speaker already
        # in the episode is deferred to a later episode instead of lost
        deferred: List[List[int]] = [[] for _ in range(n_slots)]

        avail = np.array([len(s) > 0 for s in slot_spk])
        w = np.where(avail, self._slot_w, 0.0)
        if not w.sum() > 0:
            raise ValueError('cell (rank %d, worker %d) of epoch %d has no '
                             'speaker of an arm with a positive weight' % (
                                 rank, wid, epoch))
        empty = (np.asarray(self._slot_w) > 0) & ~avail
        lost = np.where(empty, self.P * self._slot_w / self._slot_w.sum(),
                        0.0)

        def weighted_pool(a: int) -> np.ndarray:
            """one cycle of slot a: speaker s appears floor(w_s) or
            floor(w_s) + 1 times (stochastic rounding), permuted"""
            spk = slot_spk[a]
            ra = slot_arm[a]
            sw = wts.spk_w[spk]
            if wts.spk_share is not None and np.isfinite(wts.spk_share[ra]):
                sw = share_weights(sw, wts.spk_listed[spk], wts.spk_share[ra])
            n = np.floor(sw).astype(np.int64)
            n += rng.random(len(sw)) < (sw - n)
            pool = np.repeat(spk, n)
            return rng.permutation(pool if len(pool) else spk)

        def next_speaker(a: int, chosen: set) -> int:
            """next speaker of slot a: the oldest deferred entry that is
            not in the episode, else the next entry of the slot's cycle"""
            if slot_weighted[a] and deferred[a]:
                for j, s in enumerate(deferred[a]):
                    if s not in chosen:
                        return int(deferred[a].pop(j))
            if perms[a] is None or ppos[a] >= len(perms[a]):
                perms[a] = weighted_pool(a) if slot_weighted[a] else \
                    rng.permutation(slot_spk[a])
                ppos[a] = 0
            s = int(perms[a][ppos[a]])
            ppos[a] += 1
            return s

        n_ep = 0
        while max_episodes is None or n_ep < max_episodes:
            cnt = stratified_counts(self.P, w, rng)
            pos: List[int] = []
            arm: List[int] = []
            kpos: List[int] = []
            ptype: List[int] = []
            chosen: set = set()
            for a in range(n_slots):
                ra = slot_arm[a]
                for _ in range(int(cnt[a])):
                    s = next_speaker(a, chosen)
                    tries = 0
                    while s in chosen and tries < MAX_SPEAKER_RETRIES:
                        if slot_weighted[a]:
                            deferred[a].append(s)
                        s = next_speaker(a, chosen)
                        tries += 1
                    if s in chosen:
                        stats.repeat_speakers[ra] += 1
                    chosen.add(s)
                    us, pt = draw_rows(t, s, ra, rng, stats.fill)
                    stats.ptype[ra, pt] += 1
                    pos += us
                    arm += [ra] * len(us)
                    kpos += range(len(us))
                    ptype += [pt] * len(us)
                    stats.arm_draws[ra] += 1
                    if a >= n_arms:
                        stats.cell_draws[a - n_arms] += 1
            stats.episodes += 1
            stats.spk += self.P
            stats.empty_slot_episodes += empty
            stats.redistributed += lost
            n_ep += 1
            yield Episode(np.array(pos, np.int64), np.array(arm, np.int64),
                          np.array(kpos, np.int64), np.array(ptype, np.int64))

    def describe(self, ep: Episode) -> Dict[str, np.ndarray]:
        """Per-row integer fields of an episode, each (P * K,) int64.

        ``label`` classifier class of the speaker, ``row`` index row of the
        utterance, ``spk`` speaker id, then the fields emitted with every
        sample (``arm``, ``lang``, ``vid``, ``kpos``, ``ptype``, ``seg``,
        ``dev``, ``dcls``, ``dial``, ``genre``).
        """
        t = self.table
        u = ep.pos
        s = t.spk_of[u]
        return dict(label=t.spk_class[s], row=t.rows[u], spk=s, arm=ep.arm,
                    lang=t.spk_lang[s].astype(np.int64), vid=t.vid_of[u],
                    kpos=ep.kpos, ptype=ep.ptype, seg=t.seg_of[u],
                    dev=t.dev_of[u], dcls=t.dcls_of[u], dial=t.dial_of[u],
                    genre=t.genre_of[u])

    # ------------------------------------------------------------------
    # audio

    def load(self, u: int, rng: np.random.Generator
             ) -> Tuple[str, torch.Tensor, int]:
        """``(key, waveform, sample_rate)`` of utterance position ``u``.

        On a read error another utterance of the same speaker is tried
        (uniform over the speaker's utterances, drawn from ``rng``), so the
        K-group structure of the episode holds.
        """
        t = self.table
        s = int(t.spk_of[u])
        for _attempt in range(MAX_READ_ATTEMPTS):
            row = int(t.rows[u])
            try:
                wav, sr = self.reader.load(row)
                return t.index.key(row), wav, sr
            except Exception as e:  # any read or decode error
                logger.warning('episodic: read error %s (%r), retrying' % (
                    t.index.key(row), e))
                lo, hi = t.spk_ptr[s], t.spk_ptr[s + 1]
                u = int(lo + rng.integers(hi - lo))
        raise IOError('episodic: %d consecutive read errors for speaker %s'
                      % (MAX_READ_ATTEMPTS, t.spk_names[s]))

    def iter_samples(self, epoch: Optional[int] = None,
                     rank: Optional[int] = None, world: Optional[int] = None,
                     wid: Optional[int] = None, nw: Optional[int] = None,
                     max_episodes: Optional[int] = None
                     ) -> Iterator[Dict[str, Any]]:
        """Yield the samples of one cell's episodes, with audio.

        Arguments as :meth:`iter_episodes`. Every sample is a dict with the
        keys of :data:`SAMPLE_FIELDS`, in that order:

        ``key`` utterance key (str), ``spk`` speaker label (str), ``wav``
        float32 tensor (channels, samples), ``sample_rate`` (int), ``arm``
        arm index, ``lang`` language index (``table.lang_names``), ``vid``
        session id, ``kpos`` index inside the K-group, ``ptype`` positive
        type, and the recording conditions of the row ``seg``, ``dev``,
        ``dcls``, ``dial``, ``genre`` (-1 = not applicable). After a read
        retry ``key`` and ``wav`` belong to the replacement utterance while
        the other fields describe the row that was drawn.
        """
        r0, w0, i0, n0 = self.dist_info()
        epoch = self.epoch if epoch is None else epoch
        rank = r0 if rank is None else rank
        world = w0 if world is None else world
        wid = i0 if wid is None else wid
        nw = n0 if nw is None else nw
        t = self.table
        rng = np.random.default_rng(self._key(epoch, rank, wid, 1))
        st = self.stats = SamplerStats(len(t.arm_names), len(self.slot_arm))
        who = (rank, wid, epoch)
        t0 = time.time()
        logged = -1
        try:
            for ep in self._episodes(epoch, rank, world, wid, nw,
                                     max_episodes, st):
                for i, (u, a) in enumerate(zip(ep.pos.tolist(),
                                               ep.arm.tolist())):
                    key, wav, sr = self.load(u, rng)
                    s = int(t.spk_of[u])
                    yield dict(
                        key=key, spk=str(t.spk_names[s]), wav=wav,
                        sample_rate=sr, arm=a, lang=int(t.spk_lang[s]),
                        vid=int(t.vid_of[u]), kpos=int(ep.kpos[i]),
                        ptype=int(ep.ptype[i]), seg=int(t.seg_of[u]),
                        dev=int(t.dev_of[u]), dcls=int(t.dcls_of[u]),
                        dial=int(t.dial_of[u]), genre=int(t.genre_of[u]))
                if self.log_every and wid == 0 and \
                        st.episodes % self.log_every == 0:
                    self._log_stats(st, who, time.time() - t0)
                    logged = st.episodes
        finally:
            # the consumer stopped (end of the worker's epoch) or the
            # stream ended: one closing stat line per worker
            if self.log_every and self.log_at_end and st.episodes and \
                    logged != st.episodes:
                self._log_stats(st, who, time.time() - t0, end=True)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self.iter_samples()

    # ------------------------------------------------------------------
    # statistics

    def cell_stats(self, stats: Optional[SamplerStats] = None
                   ) -> Dict[str, Dict[str, Any]]:
        """Per share cell: speakers drawn, realised share of all speaker
        draws and the target share (of ``stats``, default ``self.stats``)"""
        st = self.stats if stats is None else stats
        tot = max(1, int(st.arm_draws.sum()))
        return {c.name: dict(draws=int(st.cell_draws[j]),
                             share=round(int(st.cell_draws[j]) / tot, 4),
                             target=c.share)
                for j, c in enumerate(self.table.share_cells)}

    def stat_lines(self, stats: SamplerStats, who: Tuple[int, int, int],
                   seconds: float, end: bool = False) -> List[str]:
        """The stat lines of a stream (rank, worker, epoch = ``who``)."""
        st = stats
        head = 'episodic rank %d worker %d epoch %d' % who
        if end:
            head += ' (end)'
        d = st.arm_draws
        names = self.table.arm_names
        out = [
            '%s: %d episodes, same-video fill %.4f of speakers, repeat fill '
            '%d, arm shares %s, weights %s, %.1f episodes/s' % (
                head, st.episodes, st.fill.same_video_fill / max(1, st.spk),
                st.fill.repeat_fill,
                np.round(d / max(1, d.sum()), 4).tolist(),
                np.round(self.table.arm_weights, 4).tolist(),
                st.episodes / max(seconds, 1e-9)),
            '%s: speakers repeated inside an episode %s (share of the '
            "arm's draws %s), slots without a speaker in this cell %s" % (
                head, {n: int(x) for n, x in zip(names, st.repeat_speakers)
                       if x},
                {n: round(int(x) / max(1, int(d[j])), 4)
                 for j, (n, x) in enumerate(zip(names, st.repeat_speakers))
                 if x},
                {n: dict(episodes=int(e), speakers_moved=round(float(q), 1))
                 for n, e, q in zip(self.slot_names, st.empty_slot_episodes,
                                    st.redistributed) if e})]
        pt = st.ptype.sum(0)
        out.append('%s: positive types %s' % (head, {
            PTYPES[j]: round(int(pt[j]) / max(1, int(pt.sum())), 4)
            for j in range(len(PTYPES)) if pt[j]}))
        if self.table.share_cells:
            out.append('%s: cells %s' % (head, self.cell_stats(st)))
        return out

    def _log_stats(self, stats: SamplerStats, who: Tuple[int, int, int],
                   seconds: float, end: bool = False) -> None:
        for line in self.stat_lines(stats, who, seconds, end):
            logger.warning(line)
