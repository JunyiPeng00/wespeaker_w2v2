"""Several LMDB sources drawn with fixed weights (mixed RIR sets).

:class:`MultiLmdbData` stands in wherever a core ``LmdbData`` is read:
``random_one()``, ``.keys`` and ``.db.begin().get(key)``. A draw costs one
``random.random()`` for the source, then the source's own ``random_one()``.
"""

import random
from typing import Any, List, Optional, Sequence, Tuple, Union

from ovsb.config.schema import SIMULATED_REVERB, AugmentCfg
from wespeaker.dataset.lmdb_data import LmdbData


class _MultiTxn:
    """Read transaction over several sources (key -> owning source)."""

    def __init__(self, multi: 'MultiLmdbData') -> None:
        self.multi = multi

    def __enter__(self) -> '_MultiTxn':
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def get(self, key: Union[bytes, str]) -> Optional[bytes]:
        """Value of ``key`` in the source that owns it, else None."""
        k = key.decode() if isinstance(key, bytes) else key
        src = self.multi.owner.get(k)
        if src is None:
            return None
        with src.db.begin(write=False) as txn:
            return txn.get(k.encode())


class _MultiDb:
    """The ``.db`` of a :class:`MultiLmdbData` (``begin()`` only)."""

    def __init__(self, multi: 'MultiLmdbData') -> None:
        self.multi = multi

    def begin(self, write: bool = False) -> _MultiTxn:
        """A read transaction (``write`` is ignored)."""
        return _MultiTxn(self.multi)


class MultiLmdbData:
    """Sources drawn with the given (unnormalised) weights.

    ``keys`` is the union of the sources' keys in source order; a key that
    occurs in several sources belongs to the first one.
    """

    def __init__(self, sources: Sequence[Any],
                 weights: Sequence[float]) -> None:
        assert len(sources) == len(weights) and len(sources) > 0
        w = [float(x) for x in weights]
        assert min(w) >= 0 and sum(w) > 0, weights
        self.sources = list(sources)
        self.weights = [x / sum(w) for x in w]
        self.cum: List[float] = []
        acc = 0.0
        for x in self.weights:
            acc += x
            self.cum.append(acc)
        self.cum[-1] = 1.0
        self.keys: List[str] = []
        self.owner = {}
        for s in self.sources:
            for k in s.keys:
                if k not in self.owner:
                    self.owner[k] = s
                    self.keys.append(k)
        self.db = _MultiDb(self)
        self.num_excluded = sum(getattr(s, 'num_excluded', 0)
                                for s in self.sources)

    def pick_source(self) -> int:
        """Index of a random source (one ``random.random()``)."""
        u = random.random()
        for i, c in enumerate(self.cum):
            if u < c:
                return i
        return len(self.cum) - 1

    def random_one(self) -> Tuple[str, bytes]:
        """(key, value) of a random item of a random source."""
        return self.sources[self.pick_source()].random_one()


def open_reverb_source(cfg: AugmentCfg) -> MultiLmdbData:
    """The RIR source of an augment config.

    ``cfg.reverb_sources`` lists LMDB directories; the entry ``simulated``
    stands for ``cfg.reverb_lmdb``.
    """
    return MultiLmdbData(
        [LmdbData(cfg.reverb_lmdb if s == SIMULATED_REVERB else s)
         for s in cfg.reverb_sources], cfg.reverb_weights)


def open_noise_source(cfg: AugmentCfg) -> LmdbData:
    """The noise source of an augment config, without the excluded keys."""
    return LmdbData(cfg.noise_lmdb, exclude_keys=cfg.noise_exclude_list)
