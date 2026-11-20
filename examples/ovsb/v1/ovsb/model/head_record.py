"""Record of the speaker list a classifier head was built for.

Row ``k * c + j`` of the head belongs to class ``c``, and class ``c`` is the
``c``-th speaker of the sorted training list. A checkpoint loaded into a
model for another list would give every class the rows of some other
speaker without any shape error, so the list is recorded next to the
checkpoint and checked before training loads it:

* ``<checkpoint>.head.json`` (the sidecar), written by the tool that built
  the checkpoint's head;
* ``<model_dir>/head_spk.json`` (the run record), written by the first
  successful check of a run and used when the run is resumed from one of
  its own checkpoints, which have no sidecar.

A record holds ``spk_sha256`` (sha256 of the sorted unique speaker ids
joined by newlines), ``num_classes`` and ``rows``. The check fails closed:
without a record, or with a record that cannot be read, training stops.
"""
import hashlib
import json
import os
from typing import Any, Dict, Iterable, Mapping, Optional

RUN_RECORD = 'head_spk.json'
# keys of records made for head variants this recipe does not build
_FOREIGN_KEYS = ('virtual_classes', 'k_per_class_sha256')


class HeadMismatchError(RuntimeError):
    """The head of a checkpoint cannot be shown to match the class list."""


def speaker_list_digest(spks: Iterable[str]) -> str:
    """sha256 of the sorted unique speaker ids, one per line."""
    return hashlib.sha256('\n'.join(sorted(set(spks))).encode()).hexdigest()


def head_sidecar_path(ckpt_path: str) -> str:
    """Path of the sidecar of a checkpoint."""
    return ckpt_path + '.head.json'


def _write_json(path: str, record: Mapping[str, Any]) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf8') as fout:
        json.dump(record, fout, indent=1)
    os.replace(tmp, path)


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf8') as fin:
            record = json.load(fin)
    except (OSError, ValueError) as exc:
        raise HeadMismatchError('%s: cannot read the head record (%s)' %
                                (path, exc)) from exc
    if not isinstance(record, dict):
        raise HeadMismatchError('%s: the head record is not a mapping' % path)
    return record


def write_head_sidecar(ckpt_path: str, spks: Iterable[str], rows: int,
                       source: Optional[str] = None,
                       extra: Optional[Mapping[str, Any]] = None
                       ) -> Dict[str, Any]:
    """Write ``<ckpt_path>.head.json`` and return the record.

    Args:
        ckpt_path: checkpoint the head belongs to.
        spks: speaker ids of the training list (any order, repeats allowed).
        rows: rows of the head weight, a multiple of the number of speakers.
        source: where the head came from (free text, e.g. a checkpoint path).
        extra: further keys to store (provenance of the tool).
    """
    spks = sorted(set(spks))
    if not spks or rows % len(spks) != 0:
        raise ValueError('head rows %d are not a multiple of %d classes' %
                         (rows, len(spks)))
    record = dict(spk_sha256=speaker_list_digest(spks), num_classes=len(spks),
                  rows=int(rows), rows_per_class=int(rows) // len(spks),
                  source=source)
    record.update(extra or {})
    _write_json(head_sidecar_path(ckpt_path), record)
    return record


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def verify_head(meta: Mapping[str, Any], spks: Iterable[str], rows: int,
                what: str) -> None:
    """Raise :class:`HeadMismatchError` unless ``meta`` describes this head.

    Args:
        meta: a head record (sidecar or run record).
        spks: speaker ids of the current training list.
        rows: rows of the head weight of the model being trained.
        what: name of the record for the error message.
    """
    spks = set(spks)
    digest = speaker_list_digest(spks)
    bad = []
    if meta.get('spk_sha256') != digest:
        bad.append('speaker-list sha256 %s != %s' %
                   (meta.get('spk_sha256'), digest))
    if _as_int(meta.get('num_classes')) != len(spks):
        bad.append('classes %s != %d' % (meta.get('num_classes'), len(spks)))
    if _as_int(meta.get('rows')) != int(rows):
        bad.append('rows %s != %d' % (meta.get('rows'), rows))
    for key in _FOREIGN_KEYS:
        if meta.get(key):
            bad.append('%s=%r: a head variant this recipe does not build' %
                       (key, meta[key]))
    if bad:
        raise HeadMismatchError('%s: head built for another class list: %s' %
                                (what, '; '.join(bad)))


def check_run_head(ckpt_path: str, model_dir: str, spks: Iterable[str],
                   rows: int, write: bool = True) -> str:
    """Check the head of ``ckpt_path`` against the class list before loading.

    The sidecar of the checkpoint is used when it exists, else the run
    record of ``model_dir``. After a successful check the run record is
    written if the run has none yet (``write``; pass True on one process
    only).

    Args:
        ckpt_path: checkpoint about to be loaded.
        model_dir: the run's model directory.
        spks: speaker ids of the current training list.
        rows: rows of the head weight of the model being trained.
        write: write the run record after the check.

    Returns:
        'sidecar' or 'run': the record the head was verified against.

    Raises:
        HeadMismatchError: no record exists, a record cannot be read, or
            it describes another class list or row count.
    """
    spks = sorted(set(spks))
    run_file = os.path.join(model_dir, RUN_RECORD)
    side = head_sidecar_path(ckpt_path)
    if os.path.exists(side):
        how, record = 'sidecar', side
    elif os.path.exists(run_file):
        how, record = 'run', run_file
    else:
        raise HeadMismatchError(
            '%s: neither %s nor %s exists, so the class list of this head '
            'cannot be verified; build the initial checkpoint with the '
            'recipe tools, which write the sidecar' %
            (ckpt_path, side, run_file))
    verify_head(_read_json(record), spks, rows, record)
    if write and not os.path.exists(run_file):
        _write_json(run_file, dict(
            spk_sha256=speaker_list_digest(spks), num_classes=len(spks),
            rows=int(rows), verified_from=how, source=ckpt_path))
    return how
