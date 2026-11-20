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
"""Random access into WeSpeaker tar shards without copying the audio.

An index is a directory of ``.npy`` arrays, loaded with mmap so that every
DataLoader worker of every rank shares the same page-cache pages::

    shards.txt       one shard path per line (as written in the shard list)
    keys.npy         (N,) bytes  utterance key = tar member name without its
                     last suffix, i.e. the ``key`` of the shard pipeline
    shard.npy        (N,) int32  row in shards.txt
    offset.npy       (N,) int64  byte offset of the audio member in the tar
    size.npy         (N,) int64  byte size of the audio member
    nsamples.npy     (N,) int64  PCM frames from the RIFF header (-1 unknown)
    sample_rate.npy  (N,) int32  from the RIFF header (-1 unknown)
    meta.json        build information; ``root`` is the directory against
                     which relative shard paths were resolved

Command line::

    python -m ovsb.data.index build --shard_list L --out_dir D [--root R]
    python -m ovsb.data.index merge --out_dir D --part DIR[:KEYS] ...
    python -m ovsb.data.index check --index_dir D [--root R] [--label L]

Relative shard paths are resolved against, in this order: the ``root``
argument, the ``root`` recorded in ``meta.json``, the working directory.
Pass ``root`` when the index or its shards were moved.
"""

import argparse
import io
import json
import logging
import os
import struct
import tarfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing import Pool
from typing import (Any, Collection, Dict, List, Optional, Sequence, Tuple,
                    Union)

import numpy as np
import torch

AUDIO_SUFFIXES = ('wav', 'flac')
INDEX_ARRAYS = ('keys', 'shard', 'offset', 'size', 'nsamples', 'sample_rate')
_BLOCK = 512

logger = logging.getLogger(__name__)

WavHeader = Tuple[int, int, int, int, int, int]
ShardRow = Tuple[str, int, int, int, int]


# ----------------------------------------------------------------------------
# RIFF header and decoding


def parse_wav_header(buf: bytes) -> Optional[WavHeader]:
    """Parse the first bytes of a RIFF/WAVE file.

    Returns ``(sample_rate, channels, bits, format_tag, data_offset,
    data_bytes)``, or None when ``buf`` is not RIFF/WAVE or its ``data``
    chunk header lies beyond ``buf``.
    """
    if len(buf) < 12 or buf[:4] != b'RIFF' or buf[8:12] != b'WAVE':
        return None
    pos = 12
    sr = ch = bits = tag = None
    while pos + 8 <= len(buf):
        cid = buf[pos:pos + 4]
        csz = struct.unpack('<I', buf[pos + 4:pos + 8])[0]
        if cid == b'fmt ':
            if pos + 24 > len(buf):
                return None
            tag, ch, sr = struct.unpack('<HHI', buf[pos + 8:pos + 16])
            bits = struct.unpack('<H', buf[pos + 22:pos + 24])[0]
            if tag == 0xFFFE and pos + 34 <= len(buf):  # WAVE_FORMAT_EXTENSIBLE
                tag = struct.unpack('<H', buf[pos + 32:pos + 34])[0]
        elif cid == b'data':
            if sr is None:
                return None
            return sr, ch, bits, tag, pos + 8, csz
        pos += 8 + csz + (csz & 1)
    return None


def decode_wav(data: bytes) -> Tuple[torch.Tensor, int]:
    """Decode the raw bytes of an audio member.

    Returns ``(waveform float32 (C, T), sample_rate)``, equal to
    ``torchaudio.load`` with normalisation for PCM16 and float32 RIFF files;
    every other format is handed to torchaudio.
    """
    h = parse_wav_header(data[:4096])
    if h is not None:
        sr, ch, bits, tag, off, n = h
        n = min(n, len(data) - off)
        if tag == 1 and bits == 16:
            x = np.frombuffer(data, dtype='<i2', count=n // (2 * ch) * ch,
                              offset=off)
            w = torch.from_numpy(x.astype(np.float32) / 32768.0)
            return w.reshape(-1, ch).t().contiguous(), sr
        if tag == 3 and bits == 32:
            x = np.frombuffer(data, dtype='<f4', count=n // (4 * ch) * ch,
                              offset=off)
            w = torch.from_numpy(x.copy())
            return w.reshape(-1, ch).t().contiguous(), sr
    import torchaudio
    w, sr = torchaudio.load(io.BytesIO(data))
    return w, sr


# ----------------------------------------------------------------------------
# build


def _scan_shard_fast(path: str) -> List[ShardRow]:
    """Walk the tar headers of one shard with ``os.pread``.

    Returns ``(key, offset, size, nsamples, sample_rate)`` per audio member
    and raises on anything unusual (the caller falls back to ``tarfile``).
    """
    out = []
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        except (AttributeError, OSError):
            pass
        end = os.fstat(fd).st_size
        pos = 0
        buf, buf_pos = b'', 0
        while pos + _BLOCK <= end:
            # one pread usually covers the label member, the audio member
            # header and the first bytes of the RIFF header
            if not (buf_pos <= pos and pos + _BLOCK <= buf_pos + len(buf)):
                buf = os.pread(fd, 2048, pos)
                buf_pos = pos
            hdr = buf[pos - buf_pos:pos - buf_pos + _BLOCK]
            if hdr.count(0) == _BLOCK:
                break  # end-of-archive
            ti = tarfile.TarInfo.frombuf(hdr, 'utf-8', 'surrogateescape')
            if ti.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
                raise ValueError('member type %r' % ti.type)
            data_off = pos + _BLOCK
            name = ti.name
            dot = name.rfind('.')
            if dot > 0 and name[dot + 1:] in AUDIO_SUFFIXES:
                need = data_off + min(ti.size, 256)
                if not (buf_pos <= data_off and need <= buf_pos + len(buf)):
                    buf = os.pread(fd, 2048, data_off)
                    buf_pos = data_off
                head = buf[data_off - buf_pos:]
                h = parse_wav_header(head)
                if h is None:
                    head = os.pread(fd, 4096, data_off)
                    h = parse_wav_header(head)
                if h is not None:
                    sr, ch, bits, _tag, data_pos, nbytes = h
                    nbytes = min(nbytes, ti.size - data_pos)
                    ns = nbytes // max(1, ch * bits // 8)
                else:
                    sr, ns = -1, -1
                out.append((name[:dot], data_off, ti.size, ns, sr))
            pos = data_off + ((ti.size + _BLOCK - 1) // _BLOCK) * _BLOCK
    finally:
        os.close(fd)
    return out


def _scan_shard_tarfile(path: str) -> List[ShardRow]:
    """Same rows as :func:`_scan_shard_fast`, through ``tarfile``."""
    out = []
    with tarfile.open(path, mode='r:') as tf:
        for ti in tf:
            name = ti.name
            dot = name.rfind('.')
            if not ti.isfile() or dot <= 0 or \
                    name[dot + 1:] not in AUDIO_SUFFIXES:
                continue
            f = tf.fileobj
            f.seek(ti.offset_data)
            h = parse_wav_header(f.read(4096))
            if h is not None:
                sr, ch, bits, _tag, data_pos, nbytes = h
                ns = min(nbytes, ti.size - data_pos) // max(1, ch * bits // 8)
            else:
                sr, ns = -1, -1
            out.append((name[:dot], ti.offset_data, ti.size, ns, sr))
    return out


def scan_shard(path: str) -> Tuple[str, List[ShardRow], Optional[str]]:
    """Scan one shard: ``(path, rows, error)``; fast walk, tarfile fallback."""
    try:
        return path, _scan_shard_fast(path), None
    except Exception as e1:  # noqa: B902
        try:
            return path, _scan_shard_tarfile(path), 'fallback: %r' % (e1, )
        except Exception as e2:  # noqa: B902
            return path, [], 'FAILED: %r / %r' % (e1, e2)


def _read_lines(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _save_arrays(out_dir: str, shards: Sequence[str],
                 arrays: Dict[str, np.ndarray]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'shards.txt'), 'w') as f:
        f.write('\n'.join(shards) + '\n')
    for name in INDEX_ARRAYS:
        np.save(os.path.join(out_dir, name + '.npy'), arrays[name])


def build_index(shard_list: str, out_dir: str, nj: int = 4,
                root: Optional[str] = None) -> Dict[str, Any]:
    """Scan every shard of ``shard_list`` and write the index to ``out_dir``.

    Relative shard paths are resolved against ``root`` (default: the working
    directory) and stored as written. Only headers are read: one 2 KiB pread
    per utterance, ``nj`` shards in parallel. Returns the ``meta.json`` dict.
    """
    shards = _read_lines(shard_list)
    t0 = time.time()
    paths = [s if os.path.isabs(s) or root is None else os.path.join(root, s)
             for s in shards]
    keys, sid, off, size, ns, sr = [], [], [], [], [], []
    errors = {}
    with Pool(nj) as pool:
        for i, (_path, rows, err) in enumerate(
                pool.imap(scan_shard, paths, chunksize=4)):
            if err:
                errors[shards[i]] = err
            for k, o, z, n, r in rows:
                keys.append(k)
                sid.append(i)
                off.append(o)
                size.append(z)
                ns.append(n)
                sr.append(r)
            if (i + 1) % 200 == 0:
                logger.info('%d/%d shards, %d utterances, %.0f s', i + 1,
                            len(shards), len(keys), time.time() - t0)
    _save_arrays(out_dir, shards, dict(
        keys=np.array(keys, dtype='S'), shard=np.array(sid, np.int32),
        offset=np.array(off, np.int64), size=np.array(size, np.int64),
        nsamples=np.array(ns, np.int64), sample_rate=np.array(sr, np.int32)))
    meta = dict(shard_list=os.path.abspath(shard_list),
                root=root or os.getcwd(), n_shards=len(shards),
                n_utts=len(keys),
                n_unknown_len=int(sum(1 for n in ns if n < 0)),
                errors=errors, seconds=round(time.time() - t0, 1),
                built=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    return meta


# ----------------------------------------------------------------------------
# merge


@dataclass(frozen=True)
class IndexPart:
    """One input of :func:`merge_index`.

    Attributes:
        index_dir: index directory to take rows from.
        keys: utterance keys to keep (every one must be in the index);
            None keeps every row.
        keep_paths: keep the shard paths as written (they are then relative
            to the ``root`` of the merged index). Otherwise relative paths
            are made absolute against the ``root`` in the part's
            ``meta.json`` (or the parent of ``index_dir`` without one).
        name: label recorded in the merged ``meta.json``.
    """
    index_dir: str
    keys: Optional[Collection[str]] = None
    keep_paths: bool = False
    name: str = ''


def _meta_root(index_dir: str) -> Optional[str]:
    path = os.path.join(index_dir, 'meta.json')
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f).get('root')


def merge_index(parts: Sequence[IndexPart], out_dir: str,
                root: Optional[str] = None, note: str = '') -> Dict[str, Any]:
    """Concatenate (subsets of) several indexes into one index.

    Rows keep their order inside each part, parts follow each other in the
    given order and shard rows are renumbered. ``root`` (default: the working
    directory) is recorded as the directory the kept relative shard paths
    resolve against. Returns the ``meta.json`` dict.
    """
    acc = {n: [] for n in INDEX_ARRAYS}
    shards_out = []
    n_utts = 0
    for part in parts:
        shards = _read_lines(os.path.join(part.index_dir, 'shards.txt'))
        arr = {n: np.load(os.path.join(part.index_dir, n + '.npy'),
                          mmap_mode='r') for n in INDEX_ARRAYS}
        if part.keys is None:
            sel = np.arange(len(arr['keys']))
        else:
            want = {k.encode() for k in part.keys}
            keys = np.asarray(arr['keys'])
            sel = np.flatnonzero(np.fromiter((k in want for k in keys), bool,
                                             len(keys)))
            if len(sel) != len(want):
                raise ValueError('%s: %d wanted keys are not in %s' % (
                    part.name, len(want) - len(sel), part.index_dir))
        if not part.keep_paths:
            base = _meta_root(part.index_dir) or \
                os.path.dirname(part.index_dir)
            shards = [s if os.path.isabs(s) else os.path.join(base, s)
                      for s in shards]
        for n in INDEX_ARRAYS:
            v = np.asarray(arr[n])[sel]
            acc[n].append(v + len(shards_out) if n == 'shard' else v)
        shards_out += shards
        n_utts += len(sel)
        logger.info('index part %s: %d rows, %d shards', part.name, len(sel),
                    len(shards))
    arrays = {n: np.concatenate(acc[n]) for n in INDEX_ARRAYS}
    arrays['keys'] = arrays['keys'].astype('S')
    _save_arrays(out_dir, shards_out, arrays)
    meta = dict(built=time.strftime('%Y-%m-%d %H:%M:%S'),
                parts=[(p.name, p.index_dir) for p in parts], n_utts=n_utts,
                n_shards=len(shards_out), root=root or os.getcwd(), note=note)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    return meta


# ----------------------------------------------------------------------------
# read


class TarIndex:
    """The mmap'd index arrays of one index directory (see the module doc).

    Args:
        index_dir: directory written by :func:`build_index` or
            :func:`merge_index`.
        root: directory for relative shard paths; overrides the ``root``
            recorded in ``meta.json`` (use it when the data moved).
        mmap: map the arrays read-only instead of loading them.

    Attributes:
        keys, shard, offset, size, nsamples, sample_rate: the index arrays.
        shards: resolved shard paths (row ``shard[i]`` holds utterance i).
        root: the directory used for relative shard paths (None = working
            directory).
    """

    def __init__(self, index_dir: str, root: Optional[str] = None,
                 mmap: bool = True) -> None:
        self.dir = index_dir
        mode = 'r' if mmap else None
        for name in INDEX_ARRAYS:
            setattr(self, name, np.load(
                os.path.join(index_dir, name + '.npy'), mmap_mode=mode))
        self.root = root if root is not None else _meta_root(index_dir)
        self.shards = [
            s if os.path.isabs(s) or self.root is None else
            os.path.join(self.root, s)
            for s in _read_lines(os.path.join(index_dir, 'shards.txt'))]

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    def key(self, i: int) -> str:
        """Utterance key of row ``i``."""
        return self.keys[i].decode('utf-8')


class TarReader:
    """pread-based reader of index rows.

    File descriptors are cached per process (LRU, at most ``max_open``). The
    reader is safe to build in the main process: descriptors are opened
    lazily and the cache is dropped when the pid changes (fork).
    """

    def __init__(self, index: Union[TarIndex, str], max_open: int = 8192,
                 root: Optional[str] = None) -> None:
        self.index = index if isinstance(index, TarIndex) else \
            TarIndex(index, root=root)
        self.max_open = int(max_open)
        self._fds = OrderedDict()
        self._pid = os.getpid()
        self.n_open = 0

    def _fd(self, sid: int) -> int:
        if self._pid != os.getpid():  # forked worker: never share offsets
            self._fds = OrderedDict()
            self._pid = os.getpid()
        fd = self._fds.get(sid)
        if fd is not None:
            self._fds.move_to_end(sid)
            return fd
        fd = os.open(self.index.shards[sid], os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        except (AttributeError, OSError):
            pass
        self.n_open += 1
        self._fds[sid] = fd
        if len(self._fds) > self.max_open:
            _, old = self._fds.popitem(last=False)
            os.close(old)
        return fd

    def read(self, i: int) -> bytes:
        """Raw member bytes of index row ``i``."""
        i = int(i)
        size = int(self.index.size[i])
        fd = self._fd(int(self.index.shard[i]))
        data = os.pread(fd, size, int(self.index.offset[i]))
        if len(data) != size:
            raise IOError('short read %d/%d of %s' % (
                len(data), size, self.index.key(i)))
        return data

    def load(self, i: int) -> Tuple[torch.Tensor, int]:
        """``(waveform float32 (C, T), sample_rate)`` of index row ``i``."""
        return decode_wav(self.read(i))

    def close(self) -> None:
        """Close every cached file descriptor."""
        for fd in self._fds.values():
            os.close(fd)
        self._fds = OrderedDict()


# ----------------------------------------------------------------------------
# check


def check_index(index_dir: str, root: Optional[str] = None, n_read: int = 200,
                seed: int = 0, label: Optional[str] = None) -> Dict[str, Any]:
    """Verify an index against the shards it points to.

    Checks that the arrays have one length and valid shard rows, that every
    shard file exists, that ``n_read`` random rows can be read and carry the
    recorded RIFF length and sample rate, and (with ``label``, an
    ``utt2spk``-style file) that every listed key is in the index. Returns a
    report whose ``ok`` entry is True when nothing failed.
    """
    index = TarIndex(index_dir, root=root)
    n = len(index)
    report = dict(index_dir=index_dir, root=index.root, n_utts=n,
                  n_shards=len(index.shards))
    report['arrays_ok'] = bool(
        all(len(getattr(index, a)) == n for a in INDEX_ARRAYS)
        and (n == 0 or (int(index.shard.min()) >= 0
                        and int(index.shard.max()) < len(index.shards))))
    missing = [s for s in index.shards if not os.path.isfile(s)]
    report['missing_shards'] = len(missing)
    report['first_missing_shard'] = missing[0] if missing else None
    bad_rows = []
    if report['arrays_ok'] and not missing and n:
        reader = TarReader(index)
        rows = np.random.default_rng(seed).choice(n, size=min(n_read, n),
                                                  replace=False)
        for i in np.sort(rows).tolist():
            try:
                h = parse_wav_header(reader.read(i)[:4096])
            except OSError:
                bad_rows.append(i)
                continue
            ns, sr = int(index.nsamples[i]), int(index.sample_rate[i])
            if h is None:
                ok = ns == -1 and sr == -1
            else:
                nbytes = min(h[5], int(index.size[i]) - h[4])
                ok = sr == h[0] and ns == nbytes // max(1, h[1] * h[2] // 8)
            if not ok:
                bad_rows.append(i)
        reader.close()
        report['rows_read'] = len(rows)
    report['bad_rows'] = bad_rows[:20]
    n_missing_keys = 0
    if label:
        with open(label) as f:
            want = np.array([line.split()[0] for line in f if line.strip()],
                            dtype='S')
        have = np.sort(np.asarray(index.keys))
        pos = np.minimum(np.searchsorted(have, want), max(n - 1, 0))
        n_missing_keys = int((have[pos] != want).sum()) if n else len(want)
        report['label_keys'] = len(want)
        report['label_keys_missing'] = n_missing_keys
    report['ok'] = bool(report['arrays_ok'] and not missing and not bad_rows
                        and n_missing_keys == 0)
    return report


# ----------------------------------------------------------------------------
# command line


def _parse_part(spec: str) -> IndexPart:
    """``DIR[:KEYS_FILE]`` -> IndexPart (KEYS_FILE: first column = key)."""
    index_dir, _, keys_file = spec.partition(':')
    keys = None
    if keys_file:
        with open(keys_file) as f:
            keys = {line.split()[0] for line in f if line.strip()}
    return IndexPart(index_dir, keys, name=os.path.basename(
        os.path.normpath(index_dir)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point; returns the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='index the shards of a shard list')
    b.add_argument('--shard_list', required=True)
    b.add_argument('--out_dir', required=True)
    b.add_argument('--nj', type=int, default=4)
    b.add_argument('--root', default=None,
                   help='directory for relative shard paths (default: cwd)')
    m = sub.add_parser('merge', help='concatenate indexes')
    m.add_argument('--out_dir', required=True)
    m.add_argument('--part', action='append', required=True,
                   metavar='DIR[:KEYS]',
                   help='index dir, optionally restricted to the keys in the '
                   'first column of KEYS; shard paths become absolute')
    m.add_argument('--root', default=None)
    m.add_argument('--note', default='')
    c = sub.add_parser('check', help='verify an index against its shards')
    c.add_argument('--index_dir', required=True)
    c.add_argument('--root', default=None)
    c.add_argument('--n_read', type=int, default=200)
    c.add_argument('--seed', type=int, default=0)
    c.add_argument('--label', default=None,
                   help='utt2spk-style file whose keys must be in the index')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(message)s')
    if a.cmd == 'build':
        meta = dict(build_index(a.shard_list, a.out_dir, a.nj, a.root))
        meta['errors'] = len(meta['errors'])
        print(json.dumps(meta))
        return 0
    if a.cmd == 'merge':
        parts = [_parse_part(p) for p in a.part]
        print(json.dumps(merge_index(parts, a.out_dir, a.root, a.note)))
        return 0
    report = check_index(a.index_dir, a.root, a.n_read, a.seed, a.label)
    print(json.dumps(report))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
