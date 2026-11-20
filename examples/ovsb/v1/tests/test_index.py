"""Tar index: build, merge, check, relocation, reading and decoding."""

import filecmp
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_parity as dp  # noqa: E402

from ovsb.data import index as ix  # noqa: E402


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp('toy_index')
    t = dp.make_toy(root)
    t['meta'] = ix.build_index(t['files']['shard.list'],
                               os.path.join(t['root'], 'index'), nj=2,
                               root=t['root'])
    t['index_dir'] = os.path.join(t['root'], 'index')
    return t


def test_build_arrays_and_meta(toy):
    idx = ix.TarIndex(toy['index_dir'])
    n = len(toy['utts'])
    assert len(idx) == n == toy['meta']['n_utts']
    assert toy['meta']['n_shards'] == 2 and toy['meta']['errors'] == {}
    assert toy['meta']['n_unknown_len'] == 0
    assert toy['meta']['root'] == toy['root']
    assert [a.dtype.str for a in (idx.shard, idx.offset, idx.size,
                                  idx.nsamples, idx.sample_rate)] == \
        ['<i4', '<i8', '<i8', '<i8', '<i4']
    assert idx.keys.dtype.kind == 'S'
    assert sorted(idx.key(i) for i in range(n)) == \
        sorted(k for k, _ in toy['utts'])
    # rows follow the shard order, then the member order inside a shard
    assert (np.diff(idx.shard) >= 0).all()
    for s in range(2):
        assert (np.diff(idx.offset[np.asarray(idx.shard) == s]) > 0).all()
    for i in range(n):
        blob = toy['blobs'][idx.key(i)]
        assert idx.size[i] == len(blob)
        assert idx.nsamples[i] == (len(blob) - 44) // 2
        assert idx.sample_rate[i] == dp.SR
    with open(os.path.join(toy['index_dir'], 'shards.txt')) as f:
        assert f.read().split() == toy['shards']  # stored as written


def test_reader_bytes_and_decode(toy):
    reader = ix.TarReader(toy['index_dir'])
    idx = reader.index
    for i in range(len(idx)):
        blob = toy['blobs'][idx.key(i)]
        assert reader.read(i) == blob
        w, sr = reader.load(i)
        ref = np.frombuffer(blob, '<i2', offset=44).astype(np.float32) / 32768
        assert sr == dp.SR and w.dtype == torch.float32
        assert w.shape == (1, len(ref)) and np.array_equal(w[0].numpy(), ref)
    assert reader.n_open == 2
    reader.close()
    assert reader.read(0) == toy['blobs'][idx.key(0)]  # reopens lazily
    reader.close()


def test_reader_fd_cache_is_bounded(toy):
    reader = ix.TarReader(toy['index_dir'], max_open=1)
    for i in (0, len(reader.index) - 1, 0):
        reader.read(i)
    assert reader.n_open == 3 and len(reader._fds) == 1
    reader.close()


def test_decode_wav_formats():
    rng = np.random.default_rng(1)
    stereo = dp.wav_bytes(100, rng, channels=2)
    w, sr = ix.decode_wav(stereo)
    ref = np.frombuffer(stereo, '<i2', offset=44).astype(np.float32) / 32768
    assert w.shape == (2, 100) and sr == dp.SR
    assert np.array_equal(w.numpy(), ref.reshape(-1, 2).T)
    x = rng.standard_normal(64).astype('<f4')
    f32 = b'RIFF' + struct.pack('<I', 36 + x.nbytes) + b'WAVEfmt ' + \
        struct.pack('<IHHIIHH', 16, 3, 1, 8000, 32000, 4, 32) + b'data' + \
        struct.pack('<I', x.nbytes) + x.tobytes()
    w, sr = ix.decode_wav(f32)
    assert sr == 8000 and np.array_equal(w[0].numpy(), x)
    # a data chunk that claims more bytes than the member holds is clipped
    w, _ = ix.decode_wav(stereo[:44 + 40])
    assert w.shape == (2, 10)
    assert ix.parse_wav_header(b'not a wav file') is None
    assert ix.parse_wav_header(stereo[:30]) is None


def test_decode_wav_equals_torchaudio(toy):
    torchaudio = pytest.importorskip('torchaudio')
    import io
    blob = next(iter(toy['blobs'].values()))
    try:
        ref, sr = torchaudio.load(io.BytesIO(blob))
    except Exception as e:  # noqa: B902
        pytest.skip('torchaudio cannot decode from memory here: %r' % (e, ))
    w, sr2 = ix.decode_wav(blob)
    assert sr == sr2 and torch.equal(w, ref)


def test_root_resolution_and_relocation(toy, tmp_path):
    # default: the root recorded at build time
    assert ix.TarIndex(toy['index_dir']).root == toy['root']
    # index and shards moved: the recorded root is stale, the override works
    moved = tmp_path / 'moved'
    shutil.copytree(toy['index_dir'], moved / 'idx')
    shutil.copytree(os.path.join(toy['root'], 'shards'), moved / 'shards')
    with open(moved / 'idx' / 'meta.json') as f:
        meta = json.load(f)
    meta['root'] = str(tmp_path / 'gone')
    with open(moved / 'idx' / 'meta.json', 'w') as f:
        json.dump(meta, f)
    stale = ix.TarReader(str(moved / 'idx'))
    with pytest.raises(OSError):
        stale.read(0)
    reader = ix.TarReader(str(moved / 'idx'), root=str(moved))
    assert reader.read(3) == toy['blobs'][reader.index.key(3)]
    reader.close()
    # no meta.json: relative paths resolve against the working directory
    os.remove(moved / 'idx' / 'meta.json')
    idx = ix.TarIndex(str(moved / 'idx'))
    assert idx.root is None and idx.shards == toy['shards']
    assert ix.check_index(str(moved / 'idx'), root=str(moved))['ok']
    assert not ix.check_index(str(moved / 'idx'), root=str(tmp_path))['ok']


def test_merge_index(toy, tmp_path):
    full = ix.TarIndex(toy['index_dir'])
    keys = [full.key(i) for i in range(len(full))]
    # a second corpus in its own directory with its own relative shard paths
    rng = np.random.default_rng(5)
    extra = {'new_%d/s/%d' % (i, u): dp.wav_bytes(1500, rng)
             for i in range(3) for u in range(2)}
    other = tmp_path / 'other'
    os.makedirs(other / 'sh')
    dp.write_shard(str(other / 'sh' / 'a.tar'),
                   [(k, k.split('/')[0], b) for k, b in extra.items()])
    with open(other / 'shard.list', 'w') as f:
        f.write('sh/a.tar\n')
    ix.build_index(str(other / 'shard.list'), str(other / 'index'), nj=1,
                   root=str(other))
    want = sorted(extra)[1:4]
    out = str(tmp_path / 'merged')
    meta = ix.merge_index(
        [ix.IndexPart(toy['index_dir'], keep_paths=True, name='base'),
         ix.IndexPart(str(other / 'index'), keys=want, name='new')],
        out, root=toy['root'], note='test')
    assert meta['n_utts'] == len(keys) + 3 and meta['n_shards'] == 3
    assert meta['parts'] == [('base', toy['index_dir']),
                             ('new', str(other / 'index'))]
    merged = ix.TarIndex(out)
    # the first part is a prefix of every array, unchanged
    for a in ix.INDEX_ARRAYS:
        assert np.array_equal(getattr(merged, a)[:len(keys)],
                              getattr(full, a)), a
        assert getattr(merged, a).dtype.kind == getattr(full, a).dtype.kind
    assert [merged.key(i) for i in range(len(keys), len(merged))] == want
    assert (np.asarray(merged.shard[len(keys):]) == 2).all()
    with open(os.path.join(out, 'shards.txt')) as f:
        assert f.read().split() == toy['shards'] + [str(other / 'sh/a.tar')]
    reader = ix.TarReader(merged)
    for i in range(len(merged)):
        k = merged.key(i)
        assert reader.read(i) == (toy['blobs'].get(k) or extra.get(k))
    reader.close()
    assert ix.check_index(out)['ok']
    with pytest.raises(ValueError, match='1 wanted keys'):
        ix.merge_index([ix.IndexPart(str(other / 'index'),
                                     keys=want + ['absent/key'])],
                       str(tmp_path / 'bad'))


def test_check_index_and_cli(toy, tmp_path, capsys):
    rep = ix.check_index(toy['index_dir'], label=toy['files']['utt2spk'])
    assert rep['arrays_ok'] and rep['missing_shards'] == 0
    assert rep['bad_rows'] == [] and rep['rows_read'] == min(200, len(toy['utts']))
    # the label file lists one key that is not in the index
    assert rep['label_keys_missing'] == 1 and not rep['ok']
    assert ix.main(['check', '--index_dir', toy['index_dir']]) == 0
    assert json.loads(capsys.readouterr().out)['ok'] is True
    # a wrong length in the index is caught
    bad = tmp_path / 'bad'
    shutil.copytree(toy['index_dir'], bad)
    ns = np.load(bad / 'nsamples.npy')
    ns[:] += 1
    np.save(bad / 'nsamples.npy', ns)
    assert ix.main(['check', '--index_dir', str(bad)]) == 1
    capsys.readouterr()
    # build and merge through the command line
    out = str(tmp_path / 'cli')
    assert ix.main(['build', '--shard_list', toy['files']['shard.list'],
                    '--out_dir', out, '--nj', '1', '--root',
                    toy['root']]) == 0
    for a in ix.INDEX_ARRAYS:
        assert filecmp.cmp(os.path.join(out, a + '.npy'), os.path.join(
            toy['index_dir'], a + '.npy'), shallow=False), a
    keys_file = tmp_path / 'keys'
    with open(keys_file, 'w') as f:
        f.writelines('%s x\n' % k for k, _ in toy['utts'][:7])
    merged = str(tmp_path / 'cli_merged')
    assert ix.main(['merge', '--out_dir', merged, '--part',
                    '%s:%s' % (out, keys_file)]) == 0
    m = ix.TarIndex(merged)
    assert len(m) == 7 and all(os.path.isabs(s) for s in m.shards)
    assert ix.check_index(merged)['ok']


def test_module_entry_point(toy):
    env = dict(os.environ, PYTHONPATH=dp.RECIPE_DIR)
    r = subprocess.run([sys.executable, '-m', 'ovsb.data.index', 'check',
                        '--index_dir', toy['index_dir'], '--n_read', '5'],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0 and json.loads(r.stdout)['rows_read'] == 5


# ----------------------------------------------------------------------------
# parity with the SOURCE tree

BUILD_WITH_SOURCE = r'''
import sys
from wespeaker.dataset.tar_index import build_index
build_index(sys.argv[1], sys.argv[2], nj=2, root=sys.argv[3], log=None)
'''


@dp.needs_source
def test_parity_build_equals_source(toy, tmp_path):
    out = str(tmp_path / 'source_index')
    env = dict(os.environ, PYTHONPATH=dp.SOURCE_TREE,
               PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, '-c', BUILD_WITH_SOURCE,
                    toy['files']['shard.list'], out, toy['root']],
                   env=env, cwd=str(tmp_path), check=True)  # not the recipe
    for name in [a + '.npy' for a in ix.INDEX_ARRAYS] + ['shards.txt']:
        assert filecmp.cmp(os.path.join(out, name), os.path.join(
            toy['index_dir'], name), shallow=False), name
    with open(os.path.join(out, 'meta.json')) as f:
        ref = json.load(f)
    drop = ('seconds', 'built')
    assert {k: v for k, v in ref.items() if k not in drop} == \
        {k: v for k, v in toy['meta'].items() if k not in drop}


@dp.needs_source
def test_parity_real_reader_rows_byte_identical():
    """200 random rows of the real index: raw bytes and decoded waveforms
    equal the SOURCE reader's (hashes from the source subprocess)"""
    _src, info = dp.real_source()
    _label, ep, _min_len, _seed, _n = dp.real_job()
    reader = ix.TarReader(os.path.join(dp.SOURCE_RECIPE, ep['index']),
                          root=dp.SOURCE_RECIPE)
    assert len(reader.index) == info['n_utts']  # nothing dropped
    assert len(info['reads']) == 200
    for i, sha, size, wav_sha, sr in info['reads']:
        data = reader.read(i)
        assert len(data) == size, i
        assert hashlib.sha256(data).hexdigest() == sha, i
        w, sr2 = ix.decode_wav(data)
        assert sr2 == sr, i
        assert hashlib.sha256(w.numpy().tobytes()).hexdigest() == wav_sha, i
    reader.close()


@dp.needs_source
def test_parity_real_index_reproduced_by_merge():
    """merge_index over the parts recorded in the real index's meta.json
    writes the same files (every .npy and shards.txt byte-identical)"""
    _label, ep, _min_len, _seed, _n = dp.real_job()
    real_dir = os.path.join(dp.SOURCE_RECIPE, ep['index'])
    with open(os.path.join(real_dir, 'meta.json')) as f:
        meta = json.load(f)
    parts = meta.get('parts') or []
    if len(parts) != 2 or not all(os.path.isdir(d) for _, d in parts):
        pytest.skip('the parts of the real index are not available')
    real = ix.TarIndex(real_dir)
    n_base = len(np.load(os.path.join(parts[0][1], 'keys.npy'),
                         mmap_mode='r'))
    want = [k.decode() for k in np.asarray(real.keys[n_base:])]
    os.makedirs(dp.CACHE, exist_ok=True)
    out = tempfile.mkdtemp(prefix='merge_', dir=dp.CACHE)
    try:
        new = ix.merge_index(
            [ix.IndexPart(parts[0][1], keep_paths=True, name=parts[0][0]),
             ix.IndexPart(parts[1][1], keys=want, name=parts[1][0])],
            out, root=meta['root'], note=meta['note'])
        for name in [a + '.npy' for a in ix.INDEX_ARRAYS] + ['shards.txt']:
            assert filecmp.cmp(os.path.join(out, name),
                               os.path.join(real_dir, name),
                               shallow=False), name
        assert {k: v for k, v in new.items() if k != 'built'} == \
            {k: (list(map(tuple, v)) if k == 'parts' else v)
             for k, v in meta.items() if k != 'built'}
    finally:
        shutil.rmtree(out)
