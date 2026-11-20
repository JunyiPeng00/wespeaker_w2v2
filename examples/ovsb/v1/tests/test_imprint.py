"""local/init/imprint_head.py: selection, head construction, sidecar.

Real check (skipped unless all are set): ``OVSB_INIT_REBUILT`` (output of
``imprint``), ``OVSB_INIT_REAL`` (the checkpoint a reference run started
from), ``OVSB_INIT_OLD_LABEL`` (list of the stage-1 head) and
``OVSB_INIT_NEW_LABEL`` (training list): rows of classes of the old list
must be bit-identical, imprinted rows must agree in direction.
"""

import argparse
import importlib.util
import io
import json
import os
import tarfile

import numpy as np
import pytest
import torch

RECIPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(RECIPE_DIR, 'local', 'init', 'imprint_head.py')
kaldiio = pytest.importorskip('kaldiio')


def _load():
    spec = importlib.util.spec_from_file_location('imprint_head', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ih = _load()
K, DIM = 3, 8


def _write_list(path, speakers):
    with open(path, 'w') as f:
        f.writelines('%s/s/u %s\n' % (s, s) for s in speakers)
    return str(path)


def _write_emb(tmp_path, name, vectors):
    """vectors: {key: (speaker, vector)} -> (scp, utt2spk) paths."""
    ark, scp = str(tmp_path / (name + '.ark')), str(tmp_path / (name + '.scp'))
    u2s = str(tmp_path / (name + '.utt2spk'))
    with kaldiio.WriteHelper('ark,scp:%s,%s' % (ark, scp)) as w, \
            open(u2s, 'w') as f:
        for key, (spk, vec) in vectors.items():
            w(key, np.asarray(vec, dtype=np.float32))
            f.write('%s %s\n' % (key, spk))
    return scp, u2s


def _args(tmp_path, **kw):
    base = dict(key=ih.HEAD_KEY, K=K, dup_cos=0.7, no_top1=False,
                frontend_num_layers=None, out=str(tmp_path / 'out.pt'))
    base.update(kw)
    return argparse.Namespace(**base)


def test_selftest(tmp_path):
    assert ih.main(['selftest', '--tmp', str(tmp_path)]) == 0


def test_session_of():
    assert ih.session_of('spk/video7/00001') == 'spk/video7'
    assert ih.session_of('cn2_id1/interview-02-013') == \
        'cn2_id1/interview-02'
    assert ih.session_of('cn2_id1/vlog-01-001-comb2') == 'cn2_id1/vlog-01'


def test_select_utts_round_robin_is_deterministic():
    utts = ['s/a/%d' % i for i in range(6)] + ['s/b/0', 's/c/0']
    first = ih.select_utts(utts, 4)
    assert first == ih.select_utts(list(reversed(utts)), 4)
    assert len(first) == 4 and len(set(first)) == 4
    # one utterance of every session before a second one of any
    assert {ih.session_of(u) for u in first[:3]} == {'s/a', 's/b', 's/c'}
    assert ih.select_utts(utts, 100) != utts  # pseudo-random order
    assert sorted(ih.select_utts(utts, 100)) == sorted(utts)


def test_build_head_rows():
    g = torch.Generator().manual_seed(1)
    old, new = ['b', 'd', 'x'], ['a', 'b', 'c', 'd']
    W = torch.randn(K * len(old), DIM, generator=g)
    e1, e2 = torch.randn(DIM, generator=g), torch.randn(DIM, generator=g)
    acc = {'a': e1 / e1.norm() + e2 / e2.norm(), 'c': e2 / e2.norm()}
    cnt = {'a': 2, 'c': 1}
    W2, imprinted, norm = ih.build_head(W, old, new, acc, cnt, K)
    assert W2.shape == (K * 4, DIM) and W2.dtype == W.dtype
    assert imprinted == {'a': 0, 'c': 2}
    assert norm == pytest.approx(float(W.norm(dim=1).mean()))
    assert torch.equal(W2[3:6], W[0:3])    # b: class 0 -> 1
    assert torch.equal(W2[9:12], W[3:6])   # d: class 1 -> 3; x is gone
    mean = (e1 / e1.norm() + e2 / e2.norm()) / 2
    expect = mean / mean.norm() * norm
    for k in range(K):
        assert torch.allclose(W2[k], expect, atol=1e-6)
        assert torch.equal(W2[k], W2[0])
    assert float(W2[6].norm()) == pytest.approx(norm, rel=1e-6)
    assert torch.allclose(W2[6] / W2[6].norm(), e2 / e2.norm(), atol=1e-6)


def test_build_head_needs_an_embedding_for_every_new_class():
    W = torch.zeros(K, DIM) + 1
    with pytest.raises(SystemExit, match='no imprint embedding'):
        ih.build_head(W, ['a'], ['a', 'b'], {}, {}, K)


def test_imprint_command(tmp_path):
    g = torch.Generator().manual_seed(2)
    old = ['id1', 'id2', 'id3']
    new = ['cn_1', 'id1', 'id3', 'tv_1']
    W = torch.randn(K * len(old), DIM, generator=g)
    init = str(tmp_path / 'init.pt')
    torch.save({ih.HEAD_KEY: W, 'pool.bias': torch.arange(5.0)}, init)
    from ovsb.model.head_record import verify_head, write_head_sidecar
    write_head_sidecar(init, old, K * len(old),
                       extra=dict(frontend_num_layers=22))
    c1, c2 = np.eye(DIM)[0], np.eye(DIM)[1]
    scp1, u1 = _write_emb(tmp_path, 'e1', {
        'cn_1/s/a__imp': ('cn_1', c1 * 3), 'cn_1/s/b__imp': ('cn_1', c1),
        'zz/s/a__imp': ('not_a_class', c2)})
    scp2, u2 = _write_emb(tmp_path, 'e2', {'tv_1/s/a__imp': ('tv_1', c2)})
    args = _args(tmp_path, init=init,
                 old_utt2spk=_write_list(tmp_path / 'old', old),
                 new_utt2spk=_write_list(tmp_path / 'new', new),
                 emb_scp=[scp1, scp2], emb_utt2spk=[u1, u2])
    assert ih.cmd_imprint(args) == 0
    state = torch.load(args.out, weights_only=False)
    assert list(state) == [ih.HEAD_KEY, 'pool.bias']
    assert torch.equal(state['pool.bias'], torch.arange(5.0))
    W2 = state[ih.HEAD_KEY]
    norm = float(W.norm(dim=1).mean())
    assert torch.equal(W2[3:6], W[0:3]) and torch.equal(W2[6:9], W[6:9])
    assert torch.allclose(W2[0], torch.tensor(c1, dtype=W.dtype) * norm,
                          atol=1e-5)
    assert torch.allclose(W2[9], torch.tensor(c2, dtype=W.dtype) * norm,
                          atol=1e-5)
    with open(args.out + '.json') as f:
        report = json.load(f)
    assert (report['kept'], report['dropped'], report['imprinted']) == \
        (2, 1, 2)
    with open(args.out + '.head.json') as f:
        side = json.load(f)
    verify_head(side, new, K * len(new), 'sidecar')
    assert side['rows_per_class'] == K
    assert side['frontend_num_layers'] == 22  # carried over
    assert side['imprinted']['classes'] == 2
    assert os.path.isfile(args.out + '.dup.tsv')


def test_imprint_refuses_a_head_of_another_list(tmp_path):
    init = str(tmp_path / 'init.pt')
    torch.save({ih.HEAD_KEY: torch.zeros(K * 2, DIM)}, init)
    scp, u2s = _write_emb(tmp_path, 'e', {'a/s/u__imp': ('a', np.ones(DIM))})
    args = _args(tmp_path, init=init,
                 old_utt2spk=_write_list(tmp_path / 'old', ['a', 'b', 'c']),
                 new_utt2spk=_write_list(tmp_path / 'new', ['a', 'b']),
                 emb_scp=[scp], emb_utt2spk=[u2s])
    with pytest.raises(SystemExit, match='not the list'):
        ih.cmd_imprint(args)
    assert not os.path.exists(args.out)


def test_imprint_refuses_other_class_sized_tensors(tmp_path):
    init = str(tmp_path / 'init.pt')
    torch.save({ih.HEAD_KEY: torch.ones(K * 2, DIM),
                'projection.bias': torch.zeros(K * 2)}, init)
    scp, u2s = _write_emb(tmp_path, 'e', {'c/s/u__imp': ('c', np.ones(DIM))})
    args = _args(tmp_path, init=init,
                 old_utt2spk=_write_list(tmp_path / 'old', ['a', 'b']),
                 new_utt2spk=_write_list(tmp_path / 'new', ['a', 'b', 'c']),
                 emb_scp=[scp], emb_utt2spk=[u2s])
    with pytest.raises(SystemExit, match='class-sized'):
        ih.cmd_imprint(args)


def _wav_bytes(n_samples, value):
    from scipy.io import wavfile
    buf = io.BytesIO()
    wavfile.write(buf, 16000, np.full(n_samples, value, dtype=np.int16))
    return buf.getvalue()


def test_list_command(tmp_path):
    from ovsb.data.index import build_index
    shard = tmp_path / 'shards_000.tar'
    utts = [('old1/v/a', 'old1', 8000), ('new1/v1/a', 'new1', 48000),
            ('new1/v1/b', 'new1', 8000), ('new1/v2/a', 'new1', 8000),
            ('new2/v/a', 'new2', 8000)]
    with tarfile.open(str(shard), 'w') as tar:
        for i, (key, spk, n) in enumerate(utts):
            for suffix, data in (('spk', spk.encode()),
                                 ('wav', _wav_bytes(n, i + 1))):
                info = tarfile.TarInfo(key + '.' + suffix)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    shard_list = tmp_path / 'shard.list'
    shard_list.write_text(str(shard) + '\n')
    index = str(tmp_path / 'tar_index')
    build_index(str(shard_list), index, nj=1)
    u2s = tmp_path / 'utt2spk'
    u2s.write_text(''.join('%s %s\n' % (k, s) for k, s, _ in utts))
    old = tmp_path / 'old_utt2spk'
    old.write_text('old1/v/a old1\n')
    out = tmp_path / 'imprint'
    assert ih.main(['list', '--utt2spk', str(u2s), '--index', index,
                    '--old_utt2spk', str(old), '--out', str(out),
                    '--per_spk', '2', '--max_sec', '1.0']) == 0
    rows = [json.loads(x) for x in (out / 'raw.list').read_text().split('\n')
            if x]
    assert [r['spk'] for r in rows] == ['new1', 'new1', 'new2']
    # two sessions of new1, one utterance each
    assert {ih.session_of(r['key'][:-5]) for r in rows[:2]} == \
        {'new1/v1', 'new1/v2'}
    assert all(r['key'].endswith('__imp') for r in rows)
    from scipy.io import wavfile
    for r in rows:
        rate, x = wavfile.read(r['wav'])
        assert rate == 16000 and 0 < len(x) <= 16000  # cut to max_sec
    lines = (out / 'utt2spk').read_text().splitlines()
    assert lines == ['%s %s' % (r['key'], r['spk']) for r in rows]
    with open(str(out / 'SUMMARY.json')) as f:
        assert json.load(f)['new_speakers'] == 2


@pytest.mark.skipif(not all(os.environ.get(k) for k in (
    'OVSB_INIT_REBUILT', 'OVSB_INIT_REAL', 'OVSB_INIT_OLD_LABEL',
    'OVSB_INIT_NEW_LABEL')), reason='OVSB_INIT_* not set')
def test_real_init_is_functionally_equivalent():
    def load(name):
        state = torch.load(os.environ[name], map_location='cpu',
                           weights_only=False, mmap=True)
        return state.get('model', state)[ih.HEAD_KEY]
    A, B = load('OVSB_INIT_REBUILT'), load('OVSB_INIT_REAL')
    old = set(ih.labels_sorted(os.environ['OVSB_INIT_OLD_LABEL']))
    new = ih.labels_sorted(os.environ['OVSB_INIT_NEW_LABEL'])
    assert A.shape == B.shape == (K * len(new), A.shape[1])
    kept = torch.tensor([j for j, s in enumerate(new) if s in old])
    imp = torch.tensor([j for j, s in enumerate(new) if s not in old])
    rows = (kept[:, None] * K + torch.arange(K)[None]).reshape(-1)
    assert torch.equal(A[rows], B[rows])
    cos = torch.nn.functional.cosine_similarity(
        A[imp * K].double(), B[imp * K].double(), dim=1)
    assert float(cos.min()) > 0.9999
