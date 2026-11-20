"""Shared fixtures of the data-layer tests (index, conditions, speakers, weights).

Two things live here:

* a small synthetic corpus (tar shards, label and side files) covering every
  key format the speaker table parses;
* the parity harness: the SOURCE sampler is built in a subprocess with
  ``PYTHONPATH=$OVSB_SOURCE_TREE`` (the two trees' ``wespeaker`` packages are
  never imported into one process) and its static tables are dumped to
  ``.npz`` / JSON for exact comparison.

Environment (parity tests skip without the first two):
  OVSB_SOURCE_TREE     root of the source repository
  OVSB_SOURCE_RECIPE   its recipe directory (relative data paths resolve here)
  OVSB_SOURCE_CONFIG   training config, relative to the recipe directory
                       (default conf/s3_base_v9c.yaml); it must be the
                       authored file: the resolved config.yaml of a run has
                       its keys sorted and loses the arm order
  OVSB_SOURCE_LABEL    training label file, relative to the recipe directory
                       (default: the config's train_label, else the utt2spk
                       next to the tar index)
  OVSB_PARITY_CACHE    directory for the dumps (default <tmp>/ovsb_data)
"""

import functools
import hashlib
import io
import json
import os
import random
import resource
import struct
import subprocess
import sys
import tarfile
import tempfile
import time

import numpy as np
import pytest

RECIPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RECIPE_DIR not in sys.path:
    sys.path.insert(0, RECIPE_DIR)

from ovsb.data.speakers import (ArmSpec, CellSpec, PolicySpec,  # noqa: E402
                                SpeakerTable)
from ovsb.data.weights import build_weights  # noqa: E402

SOURCE_TREE = os.environ.get('OVSB_SOURCE_TREE')
SOURCE_RECIPE = os.environ.get('OVSB_SOURCE_RECIPE')
SOURCE_CONFIG = os.environ.get('OVSB_SOURCE_CONFIG', 'conf/s3_base_v9c.yaml')
SOURCE_LABEL = os.environ.get('OVSB_SOURCE_LABEL')
CACHE = os.environ.get('OVSB_PARITY_CACHE') or os.path.join(
    tempfile.gettempdir(), 'ovsb_data')

needs_source = pytest.mark.skipif(
    not (SOURCE_TREE and SOURCE_RECIPE),
    reason='set OVSB_SOURCE_TREE and OVSB_SOURCE_RECIPE for parity tests')

TABLE_ARRAYS = (
    'rows', 'spk_of', 'vid_of', 'spk_ptr', 'spk_nvid', 'spk_names', 'spk_arm',
    'lang_names', 'spk_lang', 'seg_of', 'dev_of', 'dcls_of', 'dial_of',
    'genre_of', 'spk_nseg', 'spk_dial', 'spk_has_std', 'spk_ngenre',
    'ks_phase', 'ks_dial', 'spk_ks_both', 'spk_ks_stddial')
WEIGHT_ARRAYS = ('spk_w', 'arm_weighted', 'spk_share', 'spk_listed')

# ----------------------------------------------------------------------------
# synthetic corpus

SR = 16000
TOY_MIN_LEN = 1000
# 3D-Speaker (device code, distance code): close 0.3 m, close-talk, far 4 m,
# far 1 m, mid 0.8 m
DEVS = [(6, 8), (3, 0), (4, 12), (1, 4), (6, 10)]
TOY_ARMS = {
    'vox2': {'weight': 0.2, 'corpus': 'vox2'},
    'vb2': {'weight': 0.2, 'corpus': 'vb2'},
    'cn2': {'weight': 0.1, 'corpus': 'cn2'},
    'mandarin_t1': {'weight': 0.1, 'corpus': ['a1', 'mgd']},
    '3dspeaker': {'weight': 0.2, 'corpus': ['3ds', '3dspeaker']},
    'kespeech': {'weight': 0.1, 'corpus': 'kespeech'},
    'cn1': {'weight': 0.1, 'corpus': 'cn1'},
}


def wav_bytes(n, rng, sr=SR, channels=1):
    """PCM16 RIFF file with ``n`` frames of noise."""
    x = (rng.standard_normal(n * channels) * 3000).astype('<i2').tobytes()
    head = b'RIFF' + struct.pack('<I', 36 + len(x)) + b'WAVE' + b'fmt ' + \
        struct.pack('<IHHIIHH', 16, 1, channels, sr, sr * 2 * channels,
                    2 * channels, 16)
    return head + b'data' + struct.pack('<I', len(x)) + x


def write_shard(path, members):
    """members: [(key, speaker, wav bytes)] -> tar with <key>.spk/.wav"""
    with tarfile.open(path, 'w') as tf:
        for key, spk, data in members:
            for suffix, blob in (('spk', spk.encode()), ('wav', data)):
                ti = tarfile.TarInfo(key + '.' + suffix)
                ti.size = len(blob)
                tf.addfile(ti, io.BytesIO(blob))


def toy_utts():
    """[(key, speaker)] of the synthetic corpus (see the comments)."""
    utts = []
    for i in range(6):  # VoxCeleb2: 3 videos x 2; keys keep '.wav'
        s = 'id%05d' % i
        utts += [('%s/vid%d/%05d.wav' % (s, v, u), s)
                 for v in range(3) for u in range(2)]
    utts += [('id00900/vid0/%05d.wav' % u, 'id00900') for u in range(3)]
    for i in range(6):  # VoxBlink2: 2 videos x 2
        s = 'vb2_id%04d' % i
        utts += [('%s/v%d/%d' % (s, v, u), s)
                 for v in range(2) for u in range(2)]
    for i in range(8):  # CN-Celeb2: 4 multi-genre, 4 single-genre
        s = 'cn2_id1%04d' % i
        genres = ['interview', 'singing', 'vlog'] if i < 4 else ['vlog']
        utts += [('%s/%s-%02d-%03d' % (s, g, k, u), s) for g in genres
                 for k in range(2 if i < 4 else 3) for u in range(2)]
    for i in range(4):  # CN-Celeb1 (same key format, corpus from the file)
        s = 'cn2_id0%04d' % i
        genres = ['speech', 'drama'] if i < 2 else ['speech']
        utts += [('%s/%s-%02d-%03d' % (s, g, k, u), s) for g in genres
                 for k in range(2) for u in range(2)]

    def add3(spk, n_std, dialects=()):
        segs = [(j, 0) for j in range(n_std)] + \
            [(n_std + j, d) for j, d in enumerate(dialects)]
        out = []
        for j, d in segs:
            for dev, dist in DEVS:
                c = 'Device%02d_Distance%02d_Dialect%02d' % (dev, dist, d)
                out.append(('3ds_%s/%s/%s_%03d_%s' % (spk, c, spk, j + 1, c),
                            '3ds_' + spk))
        return out
    for i in range(6):  # 3D-Speaker Standard speakers, 4 segments
        utts += add3('3D_SPK_%05d' % i, 4)
    utts += add3('3D_SPK_00100', 3, (12, 12, 12))  # 1 speaker of dialect 12
    utts += add3('3D_SPK_00101', 3, (9, 9, 9))  # 2 speakers of dialect 9
    utts += add3('3D_SPK_00102', 3, (9, 9, 7))
    utts += add3('3D_SPK_00200', 1)  # a single segment
    for i in range(7):  # KeSpeech
        s = 'ks_%07d' % (1000000 + i)
        phases = {0: (1, 2), 1: (1, 2), 2: (1, 2), 3: (1, 2), 4: (1, ),
                  5: (1, ), 6: (1, )}[i]
        utts += [('%s/phase%d/%07d_%02d' % (s, ph, 1000000 + i, u), s)
                 for ph in phases for u in range(4)]
    for i in range(4):  # MAGICDATA-style, 2 sessions
        s = 'mgd_14_%04d' % i
        utts += [('%s/s%d/%d' % (s, k, u), s)
                 for k in range(2) for u in range(2)]
    for i in range(4):  # two-field segment keys
        s = 'a1_S%04d' % i
        utts += [('%s/rec%d_%04d_%04d' % (s, k, u * 10, u * 10 + 9), s)
                 for k in range(2) for u in range(2)]
    utts += [('zzz_0001/a/%d' % u, 'zzz_0001') for u in range(2)]  # no arm
    return utts


def toy_subdialect(utts):
    """KeSpeech sub-dialect per key: speakers 0, 1, 4, 5 mix Standard and
    a sub-dialect in phase 1; the others are Standard only"""
    out = {}
    for key, spk in utts:
        if not spk.startswith('ks_'):
            continue
        i = int(spk[3:]) - 1000000
        u = int(key.rsplit('_', 1)[1])
        mixed = i in (0, 1, 4, 5) and '/phase1/' in key and u % 2
        out[key] = ('Beijing' if i % 2 else 'Ji-Lu') if mixed else 'Mandarin'
    return out


def make_toy(root):
    """Write the synthetic corpus under ``root``; returns its description.

    The label file is in a shuffled order and additionally lists one key
    that is not in the index; one indexed utterance is shorter than
    TOY_MIN_LEN.
    """
    root = str(root)
    rng = np.random.default_rng(0)
    utts = toy_utts()
    order = list(utts)
    random.Random(0).shuffle(order)
    short_key = 'id00900/vid0/00002.wav'
    blobs = {}
    shards = []
    os.makedirs(os.path.join(root, 'shards'))
    for k in range(2):
        members = []
        for key, spk in order[k::2]:
            n = 500 if key == short_key else int(rng.integers(1200, 2400))
            blobs[key] = wav_bytes(n, rng)
            members.append((key, spk, blobs[key]))
        rel = os.path.join('shards', 'shard_%d.tar' % k)
        write_shard(os.path.join(root, rel), members)
        shards.append(rel)
    files = {n: os.path.join(root, n) for n in (
        'shard.list', 'utt2spk', 'spk2corpus', 'spk2lang', 'utt2subdialect',
        'spk_weights.txt')}
    with open(files['shard.list'], 'w') as f:
        f.write('\n'.join(shards) + '\n')
    label = list(order)
    random.Random(1).shuffle(label)
    label.insert(5, ('id00000/vid9/absent.wav', 'id00000'))
    with open(files['utt2spk'], 'w') as f:
        f.writelines('%s %s\n' % x for x in label)
    spks = sorted({s for _, s in utts})
    tag = {'cn2_id0': 'cn1', 'ks_': 'kespeech'}
    with open(files['spk2corpus'], 'w') as f:
        for s in spks:
            c = [v for p, v in tag.items() if s.startswith(p)]
            if c:
                f.write('%s %s\n' % (s, c[0]))
    with open(files['spk2lang'], 'w') as f:
        for i in range(5):  # vb2_id0005 has no entry -> 'unk'
            f.write('vb2_id%04d %s\n' % (i, 'zh' if i < 3 else 'ja'))
    with open(files['utt2subdialect'], 'w') as f:
        f.writelines('%s %s\n' % x for x in toy_subdialect(utts).items())
    with open(files['spk_weights.txt'], 'w') as f:
        f.write('mgd_14_0000 3.0\nmgd_14_0001 1.5\nnot_a_speaker 9.0\n')
    episodic = {
        'P': 12, 'K': 2, 'arms': TOY_ARMS, 'index': 'index',
        'spk2corpus': 'spk2corpus', 'spk2lang': 'spk2lang',
        'vox2_lang': 'en', 'stratify': 'systematic', 'log_every': 0,
        'construction': {
            'forbid_same_segment': True,
            'arms': {
                '3dspeaker': {'pair_policy': 'contrast', 'p_dialect': 0.5,
                              'segment_first': True, 'dialect_mass': {
                                  'boost': 3.0, 'cap_x_median': 8}},
                'cn2': {'pair_policy': 'contrast', 'w_bridge': 2.0},
                'cn1': {'pair_policy': 'contrast', 'w_bridge': 2.0},
                'mandarin_t1': {'pair_policy': 'xsession',
                                'spk_weight_file': 'spk_weights.txt',
                                'spk_weight_share': 0.5},
                'kespeech': {'pair_policy': 'ks_contrast', 'p_dialect': 0.5,
                             'subdialect_file': 'utt2subdialect'}}},
        'cells': {'vb2_zh': {'arm': 'vb2', 'share': 0.04,
                             'select': {'spk2lang': 'zh'}}}}
    return dict(root=root, utts=utts, label=label, blobs=blobs, files=files,
                shards=shards, episodic=episodic, short_key=short_key)


# ----------------------------------------------------------------------------
# legacy sampler block -> typed inputs of the port


def specs_from_episodic(ep, base):
    """SOURCE ``episodic_sampler`` dict -> keyword arguments of
    SpeakerTable.from_files (relative paths resolved against ``base``)"""
    def path(p):
        return p if p is None or os.path.isabs(p) else os.path.join(base, p)

    cons = ep.get('construction') or {}
    policies = {}
    for name, pol in (cons.get('arms') or {}).items():
        pol = dict(pol)
        for k in ('subdialect_file', 'spk_weight_file'):
            if pol.get(k):
                pol[k] = path(pol[k])
        policies[name] = PolicySpec(**pol)
    cells = [CellSpec(n, c['arm'], c['share'], c['select']['spk2lang'])
             for n, c in (ep.get('cells') or {}).items()]
    return dict(
        index=path(ep['index']), spk2corpus=path(ep.get('spk2corpus')),
        spk2lang=path(ep.get('spk2lang')),
        arms=[ArmSpec(n, a['weight'], a['corpus'])
              for n, a in ep['arms'].items()],
        cells=cells, policies=policies, vox2_lang=ep.get('vox2_lang', 'en'),
        K=ep.get('K', 2),
        forbid_same_segment=bool(cons.get('forbid_same_segment', False)),
        index_root=base)


def toy_source(toy):
    """SOURCE dump of the synthetic corpus: (arrays, info)."""
    return source_dump('utt2spk', toy['episodic'], toy['root'], TOY_MIN_LEN,
                       seed=3, n_read=20, tag='toy',
                       out_dir=os.path.join(toy['root'], 'source'))


def toy_table(toy):
    """The port's table and weights of the synthetic corpus."""
    kw = specs_from_episodic(toy['episodic'], toy['root'])
    table = SpeakerTable.from_files(toy['files']['utt2spk'],
                                    min_len=TOY_MIN_LEN, **kw)
    return table, build_weights(table)


# ----------------------------------------------------------------------------
# SOURCE dump (runs in a subprocess with the source tree on PYTHONPATH)

SOURCE_DUMP = r'''
import hashlib, json, logging, sys, time, resource
import numpy as np
job = json.load(open(sys.argv[1]))
out = sys.argv[2]
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.utils import spk2id
from wespeaker.dataset.episodic import build_episodic_sampler
from wespeaker.dataset.tar_index import decode_wav

lines = []
class Keep(logging.Handler):
    def emit(self, record):
        lines.append(record.getMessage())
log = logging.getLogger('source_dump')
log.setLevel(logging.INFO)
log.addHandler(Keep())
t0 = time.time()
utt_spk = read_table(job['train_label'])
t1 = time.time()
ep = job['episodic']
S = build_episodic_sampler(ep, utt_spk, ep['P'] * ep['K'],
                           job['sample_num_per_epoch'], job['min_len'],
                           job['seed'], logger=log)
t2 = time.time()
arr = {}
for n in job['table_arrays'] + job['weight_arrays']:
    v = getattr(S, n, None)
    if v is not None:
        arr[n] = np.asarray(v)
arr['static_weights'] = S.static_weights
arr['seg_names'] = np.array(S.seg_names, dtype='S')
arr['genre_names'] = np.array(S.genre_names, dtype='S')
arr['ks_dial_names'] = np.array(getattr(S, 'ks_dial_names', []), dtype='S')
classes = spk2id(utt_spk)
arr['class_names'] = np.array(sorted(classes, key=classes.get))
for j, c in enumerate(S.share_cells):
    arr['cell%d_mask' % j] = c['mask']
# per-cell share rescale of the weighted pools, for a few sampler cells
cells = []
if S.spk_share is not None:
    for a in np.flatnonzero(np.isfinite(S.spk_share)):
        for (epoch, rank, world, wid, nw) in job['share_cells']:
            cell = S.cell_speakers(epoch, rank, world, wid, nw)
            spk = cell[S.spk_arm[cell] == a]
            tag = 'share_%d_%d' % (a, len(cells))
            arr[tag + '_spk'] = spk
            arr[tag + '_w'] = S._share_weights(spk, S.spk_w[spk],
                                               S.spk_share[a])
            cells.append(tag)
# reader
rows = np.random.default_rng(job['read_seed']).choice(
    len(S.index), size=min(job['n_read'], len(S.index)), replace=False)
reads = []
for i in rows.tolist():
    data = S.reader.read(i)
    w, sr = decode_wav(data)
    reads.append([i, hashlib.sha256(data).hexdigest(), len(data),
                  hashlib.sha256(w.numpy().tobytes()).hexdigest(), int(sr)])
np.savez(out + '.npz', **arr)
info = dict(
    arm_names=S.arm_names, n_utts=S.n_utts, n_spk=S.n_spk,
    n_missing=S.n_missing, n_short=S.n_short, cons=S.cons,
    cells_info=S.cells_info, forbid_seg=S.forbid_seg,
    share_cells=[dict(name=c['name'], arm=int(c['arm']), share=c['share'])
                 for c in S.share_cells],
    share_tags=cells, reads=reads, log=lines,
    seconds=dict(read_label=t1 - t0, sampler=t2 - t1),
    maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
    numpy=np.__version__)
json.dump(info, open(out + '.json', 'w'), default=float)
'''


def source_dump(train_label, episodic, cwd, min_len, seed=716,
                sample_num_per_epoch=0, n_read=200, tag='dump',
                out_dir=None):
    """Run the SOURCE sampler construction and return (arrays, info).

    The dump is kept in ``out_dir`` (default CACHE) under a hash of the job,
    the dump script and the label file's size and mtime, and reused.
    """
    out_dir = out_dir or CACHE
    st = os.stat(train_label if os.path.isabs(train_label) else
                 os.path.join(cwd, train_label))
    job = dict(train_label=train_label, episodic=episodic, min_len=min_len,
               seed=seed, sample_num_per_epoch=sample_num_per_epoch,
               n_read=n_read, read_seed=0, table_arrays=list(TABLE_ARRAYS),
               weight_arrays=list(WEIGHT_ARRAYS),
               share_cells=[(1, 0, 8, 0, 12), (1, 3, 8, 5, 12),
                            (2, 7, 8, 11, 12), (1, 0, 1, 0, 1)])
    blob = json.dumps([job, SOURCE_DUMP, cwd, SOURCE_TREE, st.st_size,
                       st.st_mtime_ns, sys.version, np.__version__])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, '%s_%s' % (
        tag, hashlib.sha1(blob.encode()).hexdigest()[:16]))
    if not (os.path.isfile(out + '.npz') and os.path.isfile(out + '.json')):
        with open(out + '.job', 'w') as f:
            json.dump(job, f)
        with open(out + '.py', 'w') as f:
            f.write(SOURCE_DUMP)
        env = dict(os.environ, PYTHONPATH=SOURCE_TREE,
                   PYTHONDONTWRITEBYTECODE='1')
        tmp = out + '.tmp%d' % os.getpid()
        subprocess.run(['nice', sys.executable, out + '.py', out + '.job',
                        tmp], cwd=cwd, env=env, check=True)
        os.replace(tmp + '.npz', out + '.npz')
        os.replace(tmp + '.json', out + '.json')
    with open(out + '.json') as f:
        info = json.load(f)
    return np.load(out + '.npz'), info


# ----------------------------------------------------------------------------
# the real training list of the source recipe


def _source_yaml():
    import yaml
    with open(os.path.join(SOURCE_RECIPE, SOURCE_CONFIG)) as f:
        return yaml.safe_load(f)


def real_job():
    """(train_label, episodic block, min_len, seed, samples per epoch)"""
    cfg = _source_yaml()
    ds = cfg['dataset_args']
    shift = ds['w2vbert_sv_args']['frame_shift']
    min_len = int(shift / 1000 * ds['filter_args']['min_num_frames'] * 16000)
    ep = ds['episodic_sampler']
    # the label file is a command-line argument of the source trainer; it
    # sits next to the tar index
    label = SOURCE_LABEL or cfg.get('train_label') or os.path.join(
        os.path.dirname(ep['index']), 'utt2spk')
    return (label, ep, min_len, cfg.get('seed', 42),
            ds.get('sample_num_per_epoch', 0))


@functools.lru_cache(maxsize=None)
def real_source():
    """SOURCE dump of the real training list: (arrays, info)."""
    label, ep, min_len, seed, n_epoch = real_job()
    return source_dump(label, ep, SOURCE_RECIPE, min_len, seed, n_epoch,
                       tag='real')


@functools.lru_cache(maxsize=None)
def real_port():
    """The port on the real training list: (table, weights, timing)."""
    label, ep, min_len, _seed, _n = real_job()
    kw = specs_from_episodic(ep, SOURCE_RECIPE)
    t0 = time.time()
    table = SpeakerTable.from_files(os.path.join(SOURCE_RECIPE, label),
                                    min_len=min_len, **kw)
    t1 = time.time()
    weights = build_weights(table)
    timing = dict(table_s=round(t1 - t0, 1), weights_s=round(
        time.time() - t1, 2), maxrss_gb=round(resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2))
    print('\n[real list] port timing: %s' % timing)
    return table, weights, timing


def assert_same(a, b, name):
    """Exact equality of two arrays: dtype kind, shape, values (nan = nan)"""
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype.kind == b.dtype.kind, (name, a.dtype, b.dtype)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    equal_nan = a.dtype.kind == 'f'
    assert np.array_equal(a, b, equal_nan=equal_nan), name


def compare_table(table, src, info):
    """Every static table of the port equals the SOURCE sampler's; returns
    the list of compared names"""
    done = []
    for n in TABLE_ARRAYS:
        if n in ('ks_phase', 'ks_dial', 'spk_ks_both', 'spk_ks_stddial') \
                and n not in src.files:
            continue  # the source has no KeSpeech arrays without the policy
        assert_same(getattr(table, n), src[n], n)
        done.append(n)
    assert_same(table.arm_weights, src['static_weights'], 'arm_weights')
    assert_same(table.class_names, src['class_names'], 'class_names')
    assert_same(table.class_names[table.spk_class], src['spk_names'],
                'spk_class')
    assert_same(np.array(table.genre_names, dtype='S'), src['genre_names'],
                'genre_names')
    assert table.n_segments == len(src['seg_names'])
    if 'ks_phase' in src.files:
        assert_same(np.array(table.ks_dial_names, dtype='S'),
                    src['ks_dial_names'], 'ks_dial_names')
    assert list(table.arm_names) == info['arm_names']
    for k in ('n_utts', 'n_spk', 'n_missing', 'n_short'):
        assert getattr(table, k) == info[k], k
    assert table.forbid_same_segment == info['forbid_seg']
    assert len(table.share_cells) == len(info['share_cells'])
    for j, (c, ref) in enumerate(zip(table.share_cells, info['share_cells'])):
        assert (c.name, c.arm, c.share) == (ref['name'], ref['arm'],
                                            ref['share'])
        assert_same(c.mask, src['cell%d_mask' % j], 'cell mask')
    assert table.cells_info == (info['cells_info'] or {})
    return done + ['arm_weights', 'class_names', 'spk_class', 'genre_names',
                   'ks_dial_names', 'share cell masks']


def compare_weights(table, weights, src, info):
    """Speaker weights and the per-cell share rescale equal the SOURCE"""
    from ovsb.data.weights import share_weights
    for n in WEIGHT_ARRAYS:
        v = getattr(weights, n)
        if n not in src.files:
            assert v is None or n == 'arm_weighted', n
            continue
        assert_same(v, src[n], n)
    merged = dict(table.info)
    merged.update(weights.info)
    assert merged == info['cons']
    for tag in info['share_tags']:
        a = int(tag.split('_')[1])
        spk = src[tag + '_spk']
        w = share_weights(weights.spk_w[spk], weights.spk_listed[spk],
                          weights.spk_share[a])
        assert_same(w, src[tag + '_w'], tag)
    return len(info['share_tags'])
