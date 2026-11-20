"""local/data/build_train_list.py on a synthetic three-corpus fixture.

Real check (skipped unless set): ``OVSB_TRAIN_LIST_SITE`` (site file for
conf/data/s3c.yaml) and ``OVSB_TRAIN_LIST_EXPECT`` (SHA256SUMS of the list
it must reproduce); about 1 minute and 4 GB of memory.
"""

import importlib.util
import json
import os

import numpy as np
import pytest
import yaml

RECIPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(RECIPE_DIR, 'local', 'data', 'build_train_list.py')


def _load():
    spec = importlib.util.spec_from_file_location('build_train_list', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


btl = _load()
SEC = 16000


def _index(path, rows, shards, root=None):
    """rows: [(key, shard id, nsamples)] in index order."""
    os.makedirs(path)
    n = len(rows)
    np.save(os.path.join(path, 'keys.npy'),
            np.array([r[0] for r in rows], dtype='S'))
    np.save(os.path.join(path, 'shard.npy'),
            np.array([r[1] for r in rows], np.int32))
    np.save(os.path.join(path, 'offset.npy'),
            np.arange(n, dtype=np.int64) * 1024 + 512)
    np.save(os.path.join(path, 'size.npy'),
            np.array([r[2] * 2 + 44 for r in rows], np.int64))
    np.save(os.path.join(path, 'nsamples.npy'),
            np.array([r[2] for r in rows], np.int64))
    np.save(os.path.join(path, 'sample_rate.npy'),
            np.full(n, SEC, np.int32))
    with open(os.path.join(path, 'shards.txt'), 'w') as f:
        f.write('\n'.join(shards) + '\n')
    if root is not None:
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump(dict(root=root), f)
    return path


def _write(path, text):
    with open(path, 'w') as f:
        f.write(text)
    return str(path)


SPEC = {
    'min_sec': 1.0,
    'min_utts': 2,
    'corpora': [
        {'name': 'base', 'tag': 'vox2', 'tag_prefix': {'vb2_': 'vb2'},
         'filter': False, 'utt2spk': 'base.utt2spk', 'index': 'base.index',
         'keep_shard_paths': True, 'alias': 'base.alias',
         'shard_list': 'base.shard_list'},
        {'name': 'cn', 'tag': 'cn2', 'label_prefix': 'cn2_',
         'utt2spk': 'cn.utt2spk', 'index': 'cn.index',
         'exclude': [{'input': 'cn.heldout'}, {'input': 'cn.audit'}]},
        {'name': 'tvx', 'tag': 'tvx', 'label_prefix': 'tvx_', 'sort': True,
         'utt2spk': 'tvx.utt2spk', 'index': 'tvx.index',
         'shard_list': 'tvx.shard_list',
         'exclude': [{'input': 'tvx.table', 'format': 'tsv', 'column': 'spk',
                      'where': {'source': 'dev'}}]},
    ],
    'identity': {
        'drop': [{'input': 'id.drop'}],
        'merge': [{'input': 'id.merge'},
                  {'input': 'id.tvx_merge', 'format': 'tsv',
                   'duplicate': 'duplicate', 'canonical': 'canonical'}],
    },
    'spk2lang': [
        {'corpus': 'vb2', 'source': 'file', 'input': 'vb2.spk2lang'},
        {'corpus': 'tvx', 'source': 'dominant_key_field', 'field': 1,
         'map': {'zh-CN': 'zh'}},
    ],
    'side_files': [{'out': 'tvx_meta/utt2x', 'input': 'tvx.utt2x',
                    'corpus': 'tvx'}],
    'speaker_weights': {
        'out': 'weights.txt', 'corpora': ['cn2', 'tvx'], 'share': 0.5,
        'table': {'input': 'cn.province', 'speaker_column': 'spk',
                  'column': 'province_raw', 'values': ['he bei']}},
    'must_be_absent': [{'input': 'cn.heldout', 'prefix': 'cn2_'}],
}

BASE_U2S = ('id001/v1/b id001\nid001/v1/a id001\nvb2_a/v/x vb2_b\n'
            'vb2_b/v/y vb2_b\nvb2_c/v/z vb2_c\n')
CN_U2S = ''.join('%s %s\n' % kv for kv in [
    ('cn2_id10/g-01-001', 'id10'), ('cn2_id10/g-01-002', 'cn2_id10'),
    ('cn2_id11/g-01-001', 'id11'), ('cn2_id11/g-01-002', 'id11'),
    ('cn2_id12/g-01-001', 'id12'), ('cn2_id12/g-01-002', 'id12'),
    ('cn2_id13/g-01-001', 'id13'), ('cn2_id13/g-01-002', 'id13'),
    ('cn2_id14/g-01-001', 'id14'), ('cn2_id14/g-01-002', 'id14'),
    ('cn2_id15/g-01-001', 'id15'), ('cn2_id15/g-01-002', 'id15'),
    ('cn2_id16/g-02-001', 'id16'), ('cn2_id16/g-01-001', 'id16')])
TVX_U2S = ''.join('%s %s\n' % kv for kv in [
    ('tvx_s2/en/u1', 'tvx_s2'), ('tvx_s1/zh-CN/u2', 'tvx_s1'),
    ('tvx_s1/en/u1', 'tvx_s1'), ('tvx_s1/zh-CN/u1', 'tvx_s1'),
    ('tvx_s2/de/u1', 'tvx_s2'), ('tvx_s3/de/u1', 'tvx_s3'),
    ('tvx_s3/de/u2', 'tvx_s3'), ('tvx_s9/en/u1', 'tvx_s9'),
    ('tvx_s9/en/u2', 'tvx_s9')])


@pytest.fixture()
def site(tmp_path):
    d = tmp_path / 'in'
    d.mkdir()
    base_rows = [('vb2_c/v/z', 1, SEC), ('id001/v1/a', 0, SEC),
                 ('extra/k/0', 0, SEC), ('id001/v1/b', 0, SEC // 2),
                 ('vb2_a/v/x', 1, SEC), ('vb2_b/v/y', 1, SEC)]
    cn_keys = [line.split()[0] for line in CN_U2S.splitlines()]
    cn_rows = [(k, i % 2, SEC // 2 if k == 'cn2_id12/g-01-001' else 2 * SEC)
               for i, k in enumerate(reversed(cn_keys))
               if k != 'cn2_id14/g-01-002']  # one key is not indexed
    tvx_keys = sorted(line.split()[0] for line in TVX_U2S.splitlines())
    tvx_rows = [(k, 0, 3 * SEC) for k in reversed(tvx_keys)]
    inputs = {
        'base': {
            'utt2spk': _write(d / 'base_u2s', BASE_U2S),
            'index': _index(str(d / 'base_index'), base_rows,
                            ['data/base/s0.tar', 'data/base/s1.tar']),
            'alias': _write(d / 'base_alias', 'vb2_a vb2_b\n'),
            'shard_list': _write(d / 'base_shards',
                                 'data/base/s0.tar\ndata/base/s1.tar\n'),
        },
        'cn': {
            'utt2spk': _write(d / 'cn_u2s', CN_U2S),
            'index': _index(str(d / 'cn_index'), cn_rows,
                            ['shards/c0.tar', '/abs/c1.tar'],
                            root='/corp/cn'),
            'heldout': _write(d / 'cn_heldout', '# dev speakers\nid11\n'),
            'audit': _write(d / 'cn_audit', 'cn2_id13 0.91\n'),
            'province': _write(
                d / 'province.tsv', 'spk\tprovince_raw\n'
                'cn2_id10\the bei\ntvx_s1\tguang dong\nmgd_1\the bei\n'),
        },
        'tvx': {
            'utt2spk': _write(d / 'tvx_u2s', TVX_U2S),
            'index': _index(str(d / 'tvx_index'), tvx_rows,
                            ['/corp/tvx/t0.tar']),
            'shard_list': _write(d / 'tvx_shards', '/corp/tvx/t0.tar\n'),
            'table': _write(d / 'tvx_table.tsv',
                            'spk\tsource\ntvx_s9\tdev\ntvx_s1\ttrain\n'),
            'utt2x': _write(d / 'tvx_utt2x', ''.join(
                '%s x%d\n' % (k, i) for i, k in enumerate(tvx_keys))),
        },
        'id': {
            'drop': _write(d / 'drop', 'cn2_id15\n'),
            'merge': _write(d / 'merge', 'cn2_id16 cn2_id10\n'),
            'tvx_merge': _write(d / 'tvx_merge.tsv',
                                'duplicate\tcanonical\tcos\n'
                                'tvx_s3\ttvx_s2\t0.9\n'),
        },
        'vb2': {'spk2lang': _write(d / 'spk2lang', 'vb2_b es\nid001 en\n')},
    }
    return {'index_root': '/recipe', 'inputs': inputs}


def _build(tmp_path, site, spec=None, name='out'):
    out = str(tmp_path / name)
    summary = btl.build(spec or SPEC, btl.Inputs(site), out)
    return out, summary


def _read(out, rel):
    with open(os.path.join(out, rel)) as f:
        return f.read()


def test_rows_order_and_labels(tmp_path, site):
    out, summary = _build(tmp_path, site)
    assert _read(out, 'utt2spk') == (
        BASE_U2S +                       # base: as it is, file order
        'cn2_id10/g-01-001 cn2_id10\n'   # label prefixed
        'cn2_id10/g-01-002 cn2_id10\n'
        'cn2_id16/g-02-001 cn2_id10\n'   # merged: relabelled, key kept
        'cn2_id16/g-01-001 cn2_id10\n'   # file order (sort: false)
        'tvx_s1/en/u1 tvx_s1\n'          # sort: true -> by key
        'tvx_s1/zh-CN/u1 tvx_s1\n'
        'tvx_s1/zh-CN/u2 tvx_s1\n'
        'tvx_s2/de/u1 tvx_s2\n'
        'tvx_s2/en/u1 tvx_s2\n'
        'tvx_s3/de/u1 tvx_s2\n'
        'tvx_s3/de/u2 tvx_s2\n')
    assert summary['total_utts'] == 16 and summary['total_speakers'] == 6
    cn = summary['corpora']['cn']
    assert cn['rows_excluded'] == 4          # id11 held out, id13 audit
    assert cn['rows_short_or_unindexed'] == 2  # one short, one not indexed
    assert cn['speakers_below_min_utts'] == 2  # id12, id14
    assert summary['corpora']['tvx']['rows_excluded'] == 2  # tsv 'where'


def test_speaker_tables(tmp_path, site):
    out, _ = _build(tmp_path, site)
    assert _read(out, 'spk2corpus') == (
        'cn2_id10 cn2\nid001 vox2\ntvx_s1 tvx\ntvx_s2 tvx\n'
        'vb2_b vb2\nvb2_c vb2\n')
    # vb2 rows from the file (vb2_c has none), then the dominant language
    # of each tvx speaker (merged speaker: de 3 > en 1), mapped
    assert _read(out, 'spk2lang') == 'vb2_b es\ntvx_s1 zh\ntvx_s2 de\n'
    assert _read(out, 'spk_alias') == (
        'vb2_a vb2_b\ncn2_id16 cn2_id10\ntvx_s3 tvx_s2\n')
    assert _read(out, 'shard.list') == (
        'data/base/s0.tar\ndata/base/s1.tar\n/corp/tvx/t0.tar\n')
    # w = share / (1 - share) * (3 - 1) / 1
    assert _read(out, 'weights.txt') == 'cn2_id10 2.000000\n'
    assert _read(out, 'tvx_meta/utt2x') == ''.join(
        '%s x%d\n' % (k, i) for i, k in enumerate(sorted(
            line.split()[0] for line in TVX_U2S.splitlines()))
        if not k.startswith('tvx_s9'))


def test_merged_index(tmp_path, site):
    out, _ = _build(tmp_path, site)
    index = os.path.join(out, 'tar_index')
    keys = [k.decode() for k in np.load(os.path.join(index, 'keys.npy'))]
    # per corpus the order of ITS index, corpora in spec order
    assert keys == [
        'vb2_c/v/z', 'id001/v1/a', 'id001/v1/b', 'vb2_a/v/x', 'vb2_b/v/y',
        'cn2_id16/g-01-001', 'cn2_id16/g-02-001', 'cn2_id10/g-01-002',
        'cn2_id10/g-01-001',
        'tvx_s3/de/u2', 'tvx_s3/de/u1', 'tvx_s2/en/u1', 'tvx_s2/de/u1',
        'tvx_s1/zh-CN/u2', 'tvx_s1/zh-CN/u1', 'tvx_s1/en/u1']
    assert _read(out, 'tar_index/shards.txt').split() == [
        'data/base/s0.tar', 'data/base/s1.tar',  # kept as written
        '/corp/cn/shards/c0.tar', '/abs/c1.tar',  # made absolute
        '/corp/tvx/t0.tar']
    shard = np.load(os.path.join(index, 'shard.npy'))
    assert shard.dtype == np.int32
    assert shard[:5].tolist() == [1, 0, 0, 1, 1]
    assert set(shard[5:9].tolist()) <= {2, 3} and set(shard[9:]) == {4}
    ns = np.load(os.path.join(index, 'nsamples.npy'))
    assert ns[2] == SEC // 2  # base rows are not length-filtered
    with open(os.path.join(index, 'meta.json')) as f:
        assert json.load(f)['root'] == '/recipe'
    assert set(_read(out, 'utt2spk').split()[0::2]) == set(keys)


def test_class_order_and_sums(tmp_path, site):
    out, summary = _build(tmp_path, site)
    with open(os.path.join(out, 'class_order.json')) as f:
        order = json.load(f)
    classes = ['cn2_id10', 'id001', 'tvx_s1', 'tvx_s2', 'vb2_b', 'vb2_c']
    assert order['num_classes'] == 6
    assert order['spk_sha256'] == btl.speaker_digest(classes)
    from ovsb.model.head_record import speaker_list_digest
    assert order['spk_sha256'] == speaker_list_digest(classes)
    assert order['by_corpus']['tvx'] == dict(
        num_classes=2, first='tvx_s1', first_class_id=2, last='tvx_s2',
        last_class_id=3, contiguous_block=True)
    sums = btl.write_sums(out)
    assert list(sums)[:3] == ['build_summary.json', 'class_order.json',
                              'shard.list']
    assert sums['utt2spk'] == summary['utt2spk_sha256'] == \
        btl.sha_file(os.path.join(out, 'utt2spk'))
    lines = _read(out, 'SHA256SUMS').splitlines()
    assert lines[0] == '%s  build_summary.json' % sums['build_summary.json']
    assert 'tar_index/keys.npy' in sums and 'SHA256SUMS' not in sums


def test_build_is_deterministic(tmp_path, site):
    a, _ = _build(tmp_path, site, name='a')
    b, _ = _build(tmp_path, site, name='b')
    sa, sb = btl.write_sums(a), btl.write_sums(b)
    differing = {k for k in sa if sa[k] != sb[k]}
    assert differing <= {'build_summary.json', 'tar_index/meta.json'}


def _spec(**changes):
    spec = json.loads(json.dumps(SPEC))
    spec.update(changes)
    return spec


def test_heldout_speaker_in_list_fails(tmp_path, site):
    # id.merge lists cn2_id16 first: a merged duplicate is not a label
    spec = _spec(must_be_absent=[{'input': 'id.merge'}])
    _build(tmp_path, site, spec, 'ok')
    site['inputs']['id']['leak'] = _write(tmp_path / 'leak', 'tvx_s1\n')
    spec = _spec(must_be_absent=[{'input': 'id.leak'}])
    with pytest.raises(btl.SpecError, match='held-out'):
        _build(tmp_path, site, spec, 'bad')


def test_identity_tables_must_match_the_data(tmp_path, site):
    site['inputs']['id']['drop'] = _write(tmp_path / 'drop', 'cn2_id99\n')
    with pytest.raises(btl.SpecError, match='in no corpus'):
        _build(tmp_path, site)


def test_drop_and_merge_conflict(tmp_path, site):
    site['inputs']['id']['drop'] = _write(tmp_path / 'drop', 'cn2_id10\n')
    with pytest.raises(btl.SpecError, match='both dropped and merged'):
        _build(tmp_path, site)


def test_label_clash_between_corpora(tmp_path, site):
    spec = _spec()
    spec['corpora'][2]['label_prefix'] = ''
    site['inputs']['tvx']['utt2spk'] = _write(
        tmp_path / 'clash', 'tvx_s1/en/u1 id001\ntvx_s1/zh-CN/u1 id001\n')
    spec['identity'] = {}
    spec['side_files'] = []
    with pytest.raises(btl.SpecError, match='already used'):
        _build(tmp_path, site, spec)


def test_missing_site_entry(tmp_path, site):
    del site['inputs']['cn']['audit']
    with pytest.raises(btl.SpecError, match="'cn.audit'"):
        _build(tmp_path, site)


def test_main_expect_and_freeze(tmp_path, site):
    spec_file = _write(tmp_path / 'spec.yaml', yaml.safe_dump(SPEC))
    site_file = _write(tmp_path / 'site.yaml', yaml.safe_dump(site))
    out = str(tmp_path / 'out')
    args = ['--spec', spec_file, '--site', site_file, '--out', out]
    assert btl.main(args) == 0
    assert btl.main(args) == 2  # a finished build is kept
    expect = _write(tmp_path / 'EXPECT', _read(out, 'SHA256SUMS'))
    other = str(tmp_path / 'other')
    args2 = ['--spec', spec_file, '--site', site_file, '--out', other,
             '--expect', expect]
    assert btl.main(args2) == 0
    with open(expect) as f:
        text = f.read()
    sha = btl.sha_file(os.path.join(out, 'utt2spk'))
    _write(expect, text.replace(sha, '0' * 64))
    assert btl.main(args2 + ['--force']) == 1


def test_spec_file_has_no_paths():
    with open(os.path.join(RECIPE_DIR, 'conf', 'data', 's3c.yaml')) as f:
        text = f.read()
    spec = yaml.safe_load(text)
    assert [c['name'] for c in spec['corpora']] == [
        'base', 'cn2', 'mgd', 'ramc', 'a1', 'a3', '3ds', 'kespeech', 'tvx',
        'cn1']
    assert not any(len(tok) > 1 and tok.startswith('/')
                   for tok in text.split())
    with open(os.path.join(RECIPE_DIR, 'conf', 'data',
                           'site_paths.example.yaml')) as f:
        example = btl.Inputs(yaml.safe_load(f))

    def names(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ('input', 'utt2spk', 'index', 'alias',
                           'shard_list') and isinstance(value, str):
                    yield value
                else:
                    yield from names(value)
        elif isinstance(node, list):
            for item in node:
                yield from names(item)
    used = set(names(spec))
    assert used and used == set(example.flat)  # example covers the spec


@pytest.mark.skipif(not all(os.environ.get(k) for k in (
    'OVSB_TRAIN_LIST_SITE', 'OVSB_TRAIN_LIST_EXPECT')),
    reason='OVSB_TRAIN_LIST_SITE / OVSB_TRAIN_LIST_EXPECT not set')
def test_real_list_is_byte_identical(tmp_path):
    out = str(tmp_path / 'list')
    assert btl.main([
        '--spec', os.path.join(RECIPE_DIR, 'conf', 'data', 's3c.yaml'),
        '--site', os.environ['OVSB_TRAIN_LIST_SITE'], '--out', out,
        '--expect', os.environ['OVSB_TRAIN_LIST_EXPECT']]) == 0
