#!/usr/bin/env python
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
"""Build the training-list directory from per-corpus inputs.

    . ./path.sh
    python local/data/build_train_list.py --spec conf/data/s3c.yaml \\
        --site site_paths.yaml --out data/train [--expect SHA256SUMS]

The spec (``conf/data/*.yaml``) states the rules and names every input by a
logical name (``kespeech.utt2spk``); the site file maps those names to paths
(``conf/data/site_paths.example.yaml``). The spec holds no paths.

Rules, in this order (the line order of ``utt2spk`` is part of the result:
the sampler sorts it with a stable sort):

1. Corpora are processed in the order of the spec. For each one the rows of
   its ``utt2spk`` are read in file order; a label gets ``label_prefix``
   unless it already starts with it.
2. Rows of excluded speakers (``exclude``: held-out and audit lists) are
   removed, then rows whose key is not in the corpus tar index or is
   shorter than ``min_sec``, then speakers left with fewer than ``min_utts``
   rows. A corpus with ``filter: false`` skips this step.
3. ``sort: true`` sorts the remaining rows of the corpus by key.
4. Identity decisions: rows of ``identity.drop`` speakers are removed and
   ``identity.merge`` duplicates are relabelled to their canonical speaker
   (keys are unchanged).
5. The merged tar index holds, corpus after corpus, the rows of the kept
   keys in the order of the corpus index, and every shard path of it.

Outputs in ``--out``::

    utt2spk  spk2corpus  spk2lang  spk_alias  shard.list  tar_index/
    <speaker weight file>  <filtered side files>  class_order.json
    build_summary.json  SHA256SUMS

``--expect`` compares the sha256 of every output with a ``SHA256SUMS`` file
and prints one line per file; the exit code is 1 when a file other than
those named with ``--expect_ignore`` differs.
"""

import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import yaml

from ovsb.data.index import IndexPart, merge_index

IGNORE_DEFAULT = ('tar_index/meta.json', 'class_order.json',
                  'build_summary.json')


class SpecError(ValueError):
    """The spec or the site file is malformed or inconsistent."""


def log(*msg):
    print('[build_train_list]', *msg, file=sys.stderr, flush=True)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def speaker_digest(speakers):
    """sha256 of the sorted speaker ids, one per line (head sidecar rule)."""
    return hashlib.sha256('\n'.join(sorted(set(speakers))).encode()) \
        .hexdigest()


def read_pairs(path):
    """First two columns of every line with at least two columns."""
    out = []
    with open(path) as f:
        for line in f:
            cols = line.split()
            if len(cols) >= 2:
                out.append((cols[0], cols[1]))
    return out


class Inputs:
    """Logical input names -> paths of the site file."""

    def __init__(self, site):
        self.flat = {}
        self._flatten('', site.get('inputs') or {})
        self.index_root = site.get('index_root')
        self.used = set()

    def _flatten(self, prefix, node):
        for key, value in node.items():
            name = prefix + str(key)
            if isinstance(value, dict):
                self._flatten(name + '.', value)
            else:
                self.flat[name] = value

    def path(self, name):
        if name not in self.flat or not self.flat[name]:
            raise SpecError('input %r is not in the site file' % name)
        self.used.add(name)
        return str(self.flat[name])


def read_ids(inputs, entry, prefix=''):
    """Speaker ids of one list entry of the spec.

    ``{input: N}``: first column of a text file ('#' lines skipped).
    ``{input: N, format: tsv, column: C, where: {col: value}}``: column C of
    a tab-separated table with a header, restricted to the matching rows.
    Ids get ``prefix`` unless they already start with it.
    """
    path = inputs.path(entry['input'])
    ids = []
    if entry.get('format', 'ids') == 'tsv':
        where = entry.get('where') or {}
        with open(path) as f:
            for row in csv.DictReader(f, delimiter='\t'):
                if all(row[k] == str(v) for k, v in where.items()):
                    ids.append(row[entry['column']])
    else:
        with open(path, errors='replace') as f:
            for line in f:
                if line.strip() and not line.lstrip().startswith('#'):
                    ids.append(line.split()[0])
    return [i if i.startswith(prefix) else prefix + i for i in ids]


def read_merges(inputs, entry):
    """(duplicate, canonical) pairs of one ``identity.merge`` entry."""
    path = inputs.path(entry['input'])
    if entry.get('format', 'pairs') == 'tsv':
        with open(path) as f:
            return [(r[entry['duplicate']], r[entry['canonical']])
                    for r in csv.DictReader(f, delimiter='\t')]
    return read_pairs(path)


def index_lengths(index_dir):
    """key -> nsamples of an index, and its nominal sample rate."""
    keys = np.load(os.path.join(index_dir, 'keys.npy'), mmap_mode='r')
    ns = np.load(os.path.join(index_dir, 'nsamples.npy'), mmap_mode='r')
    sr = np.load(os.path.join(index_dir, 'sample_rate.npy'), mmap_mode='r')
    rate = int(np.median(sr[:1000]))
    return dict(zip((k.decode() for k in keys.tolist()), ns.tolist())), rate


def corpus_tag(corpus, label):
    """Corpus tag of a label: ``tag`` or the first matching ``tag_prefix``."""
    for prefix, tag in (corpus.get('tag_prefix') or {}).items():
        if label.startswith(prefix):
            return tag
    return corpus['tag']


def select_rows(corpus, inputs, min_sec, min_utts):
    """Steps 1-3 for one corpus: the kept (key, label) rows and counters."""
    prefix = corpus.get('label_prefix') or ''
    rows = [(u, s if s.startswith(prefix) else prefix + s)
            for u, s in read_pairs(inputs.path(corpus['utt2spk']))]
    stat = collections.OrderedDict(rows_in=len(rows))
    if corpus.get('filter', True):
        excluded = set()
        for entry in corpus.get('exclude') or []:
            excluded.update(read_ids(inputs, entry, prefix))
        nsamples, rate = index_lengths(inputs.path(corpus['index']))
        kept, n_excl, n_short = [], 0, 0
        for u, s in rows:
            if s in excluded:
                n_excl += 1
            elif u not in nsamples or nsamples[u] < min_sec * rate:
                n_short += 1
            else:
                kept.append((u, s))
        count = collections.Counter(s for _, s in kept)
        rows = [(u, s) for u, s in kept if count[s] >= min_utts]
        stat.update(excluded_ids=len(excluded), rows_excluded=n_excl,
                    rows_short_or_unindexed=n_short,
                    speakers_below_min_utts=sum(
                        1 for c in count.values() if c < min_utts))
        if excluded & {s for _, s in rows}:
            raise SpecError('%s: excluded speaker kept' % corpus['name'])
    if corpus.get('sort', False):
        rows.sort()
    return rows, stat


def province_weights(spec, inputs, spk2corpus, out):
    """Weight file: listed speakers get ``share`` of the uniform draws.

    w = share / (1 - share) * (n_all - n_listed) / n_listed, the same value
    for every listed speaker; the other speakers of ``corpora`` weigh 1.
    """
    pool = {s for s, c in spk2corpus.items() if c in set(spec['corpora'])}
    table = spec['table']
    values = set(table['values'])
    with open(inputs.path(table['input'])) as f:
        listed = sorted(r[table['speaker_column']]
                        for r in csv.DictReader(f, delimiter='\t')
                        if r[table['speaker_column']] in pool
                        and r[table['column']] in values)
    if not listed or len(listed) == len(pool):
        raise SpecError('speaker_weights: %d of %d speakers listed' %
                        (len(listed), len(pool)))
    share = float(spec['share'])
    w = share / (1 - share) * (len(pool) - len(listed)) / len(listed)
    with open(os.path.join(out, spec['out']), 'w') as f:
        f.writelines('%s %.6f\n' % (s, w) for s in listed)
    return dict(speakers=len(pool), listed=len(listed), weight=round(w, 6))


def write_sums(out):
    """SHA256SUMS of every regular file below ``out`` (relative paths)."""
    sums = []
    for d, _, files in sorted(os.walk(out)):
        for name in sorted(files):
            path = os.path.join(d, name)
            if name == 'SHA256SUMS' or os.path.islink(path):
                continue
            sums.append('%s  %s' % (sha_file(path),
                                    os.path.relpath(path, out)))
    with open(os.path.join(out, 'SHA256SUMS'), 'w') as f:
        f.write('\n'.join(sums) + '\n')
    return collections.OrderedDict(
        (line.split('  ', 1)[1], line.split('  ', 1)[0]) for line in sums)


def compare_sums(mine, expect_file, ignore):
    """Print one line per output; return the names that must not differ."""
    with open(expect_file) as f:
        expect = dict((line.split('  ', 1)[1].strip(), line.split('  ', 1)[0])
                      for line in f if line.strip())
    bad = []
    for rel, sha in mine.items():
        if rel not in expect:
            state = 'not-in-expected'
        elif expect[rel] == sha:
            state = 'identical'
        else:
            state = 'DIFFERENT'
            if rel not in ignore:
                bad.append(rel)
        print('%-16s %s%s' % (state, rel, '  (ignored)' if
                              state == 'DIFFERENT' and rel in ignore else ''))
    for rel in expect:
        if rel not in mine:
            print('%-16s %s' % ('not-built', rel))
    return bad


def build(spec, inputs, out):
    """Write the list directory ``out``; returns the summary dict."""
    t0 = time.time()
    min_sec = float(spec.get('min_sec', 1.0))
    min_utts = int(spec.get('min_utts', 2))
    identity = spec.get('identity') or {}
    drops = set()
    for entry in identity.get('drop') or []:
        drops.update(read_ids(inputs, entry))
    merges = []
    for entry in identity.get('merge') or []:
        merges += read_merges(inputs, entry)
    canonical_of = dict(merges)
    if len(canonical_of) != len(merges):
        raise SpecError('identity.merge: a duplicate id is listed twice')
    if set(canonical_of) & set(canonical_of.values()):
        raise SpecError('identity.merge: chained merges')
    if drops & (set(canonical_of) | set(canonical_of.values())):
        raise SpecError('identity: a speaker is both dropped and merged')

    os.makedirs(out, exist_ok=True)
    spk2corpus, aliases, shard_lines = {}, [], []
    parts, per_corpus = [], collections.OrderedDict()
    keys_of = {}
    seen_drop, seen_merge = set(), set()
    n_rows = 0
    with open(os.path.join(out, 'utt2spk'), 'w') as f_u2s:
        for corpus in spec['corpora']:
            name = corpus['name']
            rows, stat = select_rows(corpus, inputs, min_sec, min_utts)
            kept = []
            for u, s in rows:
                if s in drops:
                    seen_drop.add(s)
                    continue
                if s in canonical_of:
                    seen_merge.add(s)
                    s = canonical_of[s]
                kept.append((u, s))
            del rows
            speakers = {s for _, s in kept}
            clash = speakers & spk2corpus.keys()
            if clash:
                raise SpecError('%s: %d labels already used by an earlier '
                                'corpus (e.g. %s)' % (name, len(clash),
                                                      sorted(clash)[0]))
            for s in speakers:
                spk2corpus[s] = corpus_tag(corpus, s)
            f_u2s.writelines('%s %s\n' % row for row in kept)
            keys = {u for u, _ in kept}
            if len(keys) != len(kept):
                raise SpecError('%s: duplicate utterance keys' % name)
            keys_of[name] = keys
            parts.append(IndexPart(inputs.path(corpus['index']), keys,
                                   bool(corpus.get('keep_shard_paths')),
                                   name))
            if corpus.get('alias'):
                aliases += read_pairs(inputs.path(corpus['alias']))
            if corpus.get('shard_list'):
                with open(inputs.path(corpus['shard_list'])) as f:
                    shard_lines += [x.strip() for x in f if x.strip()]
            stat.update(rows=len(kept), speakers=len(speakers))
            per_corpus[name] = stat
            n_rows += len(kept)
            log('%s: %d speakers, %d rows' % (name, len(speakers),
                                              len(kept)))
            del kept
    missing = (drops - seen_drop) | (set(canonical_of) - seen_merge)
    if missing:
        raise SpecError('identity: %d listed speakers are in no corpus '
                        '(e.g. %s)' % (len(missing), sorted(missing)[0]))
    lost = set(canonical_of.values()) - spk2corpus.keys()
    if lost:
        raise SpecError('identity.merge: canonical speaker not in the list '
                        '(e.g. %s)' % sorted(lost)[0])

    # ---- merged tar index
    index_dir = os.path.join(out, 'tar_index')
    merge_index(parts, index_dir, inputs.index_root,
                'merged by local/data/build_train_list.py')
    del parts
    all_keys = np.load(os.path.join(index_dir, 'keys.npy'), mmap_mode='r')
    if len(all_keys) != n_rows or len(np.unique(all_keys)) != n_rows:
        raise SpecError('index rows %d / unique keys != utt2spk rows %d' %
                        (len(all_keys), n_rows))
    del all_keys

    # ---- speaker tables
    with open(os.path.join(out, 'spk2corpus'), 'w') as f:
        f.writelines('%s %s\n' % kv for kv in sorted(spk2corpus.items()))
    aliases += merges
    with open(os.path.join(out, 'spk_alias'), 'w') as f:
        f.writelines('%s %s\n' % kv for kv in aliases)
    if {d for d, _ in aliases} & spk2corpus.keys():
        raise SpecError('spk_alias: a duplicate id is a training label')
    with open(os.path.join(out, 'shard.list'), 'w') as f:
        f.writelines(x + '\n' for x in shard_lines)

    lang_stat = collections.OrderedDict()
    with open(os.path.join(out, 'spk2lang'), 'w') as f:
        for rule in spec.get('spk2lang') or []:
            tag = rule['corpus']
            speakers = sorted(s for s, c in spk2corpus.items() if c == tag)
            if rule['source'] == 'file':
                lang = dict(read_pairs(inputs.path(rule['input'])))
            elif rule['source'] == 'dominant_key_field':
                lang = dominant_language(rule, inputs, spec, keys_of,
                                         canonical_of)
            else:
                raise SpecError('spk2lang: unknown source %r' %
                                rule['source'])
            rows = [(s, lang[s]) for s in speakers if s in lang]
            f.writelines('%s %s\n' % kv for kv in rows)
            lang_stat[tag] = dict(speakers=len(speakers), tagged=len(rows))

    # ---- side files
    for side in spec.get('side_files') or []:
        keys = keys_of[side['corpus']]
        dst = os.path.join(out, side['out'])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        n = 0
        with open(inputs.path(side['input'])) as fi, open(dst, 'w') as fo:
            for line in fi:
                if line.split(None, 1)[0] in keys:
                    fo.write(line)
                    n += 1
        if n != len(keys):
            raise SpecError('%s: %d of %d keys of %s present' % (
                side['out'], n, len(keys), side['corpus']))
    weights = None
    if spec.get('speaker_weights'):
        weights = province_weights(spec['speaker_weights'], inputs,
                                   spk2corpus, out)

    # ---- held-out speakers must be absent
    absent = collections.OrderedDict()
    for entry in spec.get('must_be_absent') or []:
        ids = set(read_ids(inputs, entry, entry.get('prefix', '')))
        leak = ids & spk2corpus.keys()
        absent[entry['input']] = dict(ids=len(ids), in_list=len(leak))
        if leak:
            raise SpecError('%s: %d held-out speakers are in the list '
                            '(e.g. %s)' % (entry['input'], len(leak),
                                           sorted(leak)[0]))

    # ---- class order
    classes = sorted(spk2corpus)
    blocks = collections.OrderedDict()
    for i, s in enumerate(classes):
        b = blocks.setdefault(spk2corpus[s], dict(
            num_classes=0, first=s, first_class_id=i))
        b['num_classes'] += 1
        b['last'], b['last_class_id'] = s, i
    for b in blocks.values():
        b['contiguous_block'] = (b['last_class_id'] - b['first_class_id']
                                 + 1 == b['num_classes'])
    class_order = dict(
        rule='class id = position of the label in sorted(set(labels)) '
             '(Python str order); a classifier head stores '
             'rows_per_class rows per class in this order',
        num_classes=len(classes), spk_sha256=speaker_digest(classes),
        by_corpus=blocks)
    with open(os.path.join(out, 'class_order.json'), 'w') as f:
        json.dump(class_order, f, indent=1)

    by_corpus = collections.Counter(spk2corpus.values())
    summary = dict(
        total_utts=n_rows, total_speakers=len(spk2corpus),
        speakers_by_corpus=dict(sorted(by_corpus.items())),
        corpora=per_corpus, identity=dict(drops=len(drops),
                                          merges=len(merges)),
        spk_alias_lines=len(aliases), spk2lang=lang_stat,
        speaker_weights=weights, must_be_absent=absent,
        shard_list_lines=len(shard_lines),
        utt2spk_sha256=sha_file(os.path.join(out, 'utt2spk')),
        num_classes=len(classes), spk_sha256=class_order['spk_sha256'],
        inputs_used=sorted(inputs.used), seconds=round(time.time() - t0, 1))
    with open(os.path.join(out, 'build_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    return summary


def dominant_language(rule, inputs, spec, keys_of, canonical_of):
    """speaker -> language with most kept utterances (ties alphabetical).

    The language of an utterance is field ``field`` of its '/'-separated
    key; the speaker is the final (merged) label of the row.
    """
    corpus = next(c for c in spec['corpora'] if c['tag'] == rule['corpus'])
    prefix = corpus.get('label_prefix') or ''
    keys = keys_of[corpus['name']]
    field = int(rule.get('field', 1))
    count = collections.defaultdict(collections.Counter)
    for u, s in read_pairs(inputs.path(corpus['utt2spk'])):
        if u in keys:
            s = s if s.startswith(prefix) else prefix + s
            count[canonical_of.get(s, s)][u.split('/')[field]] += 1
    rename = rule.get('map') or {}
    out = {}
    for s, c in count.items():
        lang = min(c, key=lambda x: (-c[x], x))
        out[s] = rename.get(lang, lang)
    return out


def get_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--spec', required=True, help='conf/data/<list>.yaml')
    ap.add_argument('--site', required=True,
                    help='site file: logical input names -> paths')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--force', action='store_true',
                    help='rebuild into a directory that has SHA256SUMS')
    ap.add_argument('--expect', default=None,
                    help='SHA256SUMS file to compare the outputs with')
    ap.add_argument('--expect_ignore', nargs='*',
                    default=list(IGNORE_DEFAULT),
                    help='outputs allowed to differ from --expect '
                    '(default: %(default)s)')
    return ap.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    if os.path.isfile(os.path.join(args.out, 'SHA256SUMS')) and \
            not args.force:
        log('%s/SHA256SUMS exists: a finished build is kept (--force to '
            'rebuild)' % args.out)
        return 2
    with open(args.spec) as f:
        spec = yaml.safe_load(f)
    with open(args.site) as f:
        inputs = Inputs(yaml.safe_load(f) or {})
    summary = build(spec, inputs, args.out)
    sums = write_sums(args.out)
    log(json.dumps({k: summary[k] for k in (
        'total_utts', 'total_speakers', 'speakers_by_corpus',
        'utt2spk_sha256', 'seconds')}))
    if args.expect:
        bad = compare_sums(sums, args.expect, set(args.expect_ignore))
        if bad:
            log('outputs differing from %s: %s' %
                (args.expect, ', '.join(bad)))
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
