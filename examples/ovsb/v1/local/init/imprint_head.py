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
"""Build a classifier head for a new class list from an existing head.

A stage-1 head has rows for the speakers stage 1 was trained on. The
training list of stage 2 has other speakers too; their rows are set
("imprinted", Qi et al. 2018) from embeddings of the stage-1 model, so
that no second stage 1 is needed.

    imprint_head.py list     which speakers have no row + their audio
    imprint_head.py imprint  write the checkpoint with the new head
    imprint_head.py selftest synthetic end-to-end check (CPU, seconds)

1. ``list``: for every speaker of ``--utt2spk`` that is not in
   ``--old_utt2spk``, up to ``--per_spk`` utterances (a fixed pseudo-random
   order, round-robin over sessions), their first ``--max_sec`` seconds
   written as 16 kHz wav files read through the tar index. Writes
   ``<out>/{wav/, raw.list, utt2spk, SUMMARY.json}``.
2. Embeddings of ``raw.list`` with the model that starts stage 2
   (``local/init/imprint_extract.sh``): the new rows must live in the
   embedding space of that model.
3. ``imprint``: head layout = ``K`` rows per class, class c owns rows
   c*K .. c*K+K-1, classes = sorted speaker labels. A class of the old list
   keeps its K rows bit for bit (classes that left the list lose theirs). A
   new class gets w = normalize(mean_i normalize(e_i)) * (mean L2 norm of
   the old rows), the same row K times. Every other tensor is copied
   unchanged. Writes ``<out>``, ``<out>.head.json`` (the record the trainer
   checks), ``<out>.json`` (counts and diagnostics) and ``<out>.dup.tsv``
   (new classes whose row is closer than ``--dup_cos`` to an old class: a
   possible duplicate identity across corpora, to be reviewed).
"""

import argparse
import collections
import hashlib
import io
import json
import os
import sys
import tempfile
import time

import numpy as np

SAMPLE_RATE = 16000
HEAD_KEY = 'projection.weight'


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def labels_sorted(path):
    """Sorted speaker labels (second column) of an utt2spk file."""
    labels = set()
    with open(path) as f:
        for line in f:
            cols = line.split()
            if len(cols) >= 2:
                labels.add(cols[1])
    return sorted(labels)


def session_of(key):
    """Session of an utterance key, for the round-robin of ``list``.

    '<spk>/<session>/<utt>' -> '<spk>/<session>'; '<spk>/<a>-<b>-<c>' ->
    '<spk>/<a>-<b>' (genre and session number).
    """
    head, tail = key.rsplit('/', 1)
    if '/' in head:
        return head
    return head + '/' + '-'.join(tail.split('-')[:2])


def select_utts(utts, per_spk):
    """Up to ``per_spk`` utterances of one speaker, spread over sessions."""
    sessions = collections.defaultdict(list)
    for u in sorted(utts, key=lambda u: sha('imprint|' + u)):
        sessions[session_of(u)].append(u)
    queues = [list(v) for _, v in
              sorted(sessions.items(), key=lambda kv: sha(kv[0]))]
    order = []
    while queues and len(order) < per_spk:
        for q in queues:
            if q and len(order) < per_spk:
                order.append(q.pop(0))
        queues = [q for q in queues if q]
    return order


# ----------------------------------------------------------------------------
# list


def cmd_list(args):
    from scipy.io import wavfile
    from ovsb.data.index import TarIndex, TarReader
    t0 = time.time()
    old = set(labels_sorted(args.old_utt2spk))
    by_spk = collections.defaultdict(list)
    with open(args.utt2spk) as f:
        for line in f:
            u, s = line.split()[:2]
            if s not in old:
                by_spk[s].append(u)
    speakers = sorted(by_spk)[:args.limit_spk or None]
    selected = [(u, s) for s in speakers
                for u in select_utts(by_spk[s], args.per_spk)]
    index = TarIndex(args.index, root=args.index_root)
    want = {u.encode() for u, _ in selected}
    row = {k.decode(): i for i, k in enumerate(index.keys.tolist())
           if k in want}
    missing = len(want) - len(row)
    if missing:
        raise SystemExit('%d selected keys are not in %s' %
                         (missing, args.index))
    reader = TarReader(index)
    os.makedirs(os.path.join(args.out, 'wav'), exist_ok=True)
    n_wav = 0
    with open(os.path.join(args.out, 'raw.list'), 'w') as f_raw, \
            open(os.path.join(args.out, 'utt2spk'), 'w') as f_u2s:
        for u, s in selected:
            rate, x = wavfile.read(io.BytesIO(reader.read(row[u])))
            if rate != SAMPLE_RATE:
                raise SystemExit('%s: sample rate %d' % (u, rate))
            x = x if x.ndim == 1 else x[:, 0]
            key = (u[:-4] if u.endswith('.wav') else u) + '__imp'
            wav = os.path.join(args.out, 'wav', key + '.wav')
            os.makedirs(os.path.dirname(wav), exist_ok=True)
            wavfile.write(wav, SAMPLE_RATE,
                          x[:int(args.max_sec * SAMPLE_RATE)])
            f_raw.write(json.dumps(dict(key=key, wav=wav, spk=s)) + '\n')
            f_u2s.write('%s %s\n' % (key, s))
            n_wav += 1
    reader.close()
    summary = dict(new_speakers=len(by_spk), listed_speakers=len(speakers),
                   wavs=n_wav, per_spk=args.per_spk, max_sec=args.max_sec,
                   seconds=round(time.time() - t0, 1))
    with open(os.path.join(args.out, 'SUMMARY.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


# ----------------------------------------------------------------------------
# imprint


def class_means(emb_scps, emb_utt2spks, sample_cap=200000):
    """Per speaker: sum of the L2-normalised embeddings and their count.

    Also returns a ~6 % sample of (speaker, embedding) for the top-1 check.
    """
    import kaldiio
    import torch
    u2s = {}
    for path in emb_utt2spks:
        with open(path) as f:
            for line in f:
                if line.strip():
                    k, s = line.split()[:2]
                    u2s[k] = s
    acc, cnt, samples = {}, collections.Counter(), []
    for scp in emb_scps:
        for key, vec in kaldiio.load_scp_sequential(scp):
            s = u2s.get(key)
            if s is None:
                continue
            e = torch.from_numpy(np.array(vec, dtype=np.float32).reshape(-1))
            e = e / (e.norm() + 1e-12)
            acc[s] = acc.get(s, 0) + e
            cnt[s] += 1
            if len(samples) < sample_cap and int(sha(key)[:2], 16) < 16:
                samples.append((s, e))
    return acc, cnt, samples


def build_head(W, old, new, acc, cnt, K):
    """The head for the class list ``new`` from the head ``W`` of ``old``.

    Returns (W2, {imprinted speaker: class id}, target row norm).
    """
    import torch
    old_id = {s: i for i, s in enumerate(old)}
    target_norm = float(W.float().norm(dim=1).mean())
    missing = [s for s in new if s not in old_id and s not in acc]
    if missing:
        raise SystemExit('%d new speakers have no imprint embedding, e.g. %s'
                         % (len(missing), missing[:3]))
    D = W.shape[1]
    W2 = torch.empty((K * len(new), D), dtype=W.dtype)
    imprinted = {}
    for j, s in enumerate(new):
        if s in old_id:
            o = old_id[s]
            W2[j * K:(j + 1) * K] = W[o * K:(o + 1) * K]
        else:
            m = acc[s] / cnt[s]
            r = (m / (m.norm() + 1e-12) * target_norm).to(W.dtype)
            W2[j * K:(j + 1) * K] = r.unsqueeze(0).expand(K, D)
            imprinted[s] = j
    return W2, imprinted, target_norm


def cmd_imprint(args):
    import torch
    from ovsb.model.head_record import (head_sidecar_path, verify_head,
                                        write_head_sidecar)
    t0 = time.time()
    K = args.K
    old = labels_sorted(args.old_utt2spk)
    new = labels_sorted(args.new_utt2spk)
    old_id = {s: i for i, s in enumerate(old)}
    ckpt = torch.load(args.init, map_location='cpu', weights_only=False)
    wrapped = isinstance(ckpt, dict) and isinstance(ckpt.get('model'), dict)
    state = ckpt['model'] if wrapped else ckpt
    W = state[args.key]
    if W.shape[0] != K * len(old):
        raise SystemExit('%s has %d rows, %s needs %d x %d: not the list '
                         'this head was built for' % (
                             args.key, W.shape[0], args.old_utt2spk, K,
                             len(old)))
    old_side = {}
    if os.path.exists(head_sidecar_path(args.init)):
        with open(head_sidecar_path(args.init)) as f:
            old_side = json.load(f)
        verify_head(old_side, old, int(W.shape[0]),
                    head_sidecar_path(args.init))
    others = [k for k, v in state.items()
              if hasattr(v, 'shape') and v.ndim >= 1 and k != args.key
              and v.shape[0] in (len(old), K * len(old))]
    if others:
        raise SystemExit('other class-sized tensors would need remapping '
                         'too: %s' % others)
    acc, cnt, samples = class_means(args.emb_scp, args.emb_utt2spk)
    W2, imprinted, target_norm = build_head(W, old, new, acc, cnt, K)

    # kept classes: bit-identical rows
    kept = [j for j, s in enumerate(new) if s in old_id]
    if kept:
        src = torch.tensor([old_id[new[j]] for j in kept])
        dst = torch.tensor(kept)
        rows = torch.arange(K)[None]
        if not torch.equal(W2[(dst[:, None] * K + rows).reshape(-1)],
                           W[(src[:, None] * K + rows).reshape(-1)]):
            raise SystemExit('internal error: a kept class row changed')

    # diagnostics: closest old class of every new class; own-class top-1
    names = sorted(imprinted)
    dup, best = [], torch.zeros(0)
    if names and old:
        Wn_old = torch.nn.functional.normalize(W.float(), dim=1)
        Q = torch.nn.functional.normalize(torch.stack(
            [W2[imprinted[s] * K].float() for s in names]), dim=1)
        best, arg = [], []
        for i in range(0, len(Q), 256):
            b, ix = (Q[i:i + 256] @ Wn_old.T).max(dim=1)
            best.append(b)
            arg.append(ix // K)
        best, arg = torch.cat(best), torch.cat(arg)
        dup = [(names[i], old[int(arg[i])], round(float(best[i]), 4))
               for i in torch.argsort(-best).tolist()
               if float(best[i]) > args.dup_cos]
        del Wn_old
    hit = tot = 0
    if samples and not args.no_top1:
        W2n = torch.nn.functional.normalize(W2.float(), dim=1)
        new_id = {s: j for j, s in enumerate(new)}
        samples = [(s, e) for s, e in samples if s in new_id]
        for i in range(0, len(samples), 256):
            chunk = samples[i:i + 256]
            c = torch.stack([e for _, e in chunk]) @ W2n.T
            pred = c.view(len(chunk), len(new), K).max(dim=2).values \
                .argmax(dim=1)
            hit += sum(int(p) == new_id[s]
                       for (s, _), p in zip(chunk, pred.tolist()))
            tot += len(chunk)
        del W2n

    out_state = collections.OrderedDict(
        (k, (W2 if k == args.key else v)) for k, v in state.items())
    out = dict(ckpt, model=out_state) if wrapped else out_state
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out + '.tmp')
    os.replace(args.out + '.tmp', args.out)
    with open(args.out + '.dup.tsv', 'w') as f:
        f.write('new_class\tclosest_old_class\tcosine\n')
        f.writelines('%s\t%s\t%s\n' % row for row in dup)
    row_norm = W.float().norm(dim=1)
    n_emb = [cnt[s] for s in imprinted]
    report = dict(
        init=args.init, out=args.out, key=args.key, K=K,
        old_classes=len(old), new_classes=len(new), kept=len(kept),
        dropped=len(set(old) - set(new)), imprinted=len(imprinted),
        old_row_norm=dict(mean=round(target_norm, 4),
                          min=round(float(row_norm.min()), 4),
                          max=round(float(row_norm.max()), 4)),
        emb_per_class=dict(min=min(n_emb), mean=round(float(np.mean(n_emb)),
                                                      1)) if n_emb else None,
        max_cos_to_old=dict(mean=round(float(best.mean()), 4),
                            p99=round(float(best.quantile(0.99)), 4),
                            max=round(float(best.max()), 4))
        if len(best) else None,
        dup_cos=args.dup_cos, n_dup_flagged=len(dup),
        own_class_top1=round(hit / max(1, tot), 4), own_class_top1_n=tot,
        new_utt2spk=args.new_utt2spk, old_utt2spk=args.old_utt2spk,
        checks='every kept class row identical; no other class-sized tensor',
        seconds=round(time.time() - t0, 1))
    with open(args.out + '.json', 'w') as f:
        json.dump(report, f, indent=1)
    layers = args.frontend_num_layers or old_side.get('frontend_num_layers')
    extra = dict(train_label=args.new_utt2spk, imprinted=dict(
        classes=len(imprinted), kept_classes=len(kept),
        tool='local/init/imprint_head.py imprint',
        embeddings=list(args.emb_scp), emb_utt2spk=list(args.emb_utt2spk)))
    if layers:
        extra['frontend_num_layers'] = int(layers)
    write_head_sidecar(args.out, new, int(W2.shape[0]), source=args.init,
                       extra=extra)
    print(json.dumps(report, indent=1))
    return 0


# ----------------------------------------------------------------------------
# selftest


def cmd_selftest(args):
    """Synthetic checkpoint: kept rows copied, rows of dropped classes
    removed, new rows = scaled class mean K times, sidecar verifies."""
    import kaldiio
    import torch
    from ovsb.model.head_record import head_sidecar_path, verify_head
    rng = np.random.default_rng(0)
    d = tempfile.mkdtemp(dir=args.tmp)
    D, K = 16, 3
    old = ['id%05d' % i for i in range(50)] + ['vb2_x1', 'vb2_x2']
    extra = ['cn2_id1%04d' % i for i in range(20)]
    keep = [s for s in old if s != 'vb2_x2']
    with open(d + '/old', 'w') as f:
        f.writelines('u%s %s\n' % (s, s) for s in old)
    with open(d + '/new', 'w') as f:
        f.writelines('u%s %s\n' % (s, s) for s in keep + extra)
    W = torch.randn(K * len(old), D, generator=torch.Generator()
                    .manual_seed(0)) * 3
    torch.save({HEAD_KEY: W, 'other.bias': torch.zeros(7)}, d + '/init.pt')
    centre = {s: rng.normal(size=D) for s in extra}
    with kaldiio.WriteHelper('ark,scp:%s/e.ark,%s/e.scp' % (d, d)) as w, \
            open(d + '/eu2s', 'w') as f:
        for s in extra:
            for i in range(5):
                k = '%s/u%d__imp' % (s, i)
                w(k, (centre[s] + 0.1 * rng.normal(size=D))
                  .astype(np.float32))
                f.write('%s %s\n' % (k, s))
    cmd_imprint(argparse.Namespace(
        init=d + '/init.pt', old_utt2spk=d + '/old', new_utt2spk=d + '/new',
        emb_scp=[d + '/e.scp'], emb_utt2spk=[d + '/eu2s'],
        out=d + '/out.pt', key=HEAD_KEY, K=K, dup_cos=0.99, no_top1=False,
        frontend_num_layers=None))
    state = torch.load(d + '/out.pt', weights_only=False)
    o = state[HEAD_KEY]
    new = sorted(keep + extra)
    assert o.shape == (K * len(new), D)
    assert torch.equal(state['other.bias'], torch.zeros(7))
    mean_norm = float(W.norm(dim=1).mean())
    for j, s in enumerate(new):
        r = o[j * K:(j + 1) * K]
        if s in old:
            i = old.index(s)
            assert torch.equal(r, W[i * K:(i + 1) * K]), s
        else:
            assert torch.equal(r[0], r[1]) and torch.equal(r[0], r[2]), s
            assert abs(float(r[0].norm()) - mean_norm) < 1e-3, s
            c = torch.nn.functional.cosine_similarity(
                r[0].double(), torch.tensor(centre[s]), dim=0)
            assert c > 0.99, (s, c)
    with open(d + '/out.pt.json') as f:
        report = json.load(f)
    assert report['kept'] == len(keep) and report['dropped'] == 1, report
    assert report['imprinted'] == len(extra), report
    assert report['own_class_top1'] == 1.0 or \
        report['own_class_top1_n'] == 0, report
    with open(head_sidecar_path(d + '/out.pt')) as f:
        verify_head(json.load(f), new, K * len(new), 'selftest sidecar')
    print('SELFTEST PASS', d)
    return 0


def get_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('list', help='audio of the speakers without a row')
    p.add_argument('--utt2spk', required=True, help='training list')
    p.add_argument('--index', required=True, help='its tar index directory')
    p.add_argument('--index_root', default=None,
                   help='directory for relative shard paths of the index')
    p.add_argument('--old_utt2spk', required=True,
                   help='list the existing head was built for')
    p.add_argument('--out', required=True)
    p.add_argument('--per_spk', type=int, default=20)
    p.add_argument('--max_sec', type=float, default=10.0)
    p.add_argument('--limit_spk', type=int, default=0,
                   help='test only: the first N speakers')
    p = sub.add_parser('imprint', help='write the checkpoint with the head')
    for name in ('init', 'old_utt2spk', 'new_utt2spk', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--emb_scp', required=True, nargs='+',
                   help='embedding scp file(s) of the imprint utterances')
    p.add_argument('--emb_utt2spk', required=True, nargs='+',
                   help="their '<key> <speaker>' file(s)")
    p.add_argument('--key', default=HEAD_KEY)
    p.add_argument('--K', type=int, default=3, help='rows per class')
    p.add_argument('--dup_cos', type=float, default=0.7)
    p.add_argument('--no_top1', action='store_true',
                   help='skip the own-class top-1 diagnostic')
    p.add_argument('--frontend_num_layers', type=int, default=None,
                   help='recorded in the sidecar (default: from the '
                   'sidecar of --init)')
    p = sub.add_parser('selftest', help='synthetic check')
    p.add_argument('--tmp', default=os.environ.get('TMPDIR') or None)
    return ap.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    return dict(list=cmd_list, imprint=cmd_imprint,
                selftest=cmd_selftest)[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
