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
"""Cut a W2V-BERT + MHFA checkpoint to its first N encoder layers.

    python local/init/truncate_ckpt.py SRC N OUT \\
        [--train-label utt2spk [--rows-per-class 3]]

The tensors of encoder layers >= N are dropped and the MHFA layer weights
``weights_k`` / ``weights_v`` keep their first N + 1 entries (hidden states
0..N: the input of layer 1 and the outputs of layers 1..N). MHFA applies a
softmax over the kept entries, so the kept layers get their old weights
renormalised. Every other tensor is copied unchanged. Use the result with
``frontend_num_layers: N`` and ``nb_layer: N + 1``.

``--train-label``: the list whose speakers are the classes of the head of
SRC. The head must have exactly classes x ``--rows-per-class`` rows, an
existing ``SRC.head.json`` must agree with the list, and ``OUT.head.json``
is written (speaker-list digest, class and row counts), which the trainer
checks before it loads the checkpoint.
"""

import argparse
import json
import os
import re
import sys

import torch

from ovsb.model.head_record import (head_sidecar_path, verify_head,
                                    write_head_sidecar)

HEAD_KEY = 'projection.weight'
LAYER_WEIGHTS = ('weights_k', 'weights_v')
_LAYER = re.compile(r'encoder\.layers\.(\d+)\.')


def unwrap(ckpt):
    """The state dict of a checkpoint (plain, or under the key 'model')."""
    if isinstance(ckpt, dict) and isinstance(ckpt.get('model'), dict):
        return ckpt['model']
    return ckpt


def truncate_state(state, n):
    """(state cut to n layers, dropped tensor count, softmax mass kept)."""
    layers = sorted({int(m.group(1)) for k in state
                     for m in [_LAYER.search(k)] if m})
    if not layers or not 0 < n <= len(layers):
        raise ValueError('cannot keep %d of %d encoder layers' %
                         (n, len(layers)))
    out, dropped, mass = {}, 0, {}
    for key, value in state.items():
        m = _LAYER.search(key)
        if m and int(m.group(1)) >= n:
            dropped += 1
            continue
        if key in LAYER_WEIGHTS:
            if value.shape[0] < n + 1:
                raise ValueError('%s has %d entries, %d needed' %
                                 (key, value.shape[0], n + 1))
            w = torch.softmax(value.float(), -1)
            mass[key] = float(w[:n + 1].sum())
            value = value[:n + 1].clone()
        out[key] = value
    return out, dropped, mass


def read_speakers(train_label):
    with open(train_label) as f:
        return sorted({line.split()[1] for line in f
                       if len(line.split()) >= 2})


def get_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('src', help='checkpoint to cut')
    ap.add_argument('n', type=int, help='encoder layers to keep')
    ap.add_argument('out', help='checkpoint to write')
    ap.add_argument('--train-label', default=None,
                    help='utt2spk of the list the head of SRC was built for')
    ap.add_argument('--rows-per-class', type=int, default=3)
    return ap.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    state = unwrap(torch.load(args.src, map_location='cpu',
                              weights_only=False))
    spks = rows = None
    if args.train_label:
        spks = read_speakers(args.train_label)
        rows = int(state[HEAD_KEY].shape[0])
        need = len(spks) * args.rows_per_class
        if rows != need:
            print('%s: the head has %d rows, %s needs %d x %d = %d: it was '
                  'not built for this class list' % (
                      args.src, rows, args.train_label, len(spks),
                      args.rows_per_class, need), file=sys.stderr)
            return 1
        side = head_sidecar_path(args.src)
        if os.path.exists(side):
            with open(side) as f:
                verify_head(json.load(f), spks, rows, side)
    out, dropped, mass = truncate_state(state, args.n)
    for key, kept in mass.items():
        print('%s: softmax mass kept in hidden states 0..%d: %.4f' %
              (key, args.n, kept))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save(out, args.out + '.tmp')
    os.replace(args.out + '.tmp', args.out)
    print('%s: dropped %d tensors of encoder layers >= %d' %
          (args.out, dropped, args.n))
    if spks is not None:
        record = write_head_sidecar(
            args.out, spks, rows, source=args.src,
            extra=dict(train_label=args.train_label,
                       frontend_num_layers=args.n))
        print('%s: %s' % (head_sidecar_path(args.out), record))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
