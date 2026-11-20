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
"""Relative weight drift of a model from a reference checkpoint, with a gate.

    python local/drift_gate.py --exp exp/stage2_v9c --ref REF.pt --gate 0.08
    python local/drift_gate.py --model exp/.../model_3.pt --ref REF.pt

drift = ||theta - theta_ref||_2 / ||theta_ref||_2 over all floating-point
tensors of a group: ``encoder`` = keys starting with ``frontend.``, ``mhfa``
= every other key that does not start with ``projection.`` (the classifier
head is not part of either group). CPU only; tensors are memory-mapped.

``--exp``: every ``models/model_N.pt`` (N >= 1) not yet in
``<exp>/drift.tsv`` is measured and appended (columns epoch, encoder, mhfa,
gate). ``--model``: one checkpoint, printed only.

Exit code: 0 = no encoder drift above ``--gate``; 3 = the gate is exceeded
(for ``--exp``: by any epoch in ``drift.tsv``, old rows included); any
other code = error. The gate fails closed: it is an error when the reference
is missing or unreadable, when a non-projection floating-point tensor of the
model is absent from the reference or has another shape or type there, when
a group is empty, or when a drift is not finite.
"""

import argparse
import glob
import math
import os
import re
import sys

import torch

EXIT_OK, EXIT_ERROR, EXIT_EXCEEDED = 0, 2, 3
HEAD_PREFIX = 'projection.'
GROUPS = ('encoder', 'mhfa')


class DriftError(RuntimeError):
    """The drift cannot be measured: the gate must not pass."""


def load_state(path):
    """State dict of a checkpoint (plain, or under 'model'/'state_dict')."""
    if not os.path.isfile(path):
        raise DriftError('%s: no such checkpoint' % path)
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False,
                          mmap=True)
    except Exception as exc:  # noqa: any failure to read is a gate error
        raise DriftError('%s: cannot read the checkpoint (%r)' %
                         (path, exc)) from exc
    if isinstance(ckpt, dict):
        for wrap in ('model', 'state_dict'):
            if isinstance(ckpt.get(wrap), dict):
                ckpt = ckpt[wrap]
                break
    if not isinstance(ckpt, dict) or not ckpt or not all(
            torch.is_tensor(v) for v in ckpt.values()):
        raise DriftError('%s: not a state dict' % path)
    return ckpt


def group_of(name):
    return 'encoder' if name.startswith('frontend.') else 'mhfa'


def check_coverage(ref, model, what='model'):
    """Every non-head float tensor of the model must be comparable."""
    bad = []
    for name, x in model.items():
        if name.startswith(HEAD_PREFIX) or not x.is_floating_point():
            continue
        r = ref.get(name)
        if r is None:
            bad.append('%s: not in the reference' % name)
        elif r.shape != x.shape:
            bad.append('%s: shape %s, reference %s' % (
                name, tuple(x.shape), tuple(r.shape)))
        elif not r.is_floating_point():
            bad.append('%s: reference tensor is %s' % (name, r.dtype))
    if bad:
        raise DriftError('%s: %d tensors cannot be compared with the '
                         'reference: %s%s' % (what, len(bad),
                                              '; '.join(bad[:5]),
                                              ' ...' if len(bad) > 5 else ''))


def drift(ref, model, what='model'):
    """{'encoder': d, 'mhfa': d} of ``model`` against ``ref``.

    Raises DriftError when the two state dicts are not comparable.
    """
    check_coverage(ref, model, what)
    num, den = {}, {}
    for name, r in ref.items():  # reference order: fixed summation order
        if name not in model or name.startswith(HEAD_PREFIX) or \
                not r.is_floating_point():
            continue
        x = model[name]
        if x.shape != r.shape:
            raise DriftError('%s: %s has shape %s, reference %s' % (
                what, name, tuple(x.shape), tuple(r.shape)))
        g = group_of(name)
        num[g] = num.get(g, 0.0) + float(((x.float() - r.float()) ** 2).sum())
        den[g] = den.get(g, 0.0) + float((r.float() ** 2).sum())
    out = {}
    for g in GROUPS:
        if g not in num or not den[g] > 0:
            raise DriftError('%s: no %s tensors to compare' % (what, g))
        out[g] = (num[g] / den[g]) ** 0.5
        if not math.isfinite(out[g]):
            raise DriftError('%s: %s drift is not finite' % (what, g))
    return out


def epoch_of(path):
    return int(re.findall(r'model_(\d+)\.pt$', path)[0])


def run_exp(exp, ref, gate):
    """Measure the new epochs of ``exp``; True if any epoch exceeds."""
    models = sorted((p for p in glob.glob(os.path.join(
        exp, 'models', 'model_*.pt')) if re.search(r'model_\d+\.pt$', p)),
        key=epoch_of)
    table = os.path.join(exp, 'drift.tsv')
    done = set()
    if os.path.isfile(table):
        with open(table) as f:
            done = {line.split('\t')[0] for line in f if line[:1].isdigit()}
    else:
        if not os.path.isdir(exp):
            raise DriftError('%s: no such experiment directory' % exp)
        with open(table, 'w') as f:
            f.write('epoch\tencoder\tmhfa\tgate\n')
    for path in models:
        epoch = str(epoch_of(path))
        if epoch == '0' or epoch in done:
            continue
        d = drift(ref, load_state(path), path)
        line = '%s\t%.5f\t%.5f\t%s\n' % (
            epoch, d['encoder'], d['mhfa'],
            'ok' if d['encoder'] <= gate else 'EXCEEDED')
        with open(table, 'a') as f:
            f.write(line)
        print(line, end='', flush=True)
    with open(table) as f:
        return any(line.rstrip().endswith('EXCEEDED') for line in f)


def get_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument('--exp', help='experiment directory (models/model_N.pt)')
    what.add_argument('--model', help='one checkpoint')
    ap.add_argument('--ref', required=True, help='reference checkpoint')
    ap.add_argument('--gate', type=float, default=0.02,
                    help='largest allowed encoder drift (default 0.02)')
    return ap.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    try:
        if not args.gate > 0:
            raise DriftError('--gate must be positive')
        ref = load_state(args.ref)
        if args.exp:
            exceeded = run_exp(args.exp, ref, args.gate)
        else:
            d = drift(ref, load_state(args.model), args.model)
            exceeded = not d['encoder'] <= args.gate
            print('encoder\t%.5f\tmhfa\t%.5f\t%s' % (
                d['encoder'], d['mhfa'], 'EXCEEDED' if exceeded else 'ok'))
    except DriftError as exc:
        print('DRIFT GATE ERROR: %s' % exc, file=sys.stderr)
        return EXIT_ERROR
    if exceeded:
        print('DRIFT GATE EXCEEDED (> %.3f)' % args.gate, file=sys.stderr)
        return EXIT_EXCEEDED
    return EXIT_OK


if __name__ == '__main__':
    try:
        code = main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: never exit 0 or 3 on a crash
        print('DRIFT GATE ERROR: %r' % (exc, ), file=sys.stderr)
        code = EXIT_ERROR
    raise SystemExit(code)
