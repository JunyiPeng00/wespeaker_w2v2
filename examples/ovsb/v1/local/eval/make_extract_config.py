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
"""Write the extraction config of the evaluation protocol.

    python local/eval/make_extract_config.py --config EXP/config.yaml \\
        --out EVAL/config.yaml --maxsec 10 [--layers 22]

Input: a wespeaker-schema config of the model (``<exp_dir>/config.yaml``).
Output: the same config with the protocol keys set, for the stock
``wespeaker/bin/extract.py``:

* ``dataset_args.<frontend>_args.max_input_seconds`` = ``--maxsec``: every
  utterance is cut to its first ``maxsec`` seconds and embedded in a single
  forward pass (no chunk averaging);
* precision ``--dtype``: ``tf32`` (fp32 with TF32 matmul / cuDNN, the
  protocol), ``fp32`` or ``bf16``;
* ``extract_batch_seconds`` = ``--batch_seconds``: whole utterances in
  length-sorted, zero-padded, masked batches of at most this much audio
  (0 = one utterance per forward);
* ``--layers N`` (optional): ``dataset_args.frontend_num_layers = N`` and
  ``model_args.nb_layer = N + 1``, for a base config written for the
  untruncated encoder. Without it the base config must be consistent.

Keys that only steer training batches are removed. The script fails when
the result is not a consistent truncated / untruncated model description.
"""

import argparse
import copy
import sys

import yaml

TRAIN_ONLY = ('num_frms_choices', 'var_crop', 'episodic_sampler',
              'spk_alias', 'utt_weights')
DTYPES = ('tf32', 'fp32', 'bf16')


def make_config(base, maxsec, dtype='tf32', batch_seconds=320.0,
                layers=None):
    """Return the extraction config (a new dict) for ``base``."""
    if dtype not in DTYPES:
        raise ValueError('dtype must be one of %s' % (DTYPES, ))
    if not maxsec > 0:
        raise ValueError('maxsec must be positive')
    conf = copy.deepcopy(base)
    data = conf['dataset_args']
    frontend = data.get('frontend', 'fbank')
    if frontend not in ('w2vbert_sv', 'omniasr_sv'):
        raise ValueError('frontend %r has no max_input_seconds' % frontend)
    fe = data[frontend + '_args']
    fe['max_input_seconds'] = float(maxsec)
    if frontend == 'w2vbert_sv':
        fe.setdefault('feature_extractor', 'torch')
    conf['extract_amp'] = dtype == 'bf16'
    conf['extract_tf32'] = dtype == 'tf32'
    if dtype == 'bf16':
        conf['extract_amp_dtype'] = 'bfloat16'
    conf['extract_batch_seconds'] = float(batch_seconds)
    for key in TRAIN_ONLY:
        data.pop(key, None)
    if layers is not None:
        data['frontend_num_layers'] = int(layers)
        conf['model_args']['nb_layer'] = int(layers) + 1
    num_layers = data.get('frontend_num_layers')
    nb_layer = conf['model_args'].get('nb_layer')
    if num_layers and nb_layer != int(num_layers) + 1:
        raise ValueError('model_args.nb_layer %s != frontend_num_layers %s '
                         '+ 1' % (nb_layer, num_layers))
    feat_dim = conf['model_args'].get('feat_dim')
    if feat_dim is not None and feat_dim <= 0:
        raise ValueError('model_args.feat_dim is not resolved: use the '
                         'config.yaml of an experiment directory')
    if batch_seconds and data.get('cmvn', True):
        raise ValueError('masked batching needs dataset_args.cmvn: false')
    return conf


def get_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--config', required=True, help='model config (yaml)')
    ap.add_argument('--out', required=True, help='extraction config to write')
    ap.add_argument('--maxsec', type=float, required=True,
                    help='input cap in seconds (protocol: 10 or 60)')
    ap.add_argument('--dtype', default='tf32', choices=DTYPES)
    ap.add_argument('--batch_seconds', type=float, default=320.0)
    ap.add_argument('--layers', type=int, default=None,
                    help='encoder layers to keep (sets nb_layer too)')
    return ap.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    with open(args.config) as f:
        base = yaml.safe_load(f)
    try:
        conf = make_config(base, args.maxsec, args.dtype, args.batch_seconds,
                           args.layers)
    except (KeyError, ValueError) as exc:
        print('%s: %s' % (args.config, exc), file=sys.stderr)
        return 1
    with open(args.out, 'w') as f:
        yaml.safe_dump(conf, f)
    fe = conf['dataset_args'][conf['dataset_args']['frontend'] + '_args']
    print('%s: cap %.0f s, %s, batch %.0f s, encoder layers %s, nb_layer %s'
          % (args.out, fe['max_input_seconds'], args.dtype,
             args.batch_seconds,
             conf['dataset_args'].get('frontend_num_layers', 'all'),
             conf['model_args'].get('nb_layer')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
