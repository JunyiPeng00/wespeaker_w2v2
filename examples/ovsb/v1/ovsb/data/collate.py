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
"""Last stage of the training pipeline: padded rows with a fixed field set.

:func:`pad_batch` is a 1:1 generator stage. It zero-pads the waveform of
every row to the length the crop stage chose for its batch
(``sample['pad_to']``) and emits exactly the fields of :data:`BATCH_FIELDS`,
so that the default collate of the data loader stacks ``batch_size``
consecutive rows into one batch:

==============  =========  ================================================
field           collated   content
==============  =========  ================================================
``key``         list[str]  utterance key
``label``       int64      classifier class of the speaker
``sample_rate`` int64      sampling rate of ``wav``
``wav``         float32    (B, 1, T) waveforms, zero-padded at the end
``wav_len``     int64      valid samples of the row
``pair_role``   int64      0 plain row, 1 anchor, 2 view of the row before
``arm``         int64      sampler arm index
``kpos``        int64      position of the row in its speaker group
``ptype``       int64      positive type (``ovsb.data.policies.PTYPES``)
``aug_code``    int64      index into :data:`AUG_TYPES`, -1 = no type drawn
``chain_code``  int64      codec applied by the channel stage
                           (:data:`CHAIN_CODECS`), -1 = none
``chain_fail``  int64      codec the channel stage drew but could not
                           apply (:data:`CHAIN_CODECS`), -1 = none
``zr_lsb``      float64    share of samples at or below the LSB level
``zr_run``      int64      1 = the crop had a run of exact zeros
``zr_dith``     int64      1 = zero runs were dithered
``zr_bypass``   int64      1 = silent crop, left untouched
``eg_redraw``   int64      energy-gate redraws, -1 = row not gated
``eg_fail``     int64      1 = kept below the energy threshold
==============  =========  ================================================

Fields of a sample that are not listed (language, session, recording
conditions) are not part of a batch.
"""

from typing import Any, Dict, Iterable, Iterator

import torch

from ovsb.config.schema import AUG_TYPES, CODECS

Sample = Dict[str, Any]

# codecs of the channel stage in the index order of ``chain_code``
CHAIN_CODECS = CODECS
# fields of an emitted row, in order
BATCH_FIELDS = ('key', 'label', 'sample_rate', 'wav', 'wav_len', 'pair_role',
                'arm', 'kpos', 'ptype', 'aug_code', 'chain_code',
                'chain_fail', 'zr_lsb', 'zr_run', 'zr_dith', 'zr_bypass',
                'eg_redraw', 'eg_fail')
# fields the sampler must have put into every sample
REQUIRED_FIELDS = ('key', 'label', 'sample_rate', 'wav', 'arm', 'kpos',
                   'ptype')


def _code(value: Any, table: tuple) -> int:
    return table.index(value) if value in table else -1


def pad_batch(data: Iterable[Sample]) -> Iterator[Sample]:
    """Pad every row to ``sample['pad_to']`` and emit :data:`BATCH_FIELDS`.

    A row without ``pad_to`` keeps its length. Fields a stage did not set
    get their neutral value: ``pair_role`` 0, ``aug_code`` / ``chain_code``
    / ``chain_fail`` / ``eg_redraw`` -1, the other counters 0.

    Raises:
        KeyError: a sample lacks one of :data:`REQUIRED_FIELDS`.
        ValueError: a waveform is longer than its ``pad_to``.
    """
    for sample in data:
        missing = [k for k in REQUIRED_FIELDS if k not in sample]
        if missing:
            raise KeyError('sample %r lacks the fields %s' % (
                sample.get('key'), missing))
        wav = sample['wav']
        n = wav.shape[1]
        pad_to = sample.get('pad_to', n)
        if n > pad_to:
            raise ValueError('%s: %d samples, but the batch is padded to %d'
                             % (sample['key'], n, pad_to))
        if n < pad_to:
            wav = torch.nn.functional.pad(wav, (0, pad_to - n))
        out = dict(
            key=sample['key'],
            label=sample['label'],
            sample_rate=sample['sample_rate'],
            wav=wav,
            wav_len=n,
            pair_role=sample.get('pair_role', 0),
            arm=sample['arm'],
            kpos=sample['kpos'],
            ptype=sample['ptype'],
            aug_code=_code(sample.get('aug_type'), AUG_TYPES),
            chain_code=_code(sample.get('chain'), CHAIN_CODECS),
            chain_fail=_code(sample.get('chain_fail'), CHAIN_CODECS),
            zr_lsb=float(sample.get('zr_lsb', 0.0)),
            zr_run=int(sample.get('zr_run', 0)),
            zr_dith=int(sample.get('zr_dith', 0)),
            zr_bypass=int(sample.get('zr_bypass', 0)),
            eg_redraw=int(sample.get('eg_redraw', -1)),
            eg_fail=int(sample.get('eg_fail', 0)))
        assert tuple(out) == BATCH_FIELDS
        yield out
