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
"""Vectorised drop-in for wespeaker/bin/score.py (same arguments, same output files and format).

score.py rebuilds the mean and reloads every embedding once per call and scores one trial at a time with
sklearn (~530 s for the three cleaned Vox1 lists). Here the mean is computed once, the embeddings are
loaded once, L2-normalised with sklearn's `normalize` (as cosine_similarity does) and every trial list is
scored with one gather + row-wise dot. Scores may differ from score.py in the 5th printed decimal through
float32 summation order only.
"""
import os
from pathlib import Path

import fire
import kaldiio
import numpy as np
from sklearn.preprocessing import normalize


def main(exp_dir, eval_scp_path, cal_mean, cal_mean_dir, *trials):
    mean_vec = 0.0
    if cal_mean:
        scp_path = os.path.join(cal_mean_dir, 'xvector.scp')
        print('Calculate mean statistics from {}.'.format(scp_path))
        acc, n = None, 0
        for _, vec in kaldiio.load_scp_sequential(scp_path):
            acc = np.zeros_like(vec) if acc is None else acc
            acc += vec
            n += 1
        mean_vec = acc / n
        np.save(os.path.join(cal_mean_dir, 'mean_vec.npy'), mean_vec)
    else:
        print('Do not do mean normalization for evaluation embeddings.')
    keys, embs = [], []
    for utt, emb in kaldiio.load_scp_sequential(eval_scp_path):
        keys.append(utt)
        embs.append(emb - mean_vec)
    E = normalize(np.stack(embs))
    index = {k: i for i, k in enumerate(keys)}
    store_dir = os.path.join(exp_dir, 'scores')
    Path(store_dir).mkdir(parents=True, exist_ok=True)
    for trial in trials:
        segs = [line.split() for line in open(trial) if line.strip()]
        i1 = np.fromiter((index[s[0]] for s in segs), dtype=np.int64, count=len(segs))
        i2 = np.fromiter((index[s[1]] for s in segs), dtype=np.int64, count=len(segs))
        scores = np.einsum('ij,ij->i', E[i1], E[i2])
        out = os.path.join(store_dir, os.path.basename(trial) + '.score')
        with open(out, 'w') as w:
            for s, sc in zip(segs, scores):
                if len(s) == 3:
                    w.write('{} {} {:.5f} {}\n'.format(s[0], s[1], sc, s[2]))
                else:
                    w.write('{} {} {:.5f}\n'.format(s[0], s[1], sc))
        print('scored {} trials -> {}'.format(len(segs), out))


if __name__ == '__main__':
    fire.Fire(main)
