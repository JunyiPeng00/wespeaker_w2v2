# Copyright (c) 2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare two kaldi scp/ark embedding sets on their common keys.

Agreement is judged at float32 GPU-reproducibility level: the relative
error max|a-b| / max|b| must stay below --rtol (embeddings are unnormalized,
so an absolute threshold would be scale-dependent).

Usage:
    python compare_embeddings.py <new.scp> <ref.scp> [--rtol 1e-5]
"""

import sys

import numpy as np
import kaldiio


def main():
    new_scp, ref_scp = sys.argv[1], sys.argv[2]
    rtol = 1e-5
    if '--rtol' in sys.argv:
        rtol = float(sys.argv[sys.argv.index('--rtol') + 1])

    new = {k: v for k, v in kaldiio.load_scp_sequential(new_scp)}
    ref = kaldiio.load_scp(ref_scp)

    common = [k for k in new if k in ref]
    print(f'{len(new)} new embeddings, {len(common)} found in reference')
    if not common:
        print('FAIL: no common keys')
        sys.exit(1)

    worst_rel, worst_cos = 0.0, 1.0
    for k in common:
        a = np.asarray(new[k], dtype=np.float64)
        b = np.asarray(ref[k], dtype=np.float64)
        rel = np.abs(a - b).max() / np.abs(b).max()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        worst_rel = max(worst_rel, rel)
        worst_cos = min(worst_cos, cos)
        print(f'{k}  rel|diff|={rel:.3e}  cos={cos:.8f}')

    print(f'WORST: rel|diff|={worst_rel:.3e}  cos={worst_cos:.8f}')
    if worst_rel <= rtol:
        print(f'PASS (rtol={rtol})')
    else:
        print(f'FAIL (rtol={rtol})')
        sys.exit(1)


if __name__ == '__main__':
    main()
