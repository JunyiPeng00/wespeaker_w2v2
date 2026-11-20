#!/bin/bash

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

# Stage-1 embeddings of the utterances listed by `imprint_head.py list`
# (the speakers that have no classifier row), extracted with the truncated
# stage-1 model: the model stage 2 starts from. Evaluation protocol: first
# --maxsec seconds, one pass, TF32, masked batches. Needs a GPU. Run from
# the recipe directory:
#
#   bash local/init/imprint_extract.sh --config exp/stage1_frozen/config.yaml \
#     --model exp/init/stage1_L22.pt --list_dir exp/init/imprint \
#     --out_dir exp/init/imprint_emb
#
# Writes <out_dir>/config.yaml and <out_dir>/embeddings/imprint/xvector.scp.

config=exp/stage1_frozen/config.yaml   # config.yaml of the stage-1 run
model=exp/init/stage1_L22.pt           # truncated stage-1 checkpoint
list_dir=exp/init/imprint              # raw.list of imprint_head.py list
out_dir=exp/init/imprint_emb
layers=22                              # encoder layers of --model
maxsec=10
gpus="[0]"
nj=
num_workers=4

. ./path.sh
. tools/parse_options.sh
set -eo pipefail

for f in "${config}" "${model}" "${list_dir}/raw.list"; do
  if [ ! -f "${f}" ]; then
    echo "$0: missing ${f}" >&2
    exit 1
  fi
done
if [ -z "${nj}" ]; then
  nj=$(echo "${gpus}" | awk -F ',' '{print NF}')
fi
mkdir -p "${out_dir}"

python local/eval/make_extract_config.py --config "${config}" \
  --out "${out_dir}/config.yaml" --maxsec "${maxsec}" --layers "${layers}"

bash tools/extract_embedding.sh --exp_dir "${out_dir}" \
  --model_path "${model}" --data_type raw \
  --data_list "${list_dir}/raw.list" \
  --wavs_num "$(wc -l < "${list_dir}/raw.list")" --store_dir imprint \
  --batch_size 1 --num_workers "${num_workers}" --nj "${nj}" \
  --gpus "${gpus}" --reverb_data "" --noise_data "" --aug_prob 0.0

if ! grep -q Successfully "${out_dir}/embeddings/imprint/extract.result"; then
  echo "$0: extraction incomplete, see ${out_dir}/embeddings/imprint/log" >&2
  exit 1
fi
echo "$0: ${out_dir}/embeddings/imprint/xvector.scp"
