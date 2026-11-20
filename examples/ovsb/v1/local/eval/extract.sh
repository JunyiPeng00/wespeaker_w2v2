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

# Embeddings for the VoxCeleb1 evaluation (see local/eval/README.md):
# VoxCeleb1 whole utterances cut to --maxsec seconds, one pass each, and the
# VoxCeleb2-dev embeddings whose mean is subtracted before scoring.
# Run from the recipe directory on a node with GPUs:
#
#   bash local/eval/extract.sh --exp_dir exp/stage2_v9c \
#     --model exp/stage2_v9c/models/model_14.pt --tag ep14 --maxsec 10
#
# Writes <eval_dir>/config.yaml and <eval_dir>/embeddings/{vox1,vox2_dev}.
# Finished parts are kept; a failed extraction gives a non-zero exit code.

exp_dir=            # directory with the model's config.yaml
model=              # checkpoint
tag=eval
maxsec=10           # input cap in seconds: 10 or 60
data=data           # holds vox1/{shard.list,wav.scp}, vox2_dev/{shard.list,wav.scp}
gpus="[0]"
nj=                 # extraction processes (default: one per GPU)
dtype=tf32          # tf32 | fp32 | bf16
batch_sec=320       # seconds of audio per masked batch, 0 = one utterance
vox2_mean=frac      # frac | full | path of an existing vox2 xvector.scp
vox2_bs=64          # batch size of the fixed-length crops of vox2_mean=frac
min_mean_utts=100000
num_workers=4
eval_dir=           # default <exp_dir>/eval_<tag>_<dtype>_max<maxsec>s

. ./path.sh
. tools/parse_options.sh
set -eo pipefail

if [ -z "${exp_dir}" ] || [ ! -f "${exp_dir}/config.yaml" ] || [ ! -f "${model}" ]; then
  echo "$0: need --exp_dir (with config.yaml) and --model" >&2
  exit 1
fi
if [ "${maxsec}" != "10" ] && [ "${maxsec}" != "60" ]; then
  echo "$0: --maxsec ${maxsec} is outside the protocol (10 or 60)" >&2
fi
if [ -z "${nj}" ]; then
  nj=$(echo "${gpus}" | awk -F ',' '{print NF}')
fi
if [ -z "${eval_dir}" ]; then
  eval_dir=${exp_dir}/eval_${tag}_${dtype}_max${maxsec}s
fi
mkdir -p "${eval_dir}"

python local/eval/make_extract_config.py --config "${exp_dir}/config.yaml" \
  --out "${eval_dir}/config.yaml" --maxsec "${maxsec}" --dtype "${dtype}" \
  --batch_seconds "${batch_sec}"

count_vectors() {  # embeddings written so far in <eval_dir>/embeddings/$1
  { cat "${eval_dir}/embeddings/$1"/xvector_*.scp 2>/dev/null || true; } | wc -l
}

extract() {  # store_dir data_list wavs_num batch_size
  bash tools/extract_embedding.sh --exp_dir "${eval_dir}" \
    --model_path "${model}" --data_type shard --data_list "$2" \
    --wavs_num "$3" --store_dir "$1" --batch_size "$4" \
    --num_workers "${num_workers}" --nj "${nj}" --gpus "${gpus}" \
    --reverb_data "" --noise_data "" --aug_prob 0.0
}

# ---- VoxCeleb1: whole utterances, batch size 1 (masked batches inside)
vox1=${eval_dir}/embeddings/vox1
if grep -q Successfully "${vox1}/extract.result" 2>/dev/null; then
  echo "$0: vox1 embeddings exist"
else
  extract vox1 "${data}/vox1/shard.list" "$(wc -l < "${data}/vox1/wav.scp")" 1
  if ! grep -q Successfully "${vox1}/extract.result"; then
    echo "$0: vox1 extraction incomplete, see ${vox1}/log" >&2
    exit 1
  fi
fi

# ---- VoxCeleb2 dev: embeddings for the mean
vox2=${eval_dir}/embeddings/vox2_dev
if [ -f "${vox2}/MEAN_OK" ]; then
  echo "$0: vox2_dev mean embeddings exist: $(cat "${vox2}/MEAN_OK")"
elif [ "${vox2_mean}" = "frac" ]; then
  # every 10th shard, batches of fixed-length random crops (num_frms)
  awk 'NR % 10 == 1' "${data}/vox2_dev/shard.list" > "${eval_dir}/vox2_dev_frac.list"
  # the number of utterances in the subset is not known beforehand: the
  # completeness line of extract_embedding.sh does not apply, count instead
  extract vox2_dev "${eval_dir}/vox2_dev_frac.list" 0 "${vox2_bs}"
  n=$(count_vectors vox2_dev)
  echo "$0: vox2_dev subset: ${n} vectors (the 'Failed' line above only"
  echo "$0: says that the count differs from 0, the count it was given)"
  if [ "${n}" -lt "${min_mean_utts}" ]; then
    echo "$0: only ${n} vox2_dev vectors (< ${min_mean_utts}), see ${vox2}/log" >&2
    exit 1
  fi
  echo "frac n=${n} shards=$(wc -l < "${eval_dir}/vox2_dev_frac.list")" > "${vox2}/MEAN_OK"
elif [ "${vox2_mean}" = "full" ]; then
  # every utterance, whole (cut to maxsec), as vox1
  extract vox2_dev "${data}/vox2_dev/shard.list" "$(wc -l < "${data}/vox2_dev/wav.scp")" 1
  if ! grep -q Successfully "${vox2}/extract.result"; then
    echo "$0: vox2_dev extraction incomplete, see ${vox2}/log" >&2
    exit 1
  fi
  echo "full n=$(count_vectors vox2_dev)" > "${vox2}/MEAN_OK"
elif [ -f "${vox2_mean}" ]; then
  mkdir -p "${vox2}"
  ln -sfn "$(readlink -f "${vox2_mean}")" "${vox2}/xvector.scp"
  echo "reused ${vox2_mean}" > "${vox2}/MEAN_OK"
else
  echo "$0: --vox2_mean must be frac, full or an xvector.scp" >&2
  exit 1
fi

echo "$0: embeddings in ${eval_dir}/embeddings"
