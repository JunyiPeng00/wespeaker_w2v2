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

# Cosine scoring after subtraction of the VoxCeleb2-dev mean, then EER and
# minDCF (p_target 0.01 and 0.05) per trial list. No score normalisation,
# no calibration. CPU. Run from the recipe directory:
#
#   bash local/eval/score.sh --eval_dir exp/stage2_v9c/eval_ep14_tf32_max10s
#
# Writes <eval_dir>/scores/<trial list>.score, vox1_cos_result (p 0.01) and
# vox1_cos_result_p005 (p 0.05).

eval_dir=           # directory written by local/eval/extract.sh
data=data
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
mean_dir=           # default <eval_dir>/embeddings/vox2_dev (needs xvector.scp)

. ./path.sh
. tools/parse_options.sh
set -eo pipefail

if [ -z "${eval_dir}" ] || [ ! -f "${eval_dir}/embeddings/vox1/xvector.scp" ]; then
  echo "$0: need --eval_dir with embeddings/vox1/xvector.scp" >&2
  exit 1
fi
if [ -z "${mean_dir}" ]; then
  mean_dir=${eval_dir}/embeddings/vox2_dev
fi
if [ ! -f "${mean_dir}/xvector.scp" ]; then
  # extract_embedding.sh joins the per-process lists only when it finishes
  cat "${mean_dir}"/xvector_*.scp > "${mean_dir}/xvector.scp"
fi
if [ ! -s "${mean_dir}/xvector.scp" ]; then
  echo "$0: no embeddings for the mean in ${mean_dir}" >&2
  exit 1
fi

trial_files=()
for x in ${trials}; do
  trial_files+=("${data}/vox1/trials/${x}")
done

scores=${eval_dir}/scores
mkdir -p "${scores}"
python wespeaker/bin/score_fast.py --exp_dir "${eval_dir}" \
  --eval_scp_path "${eval_dir}/embeddings/vox1/xvector.scp" \
  --cal_mean True --cal_mean_dir "${mean_dir}" "${trial_files[@]}"

rm -f "${scores}/vox1_cos_result" "${scores}/vox1_cos_result_p005"
for x in ${trials}; do
  # positional arguments: p_target c_miss c_fa score files
  python wespeaker/bin/compute_metrics.py 0.01 1 1 "${scores}/${x}.score" \
    >> "${scores}/vox1_cos_result" 2>&1
  python wespeaker/bin/compute_metrics.py 0.05 1 1 "${scores}/${x}.score" \
    >> "${scores}/vox1_cos_result_p005" 2>&1
done

echo "== cosine after vox2 mean subtraction, p_target 0.01:"
grep -E "EER|minDCF" "${scores}/vox1_cos_result"
echo "== p_target 0.05:"
grep -E "EER|minDCF" "${scores}/vox1_cos_result_p005"
