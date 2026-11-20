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

# Initial checkpoint of stage 2, in one pass (see local/init/README.md):
#
#   stage 1  cut the last stage-1 checkpoint to 22 encoder layers      (CPU)
#   stage 2  list audio of the training speakers without a head row    (CPU)
#   stage 3  stage-1 embeddings of that audio                          (GPU)
#   stage 4  head rows for every class of the training list            (CPU)
#
# Result: <out_dir>/stage2_init.pt and stage2_init.pt.head.json.
# Run from the recipe directory:
#
#   bash local/init/build_init.sh --stage 1 --stop_stage 2
#   bash local/init/build_init.sh --stage 3 --stop_stage 3   # on a GPU node
#   bash local/init/build_init.sh --stage 4 --stop_stage 4

stage=1
stop_stage=4

stage1_exp=exp/stage1_frozen              # config.yaml, models/
stage1_ckpt=                              # default <stage1_exp>/models/final_model.pt
stage1_label=data/stage1/utt2spk          # list stage 1 was trained on
train_label=data/train/utt2spk            # training list of stage 2
train_index=data/train/tar_index
out_dir=exp/init
layers=22
rows_per_class=3
per_spk=20                                # utterances per speaker
maxsec=10                                 # seconds per utterance
gpus="[0]"

. ./path.sh
. tools/parse_options.sh
set -eo pipefail

if [ -z "${stage1_ckpt}" ]; then
  stage1_ckpt=${stage1_exp}/models/final_model.pt   # the last epoch
fi
truncated=${out_dir}/stage1_L${layers}.pt

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: keep ${layers} encoder layers of ${stage1_ckpt} ..."
  python local/init/truncate_ckpt.py "${stage1_ckpt}" "${layers}" \
    "${truncated}" --train-label "${stage1_label}" \
    --rows-per-class "${rows_per_class}"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: audio of the speakers without a head row ..."
  python local/init/imprint_head.py list --utt2spk "${train_label}" \
    --index "${train_index}" --old_utt2spk "${stage1_label}" \
    --out "${out_dir}/imprint" --per_spk "${per_spk}" --max_sec "${maxsec}"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: stage-1 embeddings of that audio ..."
  bash local/init/imprint_extract.sh --config "${stage1_exp}/config.yaml" \
    --model "${truncated}" --list_dir "${out_dir}/imprint" \
    --out_dir "${out_dir}/imprint_emb" --layers "${layers}" \
    --maxsec "${maxsec}" --gpus "${gpus}"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: head for the classes of ${train_label} ..."
  python local/init/imprint_head.py imprint --init "${truncated}" \
    --old_utt2spk "${stage1_label}" --new_utt2spk "${train_label}" \
    --emb_scp "${out_dir}/imprint_emb/embeddings/imprint/xvector.scp" \
    --emb_utt2spk "${out_dir}/imprint/utt2spk" \
    --K "${rows_per_class}" --frontend_num_layers "${layers}" \
    --out "${out_dir}/stage2_init.pt"
  echo "Initial checkpoint: ${out_dir}/stage2_init.pt"
fi
