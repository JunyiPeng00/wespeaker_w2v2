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

# OVSB recipe: W2V-BERT 2.0 (22 layers) + MHFA, two training stages.
#
#   stage 1  training list of stage 2 from the per-corpus lists        (CPU)
#   stage 2  stage-1 training: frozen encoder, stock wespeaker trainer (GPU)
#   stage 3  initial checkpoint: truncate stage 1, list imprint audio  (CPU)
#   stage 4  initial checkpoint: stage-1 embeddings of that audio      (GPU)
#   stage 5  initial checkpoint: classifier rows for all classes       (CPU)
#   stage 6  stage-2 training: full fine-tuning, recipe trainer        (GPU)
#   stage 7  average the last checkpoints of stage 2                   (CPU)
#   stage 8  VoxCeleb1 evaluation                                      (GPU)
#
# The CPU stages run here. The GPU stages print the command to run inside a
# GPU job or to submit; this script submits nothing.
#
#   bash run.sh --stage 1 --stop_stage 1
#
# Inputs that are not built here: the per-corpus shards, utt2spk files and
# tar indexes named in the site file of stage 1
# (conf/data/site_paths.example.yaml; local/data/build_tar_index.sh builds
# an index), the stage-1 list data/stage1, the noise / reverb LMDBs and the
# evaluation data under data/, and the pretrained encoder in
# assets/w2v-bert-2.0.

. ./path.sh || exit 1

stage=-1
stop_stage=-1

list_spec=conf/data/s3c.yaml
site_paths=conf/data/site_paths.yaml      # copy of site_paths.example.yaml
train_dir=data/train                      # training list of stage 2

stage1_config=conf/stage1_frozen.yaml
stage1_exp=exp/stage1_frozen
stage1_label=data/stage1/utt2spk
stage1_num_avg=3
gpus="[0,1,2,3,4,5,6,7]"

init_dir=exp/init

stage2_config=conf/stage2_v9c.yaml
stage2_exp=exp/stage2_v9c
site_env=local/slurm/site.env             # copy of site.env.example

eval_tag=final
eval_maxsec=10                            # 10: development, 60: reported

. tools/parse_options.sh || exit 1
set -o pipefail

num_gpus=$(echo "${gpus}" | awk -F ',' '{print NF}')
sbatch_options="--account <account> --partition <partition>"
init_options="--stage1_exp ${stage1_exp} --stage1_label ${stage1_label}"
init_options="${init_options} --train_label ${train_dir}/utt2spk"
init_options="${init_options} --train_index ${train_dir}/tar_index"
init_options="${init_options} --out_dir ${init_dir}"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: training list ${train_dir} ..."
  python local/data/build_train_list.py --spec "${list_spec}" \
    --site "${site_paths}" --out "${train_dir}" || exit 1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: stage-1 training (frozen encoder)."
  echo "Run in a job with ${num_gpus} GPUs (to resume, add" \
    "--checkpoint ${stage1_exp}/models/model_<N>.pt):"
  echo "  python -m torch.distributed.run --standalone --nnodes=1" \
    "--nproc_per_node=${num_gpus} wespeaker/bin/train.py" \
    "--config ${stage1_config} --exp_dir ${stage1_exp} --gpus \"${gpus}\"" \
    "--num_avg ${stage1_num_avg} --data_type shard" \
    "--train_data $(dirname "${stage1_label}")/shard.list" \
    "--train_label ${stage1_label}" \
    "--reverb_data data/rirs/lmdb --noise_data data/musan/lmdb"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: initial checkpoint, steps 1 and 2 ..."
  # shellcheck disable=SC2086
  bash local/init/build_init.sh --stage 1 --stop_stage 2 ${init_options} \
    || exit 1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: initial checkpoint, step 3. Run in a job with one GPU:"
  echo "  bash local/init/build_init.sh --stage 3 --stop_stage 3" \
    "${init_options} --gpus \"[0]\""
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: initial checkpoint, step 4 ..."
  # shellcheck disable=SC2086
  bash local/init/build_init.sh --stage 4 --stop_stage 4 ${init_options} \
    || exit 1
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: stage-2 training. Checking ${stage2_config} ..."
  python -m ovsb.bin.validate_config "${stage2_config}" --view none || exit 1
  if [ ! -f "${site_env}" ]; then
    echo "First copy local/slurm/site.env.example to ${site_env} and set" \
      "EXP=${stage2_exp} CONFIG=${stage2_config}" \
      "INIT=${init_dir}/stage2_init.pt in it."
  fi
  echo "Check the settings without a job:"
  echo "  OVSB_DRY_RUN=1 OVSB_SITE_ENV=${site_env}" \
    "bash local/slurm/train_segment.sbatch"
  echo "Submit, and again after every job until the run is finished" \
    "(local/slurm/README.md):"
  echo "  sbatch ${sbatch_options} --time <limit> --nodes 1" \
    "--gpus-per-node ${num_gpus} --signal B:TERM@300" \
    "--output logs/%x_%j.out --export=ALL,OVSB_SITE_ENV=${site_env}" \
    "local/slurm/train_segment.sbatch"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  plan=$(python -m ovsb.bin.train --config "${stage2_config}" \
    --set "run.exp_dir=${stage2_exp}" --plan) || exit 1
  eval "${plan}"
  echo "Stage 7: average of the last ${PLAN_NUM_AVG} checkpoint(s) of" \
    "${stage2_exp} ..."
  if [ ! -e "${stage2_exp}/models/final_model.pt" ]; then
    echo "${stage2_exp}: the run is not finished (epoch" \
      "${PLAN_EPOCH_DONE} of ${PLAN_NUM_EPOCHS})" >&2
    exit 1
  fi
  python wespeaker/bin/average_model.py \
    --dst_model "${stage2_exp}/models/avg_model.pt" \
    --src_path "${stage2_exp}/models" --num "${PLAN_NUM_AVG}" || exit 1
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  model=${stage2_exp}/models/avg_model.pt
  echo "Stage 8: VoxCeleb1 evaluation of ${model}. Submit:"
  echo "  sbatch ${sbatch_options} --time 1:00:00 --gpus 1" \
    "--output logs/%x_%j.out" \
    "--export=ALL,EXP=${stage2_exp},TAG=${eval_tag},MAXSEC=${eval_maxsec},MODEL=${model}" \
    "local/slurm/eval.sbatch"
fi
