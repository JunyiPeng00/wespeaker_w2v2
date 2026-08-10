#!/bin/bash

# Copyright (c) 2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Two-stage W2V-BERT 2.0 + MHFA speaker verification recipe.
#   stage 1: train with frozen frontend        (conf/w2vbert_sv_mhfa_stage1_frozen.yaml)
#   stage 2: average stage-1 models
#   stage 3: full fine-tuning, initialized from the stage-1 average
#   stage 4: average stage-2 models
#   stage 5: extract vox2_dev + vox1 embeddings
#   stage 6: cosine scoring + EER/minDCF on vox1-O/E/H
#
# Raw audio can be read from disk (default WeSpeaker behavior) or from LMDBs
# holding the original wav bytes keyed by the raw.list "wav" value; set
# use_lmdb=false if your raw.list points at real files on disk.

. ./path.sh || exit 1

stage=-1
stop_stage=-1

data=<path> original/examples/voxceleb/v4/data
data_type="raw"
use_lmdb=true  # read raw audio from ${data}/<set>/lmdb instead of the disk

stage1_config=conf/w2vbert_sv_mhfa_stage1_frozen.yaml
stage1_exp=exp/v2_w2vbert_sv/stage1_frozen
stage2_config=conf/w2vbert_sv_mhfa_stage2_ft.yaml
stage2_exp=exp/v2_w2vbert_sv/stage2_ft

gpus="[0,1,2,3,4,5,6,7]"
stage1_num_avg=2
stage2_num_avg=1
checkpoint=

trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"

. tools/parse_options.sh || exit 1

train_lmdb=
vox2_lmdb=
vox1_lmdb=
if [ "$use_lmdb" == "true" ]; then
  train_lmdb=${data}/vox2_dev/lmdb
  vox2_lmdb=${data}/vox2_dev/lmdb
  vox1_lmdb=${data}/vox1/lmdb
fi

num_gpus=$(echo $gpus | awk -F ',' '{print NF}')

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: train with frozen frontend ..."
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train.py --config $stage1_config \
      --exp_dir ${stage1_exp} \
      --gpus $gpus \
      --num_avg ${stage1_num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/vox2_dev/${data_type}.list \
      --train_label ${data}/vox2_dev/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      ${train_lmdb:+--train_lmdb ${train_lmdb}} \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: average stage-1 models ..."
  python wespeaker/bin/average_model.py \
    --dst_model ${stage1_exp}/models/avg_model.pt \
    --src_path ${stage1_exp}/models \
    --num ${stage1_num_avg}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: full fine-tuning from the stage-1 average ..."
  mkdir -p ${stage2_exp}/models
  cp ${stage1_exp}/models/avg_model.pt ${stage2_exp}/models/model_0.pt
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train.py --config $stage2_config \
      --exp_dir ${stage2_exp} \
      --gpus $gpus \
      --num_avg ${stage2_num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/vox2_dev/${data_type}.list \
      --train_label ${data}/vox2_dev/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      ${train_lmdb:+--train_lmdb ${train_lmdb}} \
      --checkpoint ${stage2_exp}/models/model_0.pt
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: average stage-2 models ..."
  python wespeaker/bin/average_model.py \
    --dst_model ${stage2_exp}/models/avg_model.pt \
    --src_path ${stage2_exp}/models \
    --num ${stage2_num_avg}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: extract vox2_dev + vox1 embeddings ..."
  local/extract_vox.sh \
    --exp_dir ${stage2_exp} \
    --model_path ${stage2_exp}/models/avg_model.pt \
    --nj 8 --gpus $gpus --data_type ${data_type} --data ${data} \
    --vox2_lmdb "${vox2_lmdb}" --vox1_lmdb "${vox1_lmdb}"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir ${stage2_exp} \
    --trials "$trials"
fi
