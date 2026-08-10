#!/bin/bash

# Copyright (c) 2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Recreate assets/ from public sources (run from the example dir).
#
#   bash local/prepare_assets.sh            # base model + released checkpoint
#   CKPT_REPO= bash local/prepare_assets.sh # base model only (skip checkpoint)
#
# Base model: official facebook/w2v-bert-2.0 (this recipe's copy is
# bit-identical, sha256 eb890c96...); conf/config_baseline.json only ADDS
# pruning fields on top of the official config.json and ships in git.
# Released checkpoint: JYP2024/wespeaker-mhfa-W2VB-ft-v2 — a full exp upload
# (avg_model.pt + vox1/vox2_dev reference embeddings), public on the HF Hub.

set -euo pipefail

CKPT_REPO=${CKPT_REPO-JYP2024/wespeaker-mhfa-W2VB-ft-v2}
CKPT_DIR=${CKPT_DIR:-assets/mhfa_W2VB_ft_v2}

mkdir -p assets/w2v-bert-2.0

if [ ! -f assets/w2v-bert-2.0/model.safetensors ]; then
  echo "Downloading facebook/w2v-bert-2.0 (base model) ..."
  python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    'facebook/w2v-bert-2.0',
    local_dir='assets/w2v-bert-2.0',
    allow_patterns=['config.json', 'model.safetensors',
                    'preprocessor_config.json'])
EOF
fi

cp conf/config_baseline.json assets/w2v-bert-2.0/config_baseline.json
echo "Base model ready: assets/w2v-bert-2.0"

if [ -n "${CKPT_REPO}" ]; then
  echo "Downloading released checkpoint + reference embeddings from ${CKPT_REPO} ..."
  python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download('${CKPT_REPO}', local_dir='${CKPT_DIR}',
                  allow_patterns=['models/avg_model.pt', 'embeddings/*'])
EOF
  # Reference scp files carry the ark paths of the machine that produced
  # them; repoint them at this checkout.
  ckpt_abs=$(cd "${CKPT_DIR}" && pwd)
  for f in "${CKPT_DIR}"/embeddings/*/*.scp*; do
    [ -f "$f" ] && sed -i "s|^\(\S\+\) .*/embeddings/|\1 ${ckpt_abs}/embeddings/|" "$f"
  done
  # Inference config in this repo's format (frontend key w2vbert_sv).
  cp conf/w2vbert_sv_mhfa.yaml "${CKPT_DIR}"/config.yaml
  echo "Checkpoint ready: ${CKPT_DIR}"
else
  echo "CKPT_REPO empty: skip checkpoint (train your own via run.sh stages 1-4)"
fi

# Official VoxCeleb1 trial lists (needed by the scoring stage; same source
# and conversion as examples/voxceleb/v2/local/prepare_data.sh).
if [ ! -f data/vox1/trials/vox1_O_cleaned.kaldi ]; then
  echo "Downloading official VoxCeleb1 trial lists ..."
  mkdir -p data/vox1/trials
  meta=https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta
  for pair in "veri_test2.txt vox1_O" "list_test_all2.txt vox1_E" \
              "list_test_hard2.txt vox1_H"; do
    set -- $pair
    wget -q --no-check-certificate "${meta}/$1" -O "data/vox1/trials/$2.txt"
    awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' \
      "data/vox1/trials/$2.txt" > "data/vox1/trials/${2}_cleaned.kaldi"
  done
fi
echo "Trials ready: data/vox1/trials"
