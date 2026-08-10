#!/bin/bash

# Copyright (c) 2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Recreate assets/ from public sources (run from the example dir).
#
#   bash local/prepare_assets.sh                       # base model only
#   CKPT_REPO=<hf_repo_id> bash local/prepare_assets.sh  # + fine-tuned checkpoint
#
# The base model is the official facebook/w2v-bert-2.0 (the local
# model.safetensors must hash to sha256 eb890c9660ed6e34...93be043);
# config_baseline.json only ADDS pruning fields on top of the official
# config.json and ships in conf/. The fine-tuned mhfa_W2VB_ft_v5 checkpoint
# is a training artifact: download it from CKPT_REPO if provided, otherwise
# reproduce it with the two-stage training (run.sh stages 1-4).

set -euo pipefail

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

if [ -n "${CKPT_REPO:-}" ]; then
  echo "Downloading fine-tuned checkpoint from ${CKPT_REPO} ..."
  mkdir -p assets/mhfa_W2VB_ft_v5/models
  python - <<EOF
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download('${CKPT_REPO}', 'avg_model.pt')
shutil.copy(p, 'assets/mhfa_W2VB_ft_v5/models/avg_model.pt')
EOF
  cp conf/w2vbert_sv_mhfa.yaml assets/mhfa_W2VB_ft_v5/config.yaml
  echo "Checkpoint ready: assets/mhfa_W2VB_ft_v5"
else
  echo "CKPT_REPO not set: skip fine-tuned checkpoint (train it via run.sh stages 1-4)"
fi
