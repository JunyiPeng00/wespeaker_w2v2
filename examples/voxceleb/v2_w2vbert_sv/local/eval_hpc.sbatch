#!/bin/bash
#SBATCH --job-name   eval_vox1
##SBATCH --account    <your_account>   # set your Slurm account and remove one #
#SBATCH --partition  gpu_short
#SBATCH --gpus-per-node 2
#SBATCH --nodes      1
#SBATCH --time       1:00:00
#SBATCH --output     logs/%x_%j.out
# Frozen evaluation protocol: fp32 extraction, whole utterance truncated to MAXSEC, vox1-O/E/H cleaned trials,
# cosine + vox2_dev mean-norm.
#   EXP=<exp dir with config.yaml>  MODEL=<checkpoint .pt>  TAG=<name>  MAXSEC=10|60
#   VOX2_SCP=<existing vox2_dev xvector.scp to reuse for the mean>  (omit to extract vox2_dev afresh, 4 s crops, batch 16)

source ~/.bashrc
conda activate "${CONDA_ENV:-w2v2}"
export TMPDIR=${TMPDIR:-/tmp}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1   # submit from examples/voxceleb/v2_w2vbert_sv
. ./path.sh
MAXSEC=${MAXSEC:-10}; TAG=${TAG:-eval}
NG=${SLURM_GPUS_ON_NODE:-8}; GL="[$(seq -s, 0 $((NG-1)))]"   # works with --gpus N as well as a full node
[ -f "$MODEL" ] && [ -f "$EXP/config.yaml" ] || { echo "need EXP (with config.yaml) and MODEL"; exit 1; }
# Default protocol: TF32 + masked length-sorted batching (320 s budget),
# 2 GPUs on gpu_short. Old protocol: EXTRACT_DTYPE=fp32 EXTRACT_BATCH_SEC=0 (keeps the old, untagged dir names).
EXTRACT_DTYPE=${EXTRACT_DTYPE:-tf32}; EXTRACT_BATCH_SEC=${EXTRACT_BATCH_SEC:-320}; export EXTRACT_DTYPE EXTRACT_BATCH_SEC
EXT=""; if [ "$EXTRACT_DTYPE" != "fp32" ] || [ "$EXTRACT_BATCH_SEC" != "0" ]; then
  EXT="_${EXTRACT_DTYPE}-$([ "$EXTRACT_BATCH_SEC" != "0" ] && echo rag || echo b1)"; fi   # protocol id in the eval dir name
E=$EXP/eval_${TAG}${EXT}_max${MAXSEC}s; mkdir -p $E
python - "$EXP/config.yaml" "$E/config.yaml" "$MAXSEC" <<'PY'
import os, sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
fe = c['dataset_args'].get('frontend', 'w2vbert_sv')   # w2vbert_sv | omniasr_sv
c['dataset_args'][fe + '_args']['max_input_seconds'] = float(sys.argv[3])
if fe == 'w2vbert_sv':
    c['dataset_args'][fe + '_args'].setdefault('feature_extractor', 'torch')
for k in ['extract_amp', 'extract_tf32']: c[k] = False   # frozen protocol: fp32
# EXTRACT_DTYPE fp32 (protocol) | tf32 | bf16 ; EXTRACT_BATCH_SEC > 0 = masked length-sorted batching of whole utterances
dt = os.environ.get('EXTRACT_DTYPE', 'tf32')
if dt == 'tf32': c['extract_tf32'] = True
elif dt == 'bf16': c['extract_amp'] = True; c['extract_amp_dtype'] = 'bfloat16'
elif dt != 'fp32': sys.exit('EXTRACT_DTYPE must be fp32|tf32|bf16')
c['extract_batch_seconds'] = float(os.environ.get('EXTRACT_BATCH_SEC', '320'))
c['dataset_args'].pop('num_frms_choices', None)   # vox2-mean crops: the config's fixed num_frms, never the training crop mix
yaml.safe_dump(c, open(sys.argv[2], 'w'))
PY
echo "== $(date) job=$SLURM_JOB_ID EXP=$EXP MODEL=$MODEL TAG=$TAG MAXSEC=$MAXSEC EXTRACT_DTYPE=${EXTRACT_DTYPE:-fp32} EXTRACT_BATCH_SEC=${EXTRACT_BATCH_SEC:-0} VOX2_SCP=${VOX2_SCP:-<extract>}"
T0=$(date +%s)
bash tools/extract_embedding.sh --exp_dir $E --model_path $MODEL --data_type shard \
  --data_list data/vox1/shard.list --wavs_num $(wc -l < data/vox1/wav.scp) --store_dir vox1 \
  --batch_size 1 --num_workers 2 --nj $NG --gpus "$GL"
echo "== vox1 extraction $(( $(date +%s) - T0 )) s"; cat $E/embeddings/vox1/extract.result
if [ -n "$VOX2_SCP" ]; then
  mkdir -p $E/embeddings/vox2_dev && ln -sfn $(readlink -f $VOX2_SCP) $E/embeddings/vox2_dev/xvector.scp
else
  T1=$(date +%s)
  # VOX2_FRAC (0<f<=1): use every 1/f-th vox2_dev shard for the mean vector (f=0.1 -> ~109k utts, enough for a
  # 256-d mean; cuts the eval from ~80 to ~25 min). Full set remains the default.
  VOX2_FRAC=${VOX2_FRAC:-1}
  V2LIST=data/vox2_dev/shard.list; V2N=$(wc -l < data/vox2_dev/wav.scp)
  if [ "$VOX2_FRAC" != "1" ]; then
    V2LIST=$E/vox2_dev_frac.list; awk -v f=$VOX2_FRAC 'BEGIN{s=int(1/f)} NR%s==1' data/vox2_dev/shard.list > $V2LIST
    V2N=0; echo "== vox2_dev mean from $(wc -l < $V2LIST) shards (VOX2_FRAC=$VOX2_FRAC)"
  fi
  bash tools/extract_embedding.sh --exp_dir $E --model_path $MODEL --data_type shard \
    --data_list $V2LIST --wavs_num $V2N --store_dir vox2_dev \
    --batch_size ${VOX2_BS:-64} --num_workers 4 --nj $NG --gpus "$GL"
  echo "== vox2_dev extraction $(( $(date +%s) - T1 )) s"; cat $E/embeddings/vox2_dev/extract.result
fi
grep -q Successfully $E/embeddings/vox1/extract.result || { echo "EVAL FAIL: vox1 extraction incomplete"; exit 1; }
mkdir -p $E/scores
# score_fast.py = vectorised score.py (one load, all three lists; ~20-35 s instead of ~530 s; identical EER/minDCF,
# printed scores differ by at most 1e-5 through float32 summation order)
python wespeaker/bin/${SCORER:-score_fast}.py --exp_dir $E --eval_scp_path $E/embeddings/vox1/xvector.scp \
  --cal_mean True --cal_mean_dir $E/embeddings/vox2_dev \
  data/vox1/trials/vox1_O_cleaned.kaldi data/vox1/trials/vox1_E_cleaned.kaldi data/vox1/trials/vox1_H_cleaned.kaldi > /dev/null
for x in vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi; do
  # compute_metrics.py: fire with *scores_files -> positional form (p_target c_miss c_fa files...)
  python wespeaker/bin/compute_metrics.py 0.01 1 1 $E/scores/$x.score 2>&1 | tee -a $E/scores/vox1_cos_result
  python wespeaker/bin/compute_metrics.py 0.05 1 1 $E/scores/$x.score 2>&1 | sed 's/^/[p=0.05] /' | tee -a $E/scores/vox1_cos_result_p005
done
echo "== RESULT $TAG max${MAXSEC}s (raw cosine + vox2 mean-norm):"; grep -E "EER|minDCF" $E/scores/vox1_cos_result | paste - - | awk '{print}'
echo "== $(date) EVAL DONE wall $(( $(date +%s) - T0 )) s"
