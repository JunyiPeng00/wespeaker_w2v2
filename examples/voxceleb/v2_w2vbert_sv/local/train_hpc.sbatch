#!/bin/bash
#SBATCH --job-name   w2vbert_train
##SBATCH --account    <your_account>   # set your Slurm account and remove one #
#SBATCH --partition  gpu
#SBATCH --gpus-per-node 8
#SBATCH --nodes      1
#SBATCH --time       2-00:00:00
#SBATCH --output     logs/%x_%j.out
#SBATCH --signal     B:TERM@600
# Production launcher for the two-stage W2V-BERT 2.0 + MHFA recipe on HPC.
# One job = one stage segment (<=48 h). Resumes from the latest epoch checkpoint
# in EXP/models, so the same command can be re-submitted (or chained with
# --dependency=afterany:<id>) until all epochs are done.
#
#   STAGE=1 TRAIN_SET=vox2_dev  EXP=exp/vox2_dev/stage1_frozen sbatch local/train_hpc.sbatch
#   STAGE=2 TRAIN_SET=vox2_dev  EXP=exp/vox2_dev/stage2_ft INIT=exp/vox2_dev/stage1_frozen/models/avg_model.pt \
#          sbatch local/train_hpc.sbatch
#
# TRAIN_SET : vox2_dev | <your set>   (data/<set>/shard.list + utt2spk)
# STAGE     : 1 (frozen frontend, conf/w2vbert_sv_mhfa_stage1_frozen.yaml)
#             2 (full FT,         conf/w2vbert_sv_mhfa_stage2_ft.yaml)
# INIT      : stage-2 only, the stage-1 averaged model (copied to models/model_0.pt on first start)
# CONFIG    : optional override of the yaml

source ~/.bashrc
conda activate "${CONDA_ENV:-w2v2}"
export TMPDIR=${TMPDIR:-/tmp}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1   # submit from examples/voxceleb/v2_w2vbert_sv
. ./path.sh

TRAIN_SET=${TRAIN_SET:-vox2_dev}
STAGE=${STAGE:-1}
EXP=${EXP:-exp/${TRAIN_SET}/stage${STAGE}}
if [ "$STAGE" = "1" ]; then CONFIG=${CONFIG:-conf/w2vbert_sv_mhfa_stage1_frozen.yaml}
else                        CONFIG=${CONFIG:-conf/w2vbert_sv_mhfa_stage2_ft.yaml}; fi
NGPU=${SLURM_GPUS_ON_NODE:-8}
GPUS="[$(seq -s, 0 $((NGPU-1)))]"
# lr is scaled by world_size*batch_size/64 inside wespeaker: refuse silently different effective lr
BS=$(grep -E '^\s*batch_size:' ${CONFIG:-/dev/null} 2>/dev/null | head -1 | awk '{print $2}')
if [ -n "$EXPECT_GLOBAL_BATCH" ] && [ -n "$BS" ] && [ $((NGPU*BS)) -ne "$EXPECT_GLOBAL_BATCH" ]; then
  echo "ABORT: world_size*batch_size=$((NGPU*BS)) != EXPECT_GLOBAL_BATCH=$EXPECT_GLOBAL_BATCH (lr scale_ratio would change)"; exit 1; fi
mkdir -p $EXP/models  # train.py tolerates a pre-created dir; it only refuses existing model_*.pt without --checkpoint
echo "== $(date) host=$(hostname) job=$SLURM_JOB_ID STAGE=$STAGE TRAIN_SET=$TRAIN_SET EXP=$EXP CONFIG=$CONFIG NGPU=$NGPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# resume logic: latest models/model_N.pt wins; stage-2 first start copies INIT to model_0.pt
CKPT=$(ls $EXP/models/model_[0-9]*.pt 2>/dev/null | sed 's/.*model_//; s/\.pt//' | sort -n | tail -1)
if [ -n "$CKPT" ]; then
  CKPT_ARG="--checkpoint $EXP/models/model_${CKPT}.pt"; echo "== resuming from epoch $CKPT"
elif [ "$STAGE" = "2" ]; then
  [ -f "$INIT" ] || { echo "STAGE=2 needs INIT=<stage1 avg_model.pt>"; exit 1; }
  cp $INIT $EXP/models/model_0.pt; CKPT_ARG="--checkpoint $EXP/models/model_0.pt"; echo "== stage-2 init from $INIT"
else
  CKPT_ARG=""; echo "== fresh start"
fi
NUM_EPOCHS=$(grep -E '^num_epochs:' $CONFIG | awk '{print $2}')
if [ -n "$CKPT" ] && [ "$CKPT" -ge "$NUM_EPOCHS" ]; then echo "== all $NUM_EPOCHS epochs done, nothing to do"; exit 0; fi

# SIGTERM (time limit - 10 min): the trainer saves per epoch only, so we just let it die
# cleanly; the next submission resumes from the last completed epoch.
trap 'echo "== $(date) SIGTERM received, job will be resumed from last epoch ckpt"; kill $TRAIN_PID; wait $TRAIN_PID' TERM

# Stage the MUSAN/RIR LMDBs (14 GB) on the node's RAM disk: streaming a large shard set (TBs per epoch) through the page cache
# evicts their mmap'd pages and every augmentation then faults 4 KB pages back from Lustre. tmpfs pages cannot be evicted. AUG_TO_SHM=0 disables.
REVERB=data/rirs/lmdb; NOISE=data/musan/lmdb
if [ "${AUG_TO_SHM:-1}" = "1" ]; then
  SHM=/dev/shm/aug_${SLURM_JOB_ID:-$$}; mkdir -p $SHM/rirs $SHM/musan
  cp data/rirs/lmdb/data.mdb $SHM/rirs/ && cp data/musan/lmdb/data.mdb $SHM/musan/ && REVERB=$SHM/rirs NOISE=$SHM/musan
  echo "== aug LMDBs staged to $SHM ($(du -sh $SHM | cut -f1))"; trap "rm -rf $SHM" EXIT
fi
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$NGPU \
  wespeaker/bin/train.py --config $CONFIG \
    --exp_dir $EXP --gpus "$GPUS" \
    --data_type shard \
    --train_data data/${TRAIN_SET}/shard.list \
    --train_label data/${TRAIN_SET}/utt2spk \
    --reverb_data $REVERB \
    --noise_data $NOISE \
    $CKPT_ARG &
TRAIN_PID=$!
wait $TRAIN_PID; RC=$?
echo "== $(date) train.py exit code $RC"; ls -la $EXP/models/
if [ $RC -eq 0 ] && [ -f $EXP/models/model_${NUM_EPOCHS}.pt ]; then
  NUM_AVG=${NUM_AVG:-$(grep -E '^num_avg:' $CONFIG | awk '{print $2}')}
  python wespeaker/bin/average_model.py --dst_model $EXP/models/avg_model.pt \
    --src_path $EXP/models --num ${NUM_AVG:-1} && echo "== averaged last ${NUM_AVG:-1} epochs -> $EXP/models/avg_model.pt"
fi
exit $RC
