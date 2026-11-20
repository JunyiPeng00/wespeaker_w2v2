#!/bin/bash
#SBATCH --job-name ovsb_eval
#
# VoxCeleb1 evaluation of one checkpoint (local/eval/README.md): extraction
# on the GPUs of the job, then cosine scoring on its CPUs. Submit from the
# recipe directory; account, partition, time limit and GPUs go on the
# command line:
#
#   sbatch --account <account> --partition <partition> --time 1:00:00 \
#     --gpus 1 --output logs/%x_%j.out \
#     --export=ALL,EXP=exp/stage2_v9c,TAG=ep14,MODEL=exp/stage2_v9c/models/model_14.pt \
#     local/slurm/eval.sbatch
#
# Environment:
#   EXP         directory with the model's config.yaml          (required)
#   MODEL       checkpoint                                       (required)
#   TAG         name of the evaluation                           (eval)
#   MAXSEC      input cap in seconds, 10 or 60                   (10)
#   VOX2_MEAN   frac | full | an existing vox2 xvector.scp       (frac)
#   DATA        data directory                                   (data)
#   OVSB_CONDA_ENV  conda environment to activate                (none)
#   OVSB_MODULES    environment modules to load, space separated (none)
# The resulting directory is EXP/eval_<TAG>_tf32_max<MAXSEC>s. A job that is
# resubmitted continues: finished extractions are kept.

set -eo pipefail

if [ -n "${OVSB_MODULES}" ]; then
  for module_name in ${OVSB_MODULES}; do
    module load "${module_name}"
  done
fi
if [ -n "${OVSB_CONDA_ENV}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${OVSB_CONDA_ENV}"
fi
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-.}"
. ./path.sh

MAXSEC=${MAXSEC:-10}
TAG=${TAG:-eval}
DATA=${DATA:-data}
VOX2_MEAN=${VOX2_MEAN:-frac}
if [ ! -f "${MODEL}" ] || [ ! -f "${EXP}/config.yaml" ]; then
  echo "need EXP (with config.yaml) and MODEL" >&2
  exit 1
fi
num_gpus=${SLURM_GPUS_ON_NODE:-1}
gpus="[$(seq -s, 0 $((num_gpus - 1)))]"
eval_dir=${EXP}/eval_${TAG}_tf32_max${MAXSEC}s

echo "== $(date) job=${SLURM_JOB_ID:-none} EXP=${EXP} MODEL=${MODEL}"
echo "== TAG=${TAG} MAXSEC=${MAXSEC} VOX2_MEAN=${VOX2_MEAN} gpus=${gpus}"
bash local/eval/extract.sh --exp_dir "${EXP}" --model "${MODEL}" \
  --tag "${TAG}" --maxsec "${MAXSEC}" --data "${DATA}" --gpus "${gpus}" \
  --vox2_mean "${VOX2_MEAN}" --eval_dir "${eval_dir}"
bash local/eval/score.sh --eval_dir "${eval_dir}" --data "${DATA}"
echo "== $(date) evaluation done: ${eval_dir}/scores"
