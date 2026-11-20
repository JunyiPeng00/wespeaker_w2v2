#!/bin/bash
#SBATCH --job-name ovsb_train
#
# One segment of a stage-2 training run (local/slurm/README.md): starts or
# resumes the run in EXP, trains at most EPOCHS_PER_SEG epochs, checks the
# encoder drift of every finished epoch, and stops cleanly before the time
# limit. Submit from the recipe directory; account, partition, time limit,
# GPUs and the early signal go on the command line:
#
#   sbatch --account <account> --partition <partition> --time 2-00:00:00 \
#     --nodes 1 --gpus-per-node 8 --signal B:TERM@300 \
#     --output logs/%x_%j.out local/slurm/train_segment.sbatch
#
# Submitting the same command again continues the run.
#
# Settings are environment variables, read from the site file
# (OVSB_SITE_ENV, default local/slurm/site.env; see site.env.example) and
# from the environment of the job. All of these must be set:
#   EXP                  run directory                (run.exp_dir)
#   CONFIG               recipe config
#   INIT                 initial checkpoint           (model.init.checkpoint)
#   EPOCHS_PER_SEG       epochs per job
#   DRIFT_REF            reference checkpoint of the drift gate
#   DRIFT_GATE           largest allowed relative encoder drift
#   EXPECT_GLOBAL_BATCH  GPUs x batch size the config was written for
#   AUG_TO_SHM           1: copy the noise and reverb LMDBs to /dev/shm
#   TMPDIR               directory for temporary files
# Optional:
#   OVSB_CONDA_ENV       conda environment to activate
#   OVSB_MODULES         environment modules to load, space separated
#   OVSB_POLL_SEC        seconds between two looks at EXP/models (20)
#   OVSB_DRY_RUN         1: check everything, print the commands that
#                        would copy, train and gate, run none of them
#
# Exit status: 0 the segment ended as intended (epochs done, or stopped by
# the time-limit signal with its state saved); 3 the drift gate is
# exceeded; 2 the drift gate could not be evaluated; 1 a setting is wrong;
# otherwise the exit status of the trainer.

set -o pipefail

die() {
  echo "ABORT: $*" >&2
  exit 1
}

cd "${SLURM_SUBMIT_DIR:-.}" || die "cannot enter the submit directory"
site_env=${OVSB_SITE_ENV:-local/slurm/site.env}
if [ -f "${site_env}" ]; then
  # shellcheck disable=SC1090
  . "${site_env}"
fi
missing=
for name in EXP CONFIG INIT EPOCHS_PER_SEG DRIFT_REF DRIFT_GATE \
    EXPECT_GLOBAL_BATCH AUG_TO_SHM TMPDIR; do
  if [ -z "${!name}" ]; then
    missing="${missing} ${name}"
  fi
done
[ -z "${missing}" ] || die "not set:${missing} (site file: ${site_env})"
export TMPDIR

if [ -n "${OVSB_MODULES}" ]; then
  for module_name in ${OVSB_MODULES}; do
    module load "${module_name}"
  done
fi
if [ -n "${OVSB_CONDA_ENV}" ]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${OVSB_CONDA_ENV}"
fi
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
. ./path.sh

dry_run=${OVSB_DRY_RUN:-0}
poll_sec=${OVSB_POLL_SEC:-20}
run() {  # a command that copies, trains or gates
  if [ "${dry_run}" = "1" ]; then
    echo "+ $*"
  else
    "$@"
  fi
}

# ---------------------------------------------------------------- checks
[ -f "${CONFIG}" ] || die "CONFIG ${CONFIG} does not exist"
# the gate fails closed: no reference, no training
[ -f "${DRIFT_REF}" ] || die "DRIFT_REF ${DRIFT_REF} does not exist"
num_gpus=${SLURM_GPUS_ON_NODE:-1}
gpus="[$(seq -s, 0 $((num_gpus - 1)))]"
overrides=(--set "run.exp_dir=${EXP}" --set "run.gpus=${gpus}"
           --set "model.init.checkpoint=${INIT}")
plan=$(python -m ovsb.bin.train --config "${CONFIG}" "${overrides[@]}" --plan) \
  || die "the trainer refused ${CONFIG}"
eval "${plan}"
# the learning rate is scaled by GPUs x batch size / 64 and the steps per
# epoch are divided by it: another number of GPUs is another schedule
if [ "${PLAN_GLOBAL_BATCH}" -ne "${EXPECT_GLOBAL_BATCH}" ]; then
  die "${num_gpus} GPU(s) x batch ${PLAN_BATCH_SIZE} = ${PLAN_GLOBAL_BATCH}," \
    "EXPECT_GLOBAL_BATCH is ${EXPECT_GLOBAL_BATCH}"
fi
if [ "${PLAN_SAVE_OPTIMIZER_STATE}" != "1" ]; then
  die "engine.save_optimizer_state must be true: train_state_<N>.pt marks" \
    "a finished epoch"
fi
if [ "${PLAN_RESUME_KIND}" = "fresh" ] && [ ! -f "${INIT}" ]; then
  die "INIT ${INIT} does not exist"
fi

drift_gate() {  # 0 pass, 3 exceeded, anything else: not evaluated
  run python local/drift_gate.py --exp "${EXP}" --ref "${DRIFT_REF}" \
    --gate "${DRIFT_GATE}"
}
gate_failed() {  # $1: exit status of the gate, $2: when
  if [ "$1" -eq 3 ]; then
    echo "== $(date) DRIFT GATE EXCEEDED $2 (gate ${DRIFT_GATE}," \
      "see ${EXP}/drift.tsv): the run is stopped"
  else
    echo "== $(date) DRIFT GATE ERROR $2 (exit status $1): the drift" \
      "could not be measured, the run is stopped"
  fi
}

echo "== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} EXP=${EXP}" \
  "CONFIG=${CONFIG} gpus=${gpus}"
echo "== ${PLAN_RESUME_KIND}: next epoch ${PLAN_NEXT_EPOCH} step" \
  "${PLAN_NEXT_STEP} of ${PLAN_NUM_EPOCHS} epochs, ${EPOCHS_PER_SEG}" \
  "epoch(s) per job, drift gate ${DRIFT_GATE} against ${DRIFT_REF}"
if [ "${PLAN_EPOCH_DONE}" -ge 1 ]; then
  drift_gate
  gate_rc=$?
  if [ ${gate_rc} -ne 0 ]; then
    gate_failed ${gate_rc} "before training"
    exit ${gate_rc}
  fi
fi
if [ "${PLAN_NEXT_EPOCH}" -gt "${PLAN_NUM_EPOCHS}" ]; then
  echo "== all ${PLAN_NUM_EPOCHS} epochs are done, nothing to do"
  exit 0
fi

# ------------------------------------------------- stop before the limit
# The scheduler sends SIGTERM to this shell only (--signal B:TERM@<sec>).
# The shell sends SIGUSR1 to the training processes, which finish the
# current step, save a partial state and exit. They are the children of the
# torchrun process; torchrun itself must not get the signal: it would pass
# a SIGTERM on and kill the processes 30 seconds later, in the middle of
# the save. SIGUSR1 is ignored from here on, so that a process that has not
# installed its handler yet ignores the signal too; it is sent again at
# every poll until the processes have exited.
launcher_pid=
stopping=
signal_ranks() {
  local pids
  [ -n "${launcher_pid}" ] || return 0
  pids=$(pgrep -P "${launcher_pid}" | tr '\n' ' ')
  if [ -n "${pids}" ]; then
    # shellcheck disable=SC2086
    kill -USR1 ${pids} 2>/dev/null
  fi
}
on_term() {
  echo "== $(date) SIGTERM: asking the training processes to save and stop"
  stopping=${stopping:-term}
  signal_ranks
}
trap on_term TERM
trap '' USR1

# ----------------------------------------------- augmentation data in RAM
# The training shards go through the page cache and push the memory-mapped
# noise and reverb LMDBs out of it; pages in /dev/shm stay.
if [ "${AUG_TO_SHM}" = "1" ]; then
  shm=/dev/shm/ovsb_aug_${SLURM_JOB_ID:-$$}
  trap 'rm -rf "${shm}"' EXIT
  run mkdir -p "${shm}/noise" "${shm}/reverb" || die "cannot create ${shm}"
  run cp "${PLAN_NOISE_LMDB}/data.mdb" "${shm}/noise/" \
    || die "cannot copy ${PLAN_NOISE_LMDB}"
  run cp "${PLAN_REVERB_LMDB}/data.mdb" "${shm}/reverb/" \
    || die "cannot copy ${PLAN_REVERB_LMDB}"
  overrides+=(--set "data.augment.noise_lmdb=${shm}/noise"
              --set "data.augment.reverb_lmdb=${shm}/reverb")
  echo "== noise and reverb LMDBs staged in ${shm}"
fi

# ------------------------------------------------------------------ train
train_cmd=(python -m torch.distributed.run --standalone --nnodes=1
           "--nproc_per_node=${num_gpus}" -m ovsb.bin.train
           --config "${CONFIG}" "${overrides[@]}" --resume
           --epochs_this_run "${EPOCHS_PER_SEG}")
if [ "${dry_run}" = "1" ]; then
  run "${train_cmd[@]}"
  echo "after every finished epoch:"
  drift_gate
  exit 0
fi
if [ -n "${stopping}" ]; then
  echo "== stop requested before the trainer was started"
  exit 0
fi
mkdir -p "${EXP}/models" || die "cannot create ${EXP}/models"
printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "${SLURM_JOB_ID:-none}" \
  "${PLAN_NEXT_EPOCH}" "${PLAN_NEXT_STEP}" >> "${EXP}/segments.tsv"
if command -v nvidia-smi > /dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
fi

"${train_cmd[@]}" &
launcher_pid=$!
checked=${PLAN_EPOCH_DONE}
gate_rc=0
while kill -0 "${launcher_pid}" 2>/dev/null; do
  sleep "${poll_sec}" &
  wait $!
  if [ -n "${stopping}" ]; then
    signal_ranks
    continue
  fi
  next=$((checked + 1))
  # written last and renamed into place: the epoch is complete
  [ -f "${EXP}/models/train_state_${next}.pt" ] || continue
  checked=${next}
  drift_gate
  gate_rc=$?
  if [ ${gate_rc} -ne 0 ]; then
    gate_failed ${gate_rc} "after epoch ${next}"
    stopping=gate
    signal_ranks
  fi
done
wait "${launcher_pid}"
train_rc=$?
echo "== $(date) trainer exit status ${train_rc}, stop request:" \
  "${stopping:-none}"

# epochs that finished between the last poll and the end of the trainer;
# after a time-limit stop the next job gates them before it trains
if [ ${gate_rc} -eq 0 ] && [ "${stopping}" != "term" ]; then
  drift_gate
  gate_rc=$?
  if [ ${gate_rc} -ne 0 ]; then
    gate_failed ${gate_rc} "after the trainer ended"
  fi
fi
ls -la "${EXP}/models"
if [ ${gate_rc} -ne 0 ]; then
  exit ${gate_rc}
fi
if [ ${train_rc} -ne 0 ]; then
  echo "== the trainer failed"
  exit ${train_rc}
fi
if [ -e "${EXP}/models/final_model.pt" ]; then
  echo "== all ${PLAN_NUM_EPOCHS} epochs are done: ${EXP}/models/final_model.pt"
else
  echo "== segment done; submit the same command again to continue"
fi
exit 0
