# Slurm launchers

| File | Job |
|---|---|
| `train_segment.sbatch` | one segment of a stage-2 training run |
| `eval.sbatch` | VoxCeleb1 evaluation of one checkpoint (`local/eval/README.md`) |
| `site.env.example` | the settings of both, to copy to `site.env` |

The scripts hold nothing of a site: account, partition, time limit and GPUs
go on the `sbatch` command line, everything else into the site file. Jobs
are submitted from the recipe directory (`cd "$SLURM_SUBMIT_DIR"`). They do
not ask for CPUs or memory; where those are not tied to the GPUs, add
`--cpus-per-gpu` / `--mem-per-gpu` on the command line (12 DataLoader
workers per GPU).

## Training

```bash
cp local/slurm/site.env.example local/slurm/site.env    # edit it
mkdir -p logs
sbatch --account <account> --partition <partition> --time 2-00:00:00 \
  --nodes 1 --gpus-per-node 8 --signal B:TERM@300 \
  --output logs/%x_%j.out local/slurm/train_segment.sbatch
```

`OVSB_DRY_RUN=1 bash local/slurm/train_segment.sbatch` runs every check on
the spot and prints the commands a job would run.

One job is one *segment*: it starts or resumes the run in `EXP` and ends
when `EPOCHS_PER_SEG` epochs are done, when the run is finished, when the
drift gate stops it, or shortly before the time limit. The same command
continues the run; nothing has to be edited between segments.

Before training starts the job checks, and stops with exit status 1 if one
fails: every setting is present; `DRIFT_REF` exists; the config is valid;
GPUs x batch size equals `EXPECT_GLOBAL_BATCH`; `engine.save_optimizer_state`
is on; `INIT` exists for a new run.

### Runs longer than the time limit

Submit the segments as a chain; each one starts when the one before it has
ended, however it ended:

```bash
submit="sbatch --parsable --account <account> --partition <partition> \
  --time 2-00:00:00 --nodes 1 --gpus-per-node 8 --signal B:TERM@300 \
  --output logs/%x_%j.out"
first=$($submit local/slurm/train_segment.sbatch)
second=$($submit --dependency=afterany:$first local/slurm/train_segment.sbatch)
third=$($submit --dependency=afterany:$second local/slurm/train_segment.sbatch)
```

A segment that finds all epochs done exits at once with status 0, and one
that finds the drift gate exceeded exits with status 3 without training, so
surplus links of a chain cost seconds. Plan the chain as
`ceil(epochs x hours per epoch / time limit)` segments plus one.

### What happens at the time limit

`--signal B:TERM@300` makes Slurm send SIGTERM to the batch shell 300
seconds before the limit. The shell sends SIGUSR1 to the training
processes. They finish the step they are in (all of them the same step),
the first process writes `models/partial_model.pt` and
`models/partial_state.pt`, and all exit with status 0; the job then ends
with status 0. The next segment continues the interrupted epoch at the same
step (`README.md` of the recipe, "Stopping and resuming").

The signal goes to the training processes directly, not to `torchrun`:
`torchrun` reacts to SIGTERM by passing it on and killing its processes 30
seconds later (`torch/distributed/elastic/multiprocessing/api.py`,
`SubprocessContext._close`), which is less than the save of a large model
takes. For the same reason do not `scancel` a running segment if its state
matters: `scancel --signal=TERM --batch <job>` asks for the same clean stop
as the time limit does.

Choose the lead time (the `300`) longer than one training step plus the
save (two files, about 8 GB for the reference model) plus one drift-gate
evaluation. If the job is killed during the save nothing is lost but the
partial state: all files are renamed into place when complete, and the next
segment resumes from the last complete epoch.

### Drift gate

`local/drift_gate.py` measures, for every `models/model_<N>.pt`, the
relative distance of the encoder weights from `DRIFT_REF` and appends it to
`EXP/drift.tsv`. The job runs it

* before training, when the run already has finished epochs;
* after every epoch (the trainer goes on with the next epoch meanwhile);
* once more when the trainer has ended.

Any exit status other than 0 stops the run: the training processes are
asked to save and stop, and the job ends with that status. 3 means an epoch
is above `DRIFT_GATE`; anything else means the drift could not be measured
(reference unreadable, a tensor missing from it, a shape mismatch). A
stopped run stays stopped: later segments find the row in `drift.tsv` and
exit with 3 before training. To go on regardless, raise `DRIFT_GATE`.

### Exit status of a training job

| Status | Meaning |
|---|---|
| 0 | segment ended as intended; `models/final_model.pt` exists when the run is finished |
| 1 | a setting or the config is wrong (nothing was trained) |
| 2 | the drift gate could not be evaluated |
| 3 | the drift gate is exceeded |
| other | exit status of the trainer |

`EXP/segments.tsv` gets one line per job that started the trainer: time,
job id, and the epoch and step it started at.
