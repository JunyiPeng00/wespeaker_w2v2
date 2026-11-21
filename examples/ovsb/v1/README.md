# OVSB recipe (v1)

Speaker verification with a fully fine-tuned W2V-BERT 2.0 encoder (22 layers)
and an MHFA backend, trained on a multi-corpus list with an episodic
P x K sampler. The recipe is a self-contained package (`ovsb/`) on top of the
unmodified wespeaker core; the stock tools (`wespeaker/bin/extract.py`,
`average_model.py`, `score_fast.py`) are used for extraction and scoring.

## Quick start

```bash
. ./path.sh
bash run.sh --stage 1 --stop_stage 1    # training list              (CPU)
bash run.sh --stage 2 --stop_stage 2    # prints the stage-1 training command
bash run.sh --stage 3 --stop_stage 5    # initial checkpoint of stage 2
                                        # (stage 4 prints a GPU command)
bash run.sh --stage 6 --stop_stage 6    # checks the config, prints the sbatch
                                        # command of stage 2
bash run.sh --stage 7 --stop_stage 7    # average of the last checkpoints
bash run.sh --stage 8 --stop_stage 8    # prints the sbatch command of the
                                        # evaluation
```

`run.sh` runs the CPU stages and prints the commands of the GPU stages; it
submits nothing. Stage 2 of the training runs through
`local/slurm/train_segment.sbatch`, set up by a copy of
`local/slurm/site.env.example` (`local/slurm/README.md`: settings, runs
longer than the time limit, drift gate). Without a scheduler the trainer is
started directly:

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  -m ovsb.bin.train --config conf/stage2_v9c.yaml [--set path=value ...] \
  [--resume] [--epochs_this_run N]
```

One process per GPU; their number must equal `len(run.gpus)`, from which
the learning rate (x processes x batch size / 64) and the steps per epoch
follow. The run directory `run.exp_dir` gets `recipe.yaml` (the resolved
recipe config, with `derived` and the arm order), `config.yaml` (the
wespeaker-schema view the extraction tools read), `train.log`,
`run_meta_<job>.json` and `models/`. Every option is described where it is
defined: `conf/stage2_v9c.yaml` (annotated reference config) and
`ovsb/config/schema.py`; see "Configuration" below.

Further reading: `local/init/README.md` (initial checkpoint),
`local/eval/README.md` (evaluation protocol), `local/slurm/README.md`.

## Layout

```
run.sh         the stages of the recipe
path.sh        environment (puts this directory on PYTHONPATH: `import ovsb`)
conf/          recipe configs
  stage2_v9c.yaml      the reference stage-2 config
  stage2_v8c1.yaml, stage2_v9a.yaml, stage2_v9b.yaml, cont_c40.yaml
                       overlays of stage2_v9c.yaml (only the keys that differ)
  cont_c40_ranksplit.yaml
                       cont_c40.yaml with `data.sampler.rank_split_arms: [cn1]`
ovsb/          the recipe package
  config/      typed config: schema, YAML + overlays, legacy import, core view
  data/        index, speaker table, sampler, crop, augmentation, collate
  model/       frontend subclass, classifier head, model builder
  losses/      consistency and pairwise losses
  engine/      training step, loop, checkpoints, signals, logging
  bin/         command-line entry points
local/         data preparation, initialisation, launch and evaluation scripts
tests/         CPU tests (pytest)
tools, wespeaker   links to the repository's tools/ and wespeaker/
```

## Layering rule

Each layer imports only what is listed; nothing imports a layer above it.

| Layer | May import |
|---|---|
| `ovsb.config` | Python standard library, `yaml` |
| `ovsb.data` | `ovsb.config`, `wespeaker.dataset.*`, numpy, torch |
| `ovsb.model` | `ovsb.config`, `wespeaker.frontend`, `wespeaker.models`, torch |
| `ovsb.losses` | torch (functions take tensors, never batch dicts or configs) |
| `ovsb.engine` | all of the above, `wespeaker.utils.*` |
| `ovsb.api` | all of the above: construction functions, `build_training` |
| `ovsb.bin` | any layer |

`ovsb.data` never imports `ovsb.model`, `ovsb.losses` or `ovsb.engine`;
`ovsb.model` never imports `ovsb.data`. The epoch loop (`ovsb.engine.loop`)
reaches the data layer through one call, `data.loader(epoch, stream,
generator)`. `ovsb.bin.train` builds a training process with
`ovsb.api.build_training`, the one construction path, and puts its dataset
behind that call.

## Changes to the wespeaker core

Two additive hooks, both no-ops for every existing recipe:

* `wespeaker/bin/extract.py` truncates the encoder to
  `dataset_args.frontend_num_layers` when the key is present, as training
  does.
* `W2VBertSVFrontend._mask_frames` is an identity extension point after the
  feature projection; the recipe's frontend subclass overrides it.

## Configuration

```bash
. ./path.sh
python -m ovsb.bin.validate_config conf/stage2_v9c.yaml              # check + print
python -m ovsb.bin.validate_config conf/cont_c40.yaml --view core    # wespeaker view
python -m ovsb.bin.validate_config conf/stage2_v9c.yaml --set run.seed=1
```

* A config is strict: an unknown key, a wrong type or an out-of-range value
  is an error that names the dotted path (`data.crop.full_prob: must be
  <= 1.0, got 1.5`).
* Overlay: a file with a top-level `base: <file>` (relative to itself) is
  merged into that file; mappings merge key by key, lists and scalars
  replace.
* Overrides: `--set dotted.path=value`, value in YAML syntax.
* Paths in a config are relative to this directory.
* Derived values (`derived.epoch_iter`, `derived.scale_ratio`,
  `derived.num_class`) are computed from the config and the training list
  (`--resolve`), never written by hand.
* The order of `data.sampler.arms` (and `cells`) is behaviour: it sets the
  arm codes and the sampler's random stream. In a hand-written file the
  order of the mapping is the order; an overlay keeps the position of an
  arm it changes and appends an arm it adds. A loaded config carries the
  order as `data.sampler.arm_order` / `cell_order`, and a printed config
  names them, so it reads back in the same order however its keys are
  sorted.
* `<exp_dir>/config.yaml --legacy <authored conf>` reads a config saved by
  the original wespeaker-schema trainer and refuses any option this recipe
  does not implement. The saved file has sorted keys, so the arm order is
  taken from the hand-written conf of that run
  (`ovsb.config.load_legacy(saved, authored)`).

## Stopping and resuming

Files in `<run.exp_dir>/models`:

| File | Content |
|---|---|
| `model_<N>.pt` | weights after epoch N |
| `train_state_<N>.pt` | optimiser and schedule after epoch N; marks N as complete |
| `aux_state_<N>.pt` | running centre of the pairwise loss, one state per process |
| `partial_model.pt`, `partial_state.pt` | weights and training state of an interrupted epoch |
| `head_spk.json` | the class list the classifier head was verified against |
| `final_model.pt` | link to the model of the last epoch, written when the run is finished |

**Resume.** `--resume` continues from the partial state if there is one for
an epoch after the last complete epoch, else from the last complete epoch,
else it starts a new run from `model.init.checkpoint`. Without `--resume` a
directory that already holds a state is refused. A resume needs the same
number of processes and the same steps per epoch as the run had. Keys that
changed since the run was started are listed in `train.log` as warnings.

**Segments.** `--epochs_this_run N` ends the process after N complete
epochs (exit status 0); the next process continues with `--resume`.

**Signals.** SIGUSR1 or SIGTERM sent to the training processes is a stop
request. The request is shared between the processes once per step, so all
of them stop after the same step, whichever process got the signal and
when. The first process then writes the partial state and all exit with
status 0. A request that arrives during the last step of an epoch, or while
an epoch is being saved, ends the run at the epoch boundary without a
partial state. Send the signal to the training processes themselves, not to
`torchrun`: it forwards SIGTERM and kills its processes 30 seconds later.
Every file is written under another name and renamed, so a process killed
during a save leaves the previous state usable.

**After a stop inside an epoch** the next process continues that epoch at
the same step and with the same learning rate, optimiser state and running
centres. The remaining steps of the epoch draw their data from a new
random stream (new speaker cells, crops, augmentation, frame masks), so no
batch already trained on is replayed. In `train.log` the running means of
that epoch restart at the resumed step.

## Reproducibility

What two runs of the same config, seed and number of processes share:

* **Exactly reproducible on a deterministic backend** (CPU; checked in
  `tests/test_train_entry.py`): a run split into segments at epoch
  boundaries, by `--epochs_this_run` or by a stop request on the last step
  of an epoch, gives the weights, optimiser state and running centres of the
  uninterrupted run, bit for bit. Every epoch after the first starts from
  random generators seeded by (seed, process, epoch), not from where the
  previous epoch left them (`ovsb/engine/seeding.py`).
* **On GPUs** two runs of the same job are not bit-identical (cuDNN
  autotuning, non-deterministic kernels); segments do not add to that.
* **Not reproduced: a run interrupted inside an epoch.** The steps before
  the interruption and the schedule are those of the uninterrupted run; the
  rest of that epoch sees other data, so the weights differ from there on.
  Later epochs again draw the batches the uninterrupted run would draw.
* **Against the original trainer** (the wespeaker-schema training tree this
  recipe was ported from): the training step is bit-identical on the same
  batch and weights (`tests/test_step.py`), and in the first epoch of a
  new run the processes are seeded and the DataLoader takes its worker seed
  as there. From the second epoch on the generators are reseeded here and
  continue there, so the two draw different, equally distributed
  augmentation and frame masks. The original
  replays the generators of the first epoch after every restart; this
  trainer does not.
* **The initial checkpoint** built by `local/init/build_init.sh` is
  functionally equivalent, not bit-identical, to the one the reference
  model started from; `local/init/README.md` lists what is identical and
  what differs.

## Deliberate differences from the reference implementation

The recipe reproduces the implementation the reference model was trained
with (`tests/parity/README.md`: sampler stream, batches, schedule, losses
and gradients are bit-identical on CPU). It differs on purpose in these
points, none of which changes a normal training step of the first epoch:

* **Head:** class cosines are clamped to `[-1 + 1e-7, 1 - 1e-7]` before the
  square root of the margin formula; without it a cosine of exactly 1 gives
  NaN gradients.
* **Running centre of the pair-AUC term:** a non-finite row mean is skipped
  and counted instead of entering the running mean.
* **Codec chain:** a row whose codec call failed is marked `chain_fail`, not
  as having passed the codec; failures are counted and logged.
* **Logging:** the first process writes the table; loss and accuracy are
  means over all processes; `Lr` is the learning rate of the back-end and
  head, `LrFE` that of the encoder.
* **Stopping and resuming:** see above; generators are reseeded per epoch,
  and a run can be stopped inside an epoch.
* **Drift gate:** a gate that cannot be evaluated stops the run.
* **Sampler statistics:** printed at the end of every worker's epoch, with
  the number of speakers repeated inside an episode. Small arms can repeat
  speakers: every (process, worker) pair draws from its own share of the
  speakers, and an arm with few speakers may have fewer in a share than an
  episode asks for. This is the behaviour of the reference run and is kept
  as the default. `data.sampler.rank_split_arms` lists arms whose speakers
  are split over the processes only, so that all workers of a process share
  them; for the reference list this removes the repeats of the `cn1` arm
  (5 % of its draws at a weight of 0.25, 22 % at 0.40).

## Tests

```bash
python -m pytest tests -q
```

Tests that compare against the original training tree need
`OVSB_SOURCE_TREE=<its repository root>` and are skipped without it.
