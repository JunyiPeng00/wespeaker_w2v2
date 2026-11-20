# Whole-recipe parity gate

This recipe is a port of a training set-up that lived in another wespeaker
tree. A parity harness (`parity.py`, kept with that tree's refactoring work,
not in this repository) recorded what the original code does on the
reference run: sampler draws, the first batches of a data-loader worker,
the learning-rate schedule, and the model with two real training steps. The
record is a JSON file, the *reference dump*.

The gate records the same quantities from this recipe, through `ovsb.api`
only, and requires them to be bit-identical to the reference dump.

| file | content |
|---|---|
| `ovsb_adapter.py` | `dump` (this recipe -> JSON in the harness layout) and `compare` (reference vs new) |
| `run_gate.sh` | both steps, exit code 0 = parity |
| `site.example.yaml` | template of the site file |

## Running it

CPU only, no job scheduler: one process, 4 threads, about 4 minutes, peak
memory about 14 GB. FFmpeg 6 must be loadable by torchaudio (the codec
chain of `data.channel` runs for real).

```bash
cp tests/parity/site.example.yaml /path/outside/the/repo/site.yaml   # fill in
export PYTHON=/path/to/env/bin/python
export OVSB_MODULES="<FFmpeg module>"        # if FFmpeg comes from a module
tests/parity/run_gate.sh /path/outside/the/repo/site.yaml /path/to/harness_dir
```

`harness_dir` holds `parity.py` and one reference dump `ref_*.json`. The
two steps by hand:

```bash
$PYTHON tests/parity/ovsb_adapter.py dump --harness $H/parity.py \
    --site site.yaml --out $OUT/new.json --tmp $OUT
$PYTHON tests/parity/ovsb_adapter.py compare --harness $H/parity.py \
    $H/<reference>.json $OUT/new.json
```

`compare` also writes `new.ref.view.json` and `new.view.json` (see
"Reductions"); the harness itself can be run on them:
`parity.py compare --strict new.ref.view.json new.view.json`.

Use the python environment and the thread count of the reference dump
(`meta.versions`, `meta.threads` in the JSON) on the same kind of machine:
floating-point results are compared bit for bit, and another BLAS, torch
build or CPU can change the last bits.

Useful options of `dump`: `--checks sampler,batches,schedule` (no model:
2 minutes, 6 GB), `--base-seed N` (run the batches with a given data-loader
base seed, to tell "the seed moved" from "the pipeline changed").

## The site file

The recipe config `conf/stage2_v9c.yaml` names data by placeholder paths.
The site file says where the data of the reference run is: the working
directory, the checkpoint for the model check, the reference run's two
config files and a mapping of overrides for the path fields of the config.
It contains machine-specific absolute paths and is kept outside the
repository.

## What is compared

Everything is read from the reference run's data: training list and tar
shards, noise and RIR LMDBs, the pretrained encoder, a checkpoint.

**sampler** - for epochs {1, 2, 9} x ranks {0, 3, 7} x workers {0, 5, 11}
of an 8 x 12 run, the first 200 episodes of the cell: a digest over, per
row, key, speaker, arm, language, session, position in the K-group,
positive type and the five recording conditions. Also the speaker table
(sizes, speakers per arm, digests of the arm, row and speaker arrays).

**batches** - rank 0, worker 0, epoch 1, the first 12 batches through the
real pipeline (tar read, resampling, crop, zero-run dither, noise / reverb,
codec chain in the helper process, padding, default collate), produced by
torch's own worker loop in a forked process. Per batch: digest, shape and
dtype of every field including the waveform bytes; integer fields in full.
`base_seed` is the number the first data-loader iterator draws from the
global torch generator after `ovsb.api.build_training`; it must equal the
reference value, which was derived from the original trainer's statement
order (seed, dataset, model). A construction that consumes one random
number more or less before the first iterator fails here first.

**schedule** - learning rate of both param groups and the margin for all
steps of the run (digest and 13 steps in clear): uninterrupted, resumed at
every epoch boundary and resumed every 2 epochs. A resume writes the train
state and the auxiliary state as `ovsb.engine.checkpoint.save_epoch` does
(without the model file), builds fresh optimiser and schedulers and calls
`ovsb.engine.checkpoint.resume`.

**model** - key, shape and dtype digests of the built model and of the
checkpoint; strict load (head check through
`ovsb.engine.checkpoint.load_init`); a fingerprint of the loaded weights;
eval-mode embeddings of a fixed synthetic batch (ragged batch and single
utterances); two real optimisation steps on CPU in fp32 through
`ovsb.engine.step.train_step` (a plain batch, then a pair batch on the
updated weights): total loss, gradient norm per module and per encoder
layer before clipping, clipped norm, learning rates, norm of the weight
change per module.
The loss components are derived exactly as in the reference, as
differences of totals: the same batch is forwarded with one loss weight set
to zero (the step is stopped at `backward`), `ce` = total without the pair
terms, `consistency` = (total with consistency - `ce`) / weight,
`pair_auc` = (total - total without pair_auc) / weight. The reference
trainer kept one running centre per warm-up setting; the adapter does the
same, so the forward with the configured warm-up of 20 batches and the two
steps with warm-up 1 each have the centre state they had there.

**config_port** (not strict) - the harness's own config check reads guard
internals of the original tree. Here instead: the reference run's saved
config, read with the legacy reader, must equal `conf/stage2_v9c.yaml`
except for path fields; the same for the config the gate ran with (recipe
config plus site overrides). Differences in the order of name-keyed
mappings (`policies`, `arm_probs`, `arm_caps`) are listed and are not
behaviour; a difference in `arm_order` or `cell_order` fails.

## Reductions

`compare` reduces the two dumps to what both sides define, prints each
reduction, and then requires exact equality (a float that differs in the
last bit fails):

* `config` of the reference is dropped (see config_port).
* `batches.base_seed_source` is a label and is dropped; `base_seed` and
  `base_seed_train_order` are compared.
* Batch fields: the reference collate also emitted `label_b`, `lam`,
  `lang`, `vid`, `seg`, `dev`, `dcls`, `dial`, `genre`, which no training
  step of this recipe reads; this recipe emits `chain_fail` in addition.
  The fields both sides have are compared; the others are listed.
  `chain_fail` must be -1 on every row (no codec call failed), which is
  reported; `chain_code` is then comparable on all rows.
* `sampler.build_log`: the summary line of the speaker table is compared
  after its first comma (the label in front differs).
* If the resumed schedule trajectories could not be produced they are
  marked `pending engine` in the dump and left out, with a note.

## What the gate does not cover

Multi-GPU execution (DDP set-up, gradient all-reduce, the rank-symmetric
non-finite skip), bf16 autocast and TF32 on CUDA, checkpoint writing,
mid-epoch interruption and resume, workers other than worker 0 and ranks
other than rank 0 for audio, batches after the 12th, epochs after the first
for audio, evaluation. Those have their own unit tests or need a GPU job.
