# OVSB recipe (v1)

Speaker verification with a fully fine-tuned W2V-BERT 2.0 encoder (22 layers)
and an MHFA backend, trained on a multi-corpus list with an episodic
P x K sampler. The recipe is a self-contained package (`ovsb/`) on top of the
unmodified wespeaker core; the stock tools (`wespeaker/bin/extract.py`,
`average_model.py`, `score_fast.py`) are used for extraction and scoring.

## Layout

```
path.sh        environment (puts this directory on PYTHONPATH: `import ovsb`)
conf/          recipe configs
  stage2_v9c.yaml      the reference stage-2 config
  stage2_v8c1.yaml, stage2_v9a.yaml, stage2_v9b.yaml, cont_c40.yaml
                       overlays of stage2_v9c.yaml (only the keys that differ)
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
| `ovsb.bin` | any layer |

`ovsb.data` never imports `ovsb.model`, `ovsb.losses` or `ovsb.engine`;
`ovsb.model` never imports `ovsb.data`.

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

## Tests

```bash
python -m pytest tests -q
```

Tests that compare against the original training tree need
`OVSB_SOURCE_TREE=<its repository root>` and are skipped without it.
