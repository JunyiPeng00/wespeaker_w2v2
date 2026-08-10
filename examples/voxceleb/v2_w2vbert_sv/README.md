# W2V-BERT 2.0 + MHFA speaker verification

Self-contained recipe: model weights live in `assets/`, data lists + LMDB
links in `data/` (both git-ignored — see layout below), all stage logic in
`run.sh`. On a machine where `assets/` and `data/` are populated, everything
below works as-is.

## Quick start

```bash
cd examples/voxceleb/v2_w2vbert_sv

# (0) on a bare clone: fetch base model + released checkpoint from the HF Hub
bash local/prepare_assets.sh

# (a) score the released reference embeddings (no GPU, minutes)
bash run.sh --stage 6 --stop_stage 6 --stage2_exp assets/mhfa_W2VB_ft_v2

# (b) evaluate the released fine-tuned checkpoint from scratch (GPU)
bash run.sh --stage 5 --stop_stage 6 --stage2_exp assets/mhfa_W2VB_ft_v2

# (c) full two-stage training (GPU, multi-day)
bash run.sh --stage 1 --stop_stage 2   # stage-1: frozen-frontend training + average
bash run.sh --stage 3 --stop_stage 4   # stage-2: full fine-tuning + average
bash run.sh --stage 5 --stop_stage 6   # extract embeddings + score vox1-O/E/H
```

`use_lmdb=true` (default) reads raw audio from the `data/<set>/lmdb` links;
with VoxCeleb wavs on disk pass `--use_lmdb false --data <your_data_dir>`
(standard WeSpeaker `raw.list`/`utt2spk` layout). On a cluster, wrap `run.sh`
in your scheduler submission script.

## Results (cosine + vox2_dev mean-norm)

| Checkpoint | vox1-O EER / minDCF | vox1-E | vox1-H |
|---|---|---|---|
| **mhfa_W2VB_ft_v2** (released, [HF Hub](https://huggingface.co/JYP2024/wespeaker-mhfa-W2VB-ft-v2)) | 0.218 / 0.024 | 0.378 / 0.036 | 0.770 / 0.074 |

ft_v2 is a 3-epoch full fine-tune of the stage-1 recipe; numbers reproduced with
quick-start command (a).

## Layout of the git-ignored resources

```
assets/                           # rebuild anywhere with local/prepare_assets.sh
├── w2v-bert-2.0/                 # base model: config_baseline.json + model.safetensors
└── mhfa_W2VB_ft_v2/              # released two-stage result (vox2_dev, 3 ep ft)
    ├── config.yaml               # inference config (this repo's format)
    ├── models/avg_model.pt
    └── embeddings/{vox1,vox2_dev}  # reference embeddings (scp+ark)
data/
├── vox2_dev/  raw.list utt2spk wav.scp spk2utt  lmdb -> <fast storage>/wespeaker_vbclean_vox2dev.lmdb
├── vox1/      raw.list utt2spk wav.scp spk2utt trials/  lmdb -> <fast storage>/wespeaker_vox1.lmdb
├── musan/     lmdb -> (musan aug lmdb)
└── rirs/      lmdb -> (rirs aug lmdb)
```

The LMDBs map the raw.list `wav` value (original absolute wav path) to the
wav bytes; `raw.list`/`utt2spk` are standard WeSpeaker JSON-lines/kaldi
formats, so the same recipe runs anywhere by regenerating this layout.

## Reproducing from a bare clone

1. **Environment**: `pip install -r requirements.txt` plus `torchnet`
   (needed by the training executor, missing from upstream requirements).
2. **Models** (fully public): `bash local/prepare_assets.sh` downloads the
   official `facebook/w2v-bert-2.0` base (our copy is bit-identical, sha256
   `eb890c96...`), installs `conf/config_baseline.json` (the official config
   plus additive pruning fields, shipped in this repo), and fetches the
   released `mhfa_W2VB_ft_v2` checkpoint + reference embeddings from
   [JYP2024/wespeaker-mhfa-W2VB-ft-v2](https://huggingface.co/JYP2024/wespeaker-mhfa-W2VB-ft-v2).
   Quick-start (a) then reproduces the results table with no data at all;
   (b) additionally needs the vox1/vox2_dev audio.
3. **Data** (bring your own, licenses forbid redistribution): VoxCeleb1/2 +
   MUSAN + RIRs. Build the standard WeSpeaker layout with the upstream
   tooling (`examples/voxceleb/v2/local/prepare_data.sh`,
   `tools/make_raw_list.py`, `tools/make_lmdb.py` for musan/rirs), place or
   link it as `data/` here, and use `--use_lmdb false` for wavs on disk.
   `data/vox1/trials` are the official cleaned VoxCeleb trial lists.
4. **Retraining** (optional): `run.sh` stages 1-4, two-stage recipe as
   configured; note the `frontend_lr_ratio` caveat below.

## What was integrated into wespeaker (minimal diff to upstream)

| Added / changed | What |
|---|---|
| `wespeaker/frontend/__init__.py` | Registers frontend key **`w2vbert_sv`** (lazy import); guards optional `s3prl` / `whisper` imports |
| `wespeaker/models/ssl_backend/` | `SSL_BACKEND_MHFA` / `SSL_BACKEND_MHFA_LayerNorm` backends |
| `wespeaker/models/speaker_model.py` | Dispatches `SSL_BACKEND*` model names |
| `wespeaker/utils/schedulers.py` | BaseClass schedulers support `frontend_modules` / `frontend_lr_ratio` (differential frontend lr) |
| `wespeaker/bin/train.py` | Passes the model to BaseClass schedulers so the differential lr actually applies |
| `wespeaker/dataset/` + bin + `tools/extract_embedding.sh` | Optional `train_lmdb` (raw audio from LMDB, keyed by the raw.list `wav` field); default behavior unchanged |

Note: the frontend key is `w2vbert_sv`, NOT `w2vbert` — the upstream
`w2vbert` key is the LoRA-adapter variant from wespeaker PR #439. Configs
written for the original recipe (`frontend: w2vbert`, `w2vbert_args:`) must
be renamed accordingly (see `conf/`). Training is launched via
`python -m torch.distributed.run` (not the `torchrun` binary) so worker
processes inherit the active interpreter in venv-over-conda containers.

Note: the released checkpoints were trained with `frontend_lr_ratio: 0.1`
(frontend lr = 0.1 x backend lr); keep this setting to reproduce them.

## Equivalence to the original recipe

Verified (fp32): the checkpoint loads with 0 missing / 0 unexpected keys, and 16 vox1 utterances
(12 dev + 4 test) extracted through this repo match the corresponding
reference arks at float32 GPU-reproducibility level — worst-case relative
diff 3.1e-06 (ft_v2, vs its HF reference embeddings),
cosine 1.00000000, identically via the disk and LMDB input paths. Check any
new extraction against the reference:

```bash
python local/compare_embeddings.py \
  <new_xvector.scp> assets/mhfa_W2VB_ft_v2/embeddings/vox1/xvector.scp
```
