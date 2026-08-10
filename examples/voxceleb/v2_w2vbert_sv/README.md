# W2V-BERT 2.0 + MHFA speaker verification

Self-contained recipe: model weights live in `assets/`, data lists + LMDB
links in `data/` (both git-ignored — see layout below), all stage logic in
`run.sh`. On a machine where `assets/` and `data/` are populated, everything
below works as-is.

## Quick start

```bash
cd examples/voxceleb/v2_w2vbert_sv

# (a) score the shipped reference embeddings (no GPU, minutes)
bash run.sh --stage 6 --stop_stage 6 --stage2_exp assets/mhfa_W2VB_ft_v5

# (b) evaluate the shipped fine-tuned checkpoint from scratch (GPU)
bash run.sh --stage 5 --stop_stage 6 --stage2_exp assets/mhfa_W2VB_ft_v5

# (c) full two-stage training (GPU, multi-day)
bash run.sh --stage 1 --stop_stage 2   # stage-1: frozen-frontend training + average
bash run.sh --stage 3 --stop_stage 4   # stage-2: full fine-tuning + average
bash run.sh --stage 5 --stop_stage 6   # extract embeddings + score vox1-O/E/H
```

`use_lmdb=true` (default) reads raw audio from the `data/<set>/lmdb` links;
with VoxCeleb wavs on disk pass `--use_lmdb false --data <your_data_dir>`
(standard WeSpeaker `raw.list`/`utt2spk` layout). On a cluster, wrap `run.sh`
in your scheduler submission script.

## Results (shipped mhfa_W2VB_ft_v5 checkpoint, cosine + mean-norm)

| Trials | EER (%) | minDCF (p=0.01) |
|---|---|---|
| vox1-O cleaned | 0.207 | 0.019 |
| vox1-E cleaned | 0.377 | 0.036 |
| vox1-H cleaned | 0.756 | 0.072 |

Reproduced 2026-08-10 with quick-start command (a) on the shipped reference
embeddings.

## Layout of the git-ignored resources

```
assets/
├── w2v-bert-2.0/                 # base model: config_baseline.json + model.safetensors
└── mhfa_W2VB_ft_v5/              # released two-stage result (vox2_dev, 3 ep ft)
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

Verified 2026-08-10 (PyTorch 2.6.0 rocm container, MI250X, fp32): the
checkpoint loads with 0 missing / 0 unexpected keys; 16 vox1 utterances
(12 dev + 4 test) extracted through this repo match the reference arks with
worst-case relative diff 2.9e-06 and cosine 1.00000000, identically via the
disk and LMDB input paths. Check any new extraction against the reference:

```bash
python local/compare_embeddings.py \
  <new_xvector.scp> assets/mhfa_W2VB_ft_v5/embeddings/vox1/xvector.scp
```
