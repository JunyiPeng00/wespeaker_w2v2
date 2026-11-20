# Evaluation protocol

VoxCeleb1-O/E/H (cleaned trial lists) with the stock wespeaker tools:
`wespeaker/bin/extract.py`, `score_fast.py`, `compute_metrics.py`.

```bash
. ./path.sh
bash local/eval/extract.sh --exp_dir exp/stage2_v9c \
  --model exp/stage2_v9c/models/model_14.pt --tag ep14 --maxsec 10   # GPU
bash local/eval/score.sh --eval_dir exp/stage2_v9c/eval_ep14_tf32_max10s  # CPU
```

or both in one job: `local/slurm/eval.sbatch` (usage in its header).

Needs `data/vox1/{shard.list, wav.scp, trials/vox1_{O,E,H}_cleaned.kaldi}`,
`data/vox2_dev/{shard.list, wav.scp}` and `<exp_dir>/config.yaml`, the
wespeaker-schema view of the model that the trainer writes.

## What is computed

**Model.** The checkpoint as trained: 22 encoder layers
(`dataset_args.frontend_num_layers: 22`, `model_args.nb_layer: 23`; core
`extract.py` cuts the 24-layer `assets/w2v-bert-2.0` at load time). No
averaging of checkpoints for the reference model (`num_avg: 1`).

**Embeddings of the trial utterances** (`make_extract_config.py` writes
the config):

* the whole utterance, cut to its first `MAXSEC` seconds
  (`max_input_seconds`), in **one forward pass**: no chunking, no averaging
  of segment embeddings, no test-time augmentation;
* `MAXSEC` = **10** for development numbers, **60** for reported numbers;
* fp32 weights with TF32 matmul and cuDNN (`extract_tf32: true`), no
  autocast;
* 16 kHz input, GPU filterbank (`feature_extractor: torch`), no CMVN;
* batching: utterances are read with batch size 1, sorted by length inside
  a window of 512 and embedded in zero-padded batches of at most 320 s of
  audio (`extract_batch_seconds: 320`) with frame masks in the encoder and
  in the pooling. Padding is masked, but batch composition can change the
  last bits of an embedding, so results are not guaranteed bit for bit
  when the number of extraction processes changes.

**Mean.** The mean of VoxCeleb2-dev embeddings from the same checkpoint is
subtracted from every trial embedding (`score_fast.py --cal_mean True`).
Two variants, chosen with `--vox2_mean`:

* `frac` (default; development numbers): every 10th shard of
  `data/vox2_dev/shard.list` (lines 1, 11, 21, ...; about 110,000
  utterances), each utterance a **random 3 s crop** (`num_frms: 150`) in
  batches of 64. The crops are not seeded, so the mean vector differs
  slightly between runs;
* `full` (reported numbers): all 1,092,009 utterances, whole utterance with
  the same cap and batching as the trial utterances.

**Scores.** Cosine similarity of the mean-subtracted, length-normalised
embeddings. No score normalisation (no AS-norm), no quality measure
functions, no calibration, no fusion.

**Metrics.** EER and minDCF with C_miss = C_fa = 1 at P_target 0.01
(`scores/vox1_cos_result`) and 0.05 (`scores/vox1_cos_result_p005`).

## Relation to the reference numbers

The reference evaluations ran the same extraction and scoring code with a
config that had the same values for every key written here (model, layer
count, cap, feature settings, precision, batching). Differences in form:

* they loaded an asset directory whose `config.json` declares 22 layers
  instead of cutting the 24-layer model at load time (same weights, same
  computation);
* they used one extraction process on one GPU (`--nj 1`); `extract.sh`
  starts one process per GPU of `--gpus`;
* shard lists were given with absolute paths.

`extract.sh` covers VoxCeleb1 only. Numbers on other trial sets use the
same embedding protocol (`MAXSEC` 60, mean `full`) on those sets' audio and
are not produced by this directory.
