# Initial checkpoint of stage 2

Stage 2 starts from the stage-1 model (frozen-encoder training,
`conf/stage1_frozen.yaml`) with two changes:

1. the encoder is cut to its first 22 layers (`truncate_ckpt.py`);
2. the classifier head gets rows for every class of the stage-2 training
   list (`imprint_head.py`): a speaker stage 1 was trained on keeps its 3
   rows, any other speaker gets 3 identical rows pointing at the mean of
   its stage-1 embeddings, scaled to the mean row norm of the stage-1 head.

`build_init.sh` runs the four steps; only step 3 needs a GPU.

```bash
. ./path.sh
bash local/init/build_init.sh --stage 1 --stop_stage 2    # CPU
bash local/init/build_init.sh --stage 3 --stop_stage 3    # GPU
bash local/init/build_init.sh --stage 4 --stop_stage 4    # CPU, ~5 GB
```

| Step | Tool | Output (under `exp/init/`) |
|---|---|---|
| 1 | `truncate_ckpt.py <stage-1 model> 22` | `stage1_L22.pt` (+ `.head.json`) |
| 2 | `imprint_head.py list` | `imprint/{wav/, raw.list, utt2spk}` |
| 3 | `imprint_extract.sh` | `imprint_emb/embeddings/imprint/xvector.scp` |
| 4 | `imprint_head.py imprint` | `stage2_init.pt`, `.head.json`, `.json`, `.dup.tsv` |

Inputs (options of `build_init.sh`):

* `--stage1_exp` (default `exp/stage1_frozen`): its `config.yaml` and
  `models/final_model.pt`. **The model of the last epoch is used, not the
  average** written by the stage-1 run: that is what the reference model
  started from. Pass `--stage1_ckpt` to use another checkpoint.
* `--stage1_label`: the `utt2spk` stage 1 was trained on (its sorted
  speakers are the classes of the stage-1 head).
* `--train_label`, `--train_index`: the stage-2 training list
  (`local/data/build_train_list.py`).

`stage2_init.pt.head.json` records the speaker list the head was built for
(sha256 of the sorted speaker ids, class and row counts). The trainer
refuses a checkpoint whose record does not match its training list.
`stage2_init.pt.dup.tsv` lists new classes whose row has a cosine above 0.7
to a stage-1 class: possible duplicate identities across corpora, to be
reviewed by hand (nothing is merged automatically).

`python local/init/imprint_head.py selftest` checks the head construction
on a synthetic checkpoint in a few seconds.

## Equivalence with the checkpoint the reference model started from

The result is **functionally equivalent, not bit-identical**, to that
checkpoint.

Identical, bit for bit (checked on the real files):

* every tensor other than the classifier head (encoder layers 1-22, MHFA
  with its layer weights cut to 23 entries);
* the head rows of every class that was in the stage-1 list (110,291 of
  the 152,367 classes).

Different: the rows of the 42,076 classes that were not in the stage-1
list. The reference checkpoint was not built in one pass. Its head grew
in four steps as corpora were added to the training list, and each step
set its new rows separately:

| Step | New classes | Embedding model | Utterances per speaker | Cut |
|---|---|---|---|---|
| a | CN-Celeb2, MagicData, RAMC, AISHELL-1/3, 3D-Speaker | stage 1, 24 layers | 20 | 10 s |
| b | KeSpeech | stage 1, 22 layers | 10 | 6 s |
| c | TidyVoiceX | stage 1, 22 layers | 20 | 60 s |
| d | CN-Celeb1 dev | stage 1, 22 layers | 20 | 60 s |

(after step a the encoder was cut to 22 layers; step b also removed the
rows of the identities dropped from the list or merged into another
speaker). The one-pass builder differs in three ways:

1. **Row scale.** New rows are scaled to the mean row norm of the head they
   are added to. The reference steps used the head of the previous step
   (9.6033, 9.6026, 9.6028, 9.6028), the one-pass builder uses the stage-1
   head (9.6057). With the embeddings of the reference steps the rebuilt
   rows have cosine 1.0 to the reference rows (minimum over the 42,076
   classes: 1 - 1e-15) and are longer by a factor 1.00026 to 1.00032.
2. **Embedding model of step a.** Those embeddings came from the 24-layer
   stage-1 model; the builder embeds everything with the 22-layer model,
   the model that actually starts stage 2.
3. **Utterance selection.** One setting for all corpora (20 utterances per
   speaker, first 10 s). A merged identity draws its utterances from all
   of its source speakers; in the reference, an identity merged after
   step a kept the row of its canonical speaker alone. Class means
   therefore differ slightly from the reference ones.

Points 2 and 3 change the direction of the new rows a little; their size
was not measured (it needs a GPU extraction). None of them changes which
classes exist, the class order, or any row the stage-1 training produced.
An exact reproduction of a reference run needs the reference checkpoint
itself.
