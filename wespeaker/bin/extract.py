# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os

import fire
import numpy as np
import kaldiio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wespeaker.dataset.dataset import Dataset
from wespeaker.dataset.dataset_utils import apply_cmvn, spec_aug
from wespeaker.frontend import *
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.utils import parse_config_or_kwargs, validate_path


def extract(config='conf/config.yaml', **kwargs):
    # parse configs first
    configs = parse_config_or_kwargs(config, **kwargs)

    model_path = configs['model_path']
    embed_ark = configs['embed_ark']
    batch_size = configs.get('batch_size', 1)
    num_workers = configs.get('num_workers', 1)

    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False
    # Inference precision is independent of the training knobs so the eval
    # protocol stays fp32 unless explicitly asked: --extract_amp True
    # [--extract_amp_dtype bfloat16] [--extract_tf32 True].
    if configs.get('extract_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    enable_amp = bool(configs.get('extract_amp', False))
    amp_dtype = getattr(torch, configs.get('extract_amp_dtype', 'bfloat16'))
    # Whole-utterance extraction in length-sorted, zero-padded batches with
    # masks (frontends w2vbert_sv / omniasr_sv + an MHFA backend that takes
    # frame lengths): extract_batch_seconds = audio budget per batch
    # (B x longest utterance), 0 = off (one utterance per forward, the
    # original protocol); extract_buffer = utterances sorted per window.
    batch_seconds = float(configs.get('extract_batch_seconds', 0) or 0)
    sort_buffer = int(configs.get('extract_buffer', 512))

    test_conf = copy.deepcopy(configs['dataset_args'])
    frontend_type = test_conf.get('frontend', 'fbank')

    if frontend_type == 'tfmel':
        print('Loading checkpoint for tfmel model ...')
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'model_config' in ckpt:
            model_config = ckpt['model_config'].copy()
            model_config['spec'] = None
        else:
            model_config = configs['model_args']
        model = get_speaker_model(configs['model'])(**model_config)
        print(f'Initializing {frontend_type} frontend ...')
        frontend_args_dict = test_conf.get(frontend_type + "_args", {}).copy()
        frontend_args_dict.setdefault('sample_rate',
                                      test_conf['resample_rate'])
        frontend = frontend_class_dict[frontend_type](**frontend_args_dict)
        model.add_module("frontend", frontend)
        if hasattr(model, 'prepare_for_frontend'):
            model.prepare_for_frontend(frontend_type)
        load_checkpoint(model, model_path)
    else:
        model = get_speaker_model(configs['model'])(**configs['model_args'])
        if frontend_type != 'fbank':
            frontend_args = frontend_type + "_args"
            print('Initializing frontend model (this could take some time) ...')
            frontend = frontend_class_dict[frontend_type](
                **test_conf[frontend_args], sample_rate=test_conf['resample_rate'])
            model.add_module("frontend", frontend)
        print('Loading checkpoint ...')
        load_checkpoint(model, model_path)
    print('Finished !!! Start extracting ...')
    device = torch.device("cuda")
    model.to(device).eval()

    # test_configs
    # test_conf = copy.deepcopy(configs['dataset_args'])
    test_conf['speed_perturb'] = False
    if 'fbank_args' in test_conf:
        test_conf['fbank_args']['dither'] = 0.0
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['aug_prob'] = configs.get('aug_prob', 0.0)
    test_conf['filter'] = False

    dataset = Dataset(configs['data_type'],
                      configs['data_list'],
                      test_conf,
                      spk2id_dict={},
                      whole_utt=(batch_size == 1),
                      reverb_lmdb_file=configs.get('reverb_data', None),
                      noise_lmdb_file=configs.get('noise_data', None),
                      repeat_dataset=False,
                      train_lmdb_file=configs.get('train_lmdb') or None)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            prefetch_factor=4)

    validate_path(embed_ark)
    embed_ark = os.path.abspath(embed_ark)
    embed_scp = embed_ark[:-3] + "scp"

    use_ragged = (batch_seconds > 0 and batch_size == 1
                  and frontend_type in ('w2vbert_sv', 'omniasr_sv')
                  and not test_conf.get('cmvn', True)
                  and not test_conf.get('spec_aug', False))
    if batch_seconds > 0 and not use_ragged:
        print(f'WARNING: extract_batch_seconds={batch_seconds} requested but '
              f'ragged batching is unavailable (batch_size={batch_size}, '
              f'frontend={frontend_type}, cmvn={test_conf.get("cmvn", True)}, '
              f'spec_aug={test_conf.get("spec_aug", False)}): '
              'using one utterance per forward', flush=True)
    if use_ragged:
        max_samples = int(model.frontend.max_input_samples)
        budget = int(batch_seconds * test_conf.get('resample_rate', 16000))
        print(f'ragged batching: {batch_seconds} s of audio per batch, '
              f'sort buffer {sort_buffer}, cap {max_samples} samples')

    def flush(buf, writer):
        # buf: list of (key, wav 1-D cpu tensor) in data-list order. Batches
        # are formed longest-first, but embeddings are written back in the
        # original order (downstream consumers index the scp by
        # position).
        order = sorted(range(len(buf)), key=lambda j: -buf[j][1].shape[0])
        result = {}
        i = 0
        while i < len(order):
            longest = min(buf[order[i]][1].shape[0], max_samples)
            n = max(1, budget // longest)
            idx = order[i:i + n]
            chunk = [buf[j] for j in idx]
            i += n
            lens = torch.tensor([min(w.shape[0], max_samples)
                                 for _, w in chunk])
            x = torch.zeros(len(chunk), longest)
            for r, (_, w) in enumerate(chunk):
                x[r, :lens[r]] = w[:lens[r]]
            x, lens = x.to(device, non_blocking=True), lens.to(device)
            with torch.autocast('cuda', enabled=enable_amp,
                                dtype=amp_dtype):
                features, flens = model.frontend(x, lens)
                if flens is not None:
                    # e.g. an omniASR input shorter than the 400-sample CNN
                    # receptive field: fail loudly like the batch-1 path
                    assert bool((flens > 0).all()), \
                        f'no valid frame for {[k for (k, _), f in zip(chunk, flens.tolist()) if f <= 0]}'
                outputs = model(features) if flens is None \
                    else model(features, flens)
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            embeds = embeds.float().cpu().numpy()
            assert np.isfinite(embeds).all(), \
                f'non-finite embedding in batch {[k for k, _ in chunk]}'
            for r, j in enumerate(idx):
                result[j] = embeds[r]
        for j, (utt, _) in enumerate(buf):
            writer(utt, result[j])

    with torch.no_grad():
        with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +
                                 embed_scp) as writer:
          if use_ragged:
            buf = []
            for _, batch in tqdm(enumerate(dataloader)):
                # cap at the frontend's input limit on entry (vox1 has up
                # to 145 s utterances; the buffer holds 512 of them)
                buf.append((batch['key'][0],
                            batch['wav'][0, 0, :max_samples].float()))
                if len(buf) >= sort_buffer:
                    flush(buf, writer)
                    buf = []
            if buf:
                flush(buf, writer)
          else:
            for _, batch in tqdm(enumerate(dataloader)):
                utts = batch['key']
                if frontend_type == 'fbank':
                    features = batch['feat']
                    features = features.float().to(device)  # (B,T,F)
                else:  # 's3prl', 'tfmel', etc.
                    wavs = batch['wav']  # (B,1,W)
                    wavs = wavs.squeeze(1).float().to(device)  # (B,W)
                    wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                        wavs.shape[0]).to(device)  # (B)
                    with torch.autocast('cuda', enabled=enable_amp,
                                        dtype=amp_dtype):
                        features, _ = model.frontend(wavs, wavs_len)

                with torch.autocast('cuda', enabled=enable_amp,
                                    dtype=amp_dtype):
                    # apply cmvn
                    if test_conf.get('cmvn', True):
                        features = apply_cmvn(
                            features, **test_conf.get('cmvn_args', {}))
                    # spec augmentation
                    if test_conf.get('spec_aug', False):
                        features = spec_aug(features,
                                            **test_conf['spec_aug_args'])

                    # Forward through model
                    outputs = model(features)  # embed or (embed_a, embed_b)
                embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                embeds = embeds.float().cpu().detach().numpy()  # (B,F)

                for i, utt in enumerate(utts):
                    embed = embeds[i]
                    writer(utt, embed)


if __name__ == '__main__':
    fire.Fire(extract)
