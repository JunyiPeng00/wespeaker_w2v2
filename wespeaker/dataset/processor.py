# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Zhengyang Chen (chenzhengyang117@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import kaldiio
import json
import logging
import random
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import numpy as np
from scipy import signal
from scipy.io import wavfile
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'wget -q -O - {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, spk, sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r:*")
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix in ['spk']:
                        example[postfix] = file_obj.read().decode(
                            'utf8').strip()
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        example['wav'] = waveform
                        example['sample_rate'] = sample_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning('error to parse {}'.format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()


def parse_raw(data, lmdb_file_path=None):
    """ Parse key/wav/spk from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/spk
            lmdb_file_path: optional path to an LMDB holding the raw audio
                bytes keyed by the json line's "wav" value; when given, audio
                is read from the LMDB instead of the filesystem

        Returns:
            Iterable[{key, wav, spk, sample_rate}]
    """
    if lmdb_file_path is not None:
        import lmdb  # optional dependency, only needed for LMDB reading

    def read_audio(wav):
        if lmdb_file_path is not None:
            with lmdb.open(lmdb_file_path, readonly=True, lock=False) as env:
                with env.begin() as txn:
                    raw_bytes = txn.get(wav.encode())
                    waveform, sample_rate = torchaudio.load(
                        io.BytesIO(raw_bytes))
        elif wav.endswith('|'):
            p = Popen(wav[:-1], shell=True, stdout=PIPE)
            data = p.stdout.read()
            waveform, sample_rate = torchaudio.load(io.BytesIO(data))
        else:
            waveform, sample_rate = torchaudio.load(wav)
        return waveform, sample_rate

    def apply_vad(waveform, sample_rate, vad):
        voice_part_list = []
        for start, end in vad:
            start, end = float(start), float(end)
            start, end = int(start * sample_rate), int(end * sample_rate)
            voice_part_list.append(waveform[:, start:end])
        waveform = torch.cat(voice_part_list, dim=1)
        return waveform, sample_rate

    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'wav' in obj
        assert 'spk' in obj
        key = obj['key']
        wav_file = obj['wav']
        spk = obj['spk']
        try:
            waveform, sample_rate = read_audio(wav_file)
            if 'vad' in obj:
                waveform, sample_rate = apply_vad(waveform, sample_rate,
                                                  obj['vad'])
            example = dict(key=key,
                           spk=spk,
                           wav=waveform,
                           sample_rate=sample_rate)
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))


def parse_feat(data):
    """ Parse key/feat/spk from json line

        Args:
            data: Iterable[str], str is a json line has key/feat/spk

        Returns:
            Iterable[{key, feat, spk}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'feat' in obj
        assert 'spk' in obj
        key = obj['key']
        feat_ark = obj['feat']
        spk = obj['spk']
        try:
            feat = torch.from_numpy(kaldiio.load_mat(feat_ark))
            example = dict(key=key, spk=spk, feat=feat)
            yield example
        except Exception as ex:
            logging.warning('Failed to load {}'.format(feat_ark))


def shuffle(data, shuffle_size=2500):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, wav/feat, spk}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, wav/feat, spk}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def spk_to_id(data, spk2id):
    """ Parse spk id

        Args:
            data: Iterable[{key, wav/feat, spk}]
            spk2id: Dict[str, int]

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert 'spk' in sample
        if sample['spk'] in spk2id:
            label = spk2id[sample['spk']]
        elif len(spk2id) == 0:
            label = -1  # extraction: no label needed
        else:
            # speaker removed from train_label (excluded) but still
            # present in the shards: drop the sample instead of feeding -1
            continue
        sample['label'] = label
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.
        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate
        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def speed_perturb(data, num_spks):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    speeds = [1.0, 0.9, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed_idx = random.randint(0, 2)
        if speed_idx > 0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speeds[speed_idx])], ['rate',
                                                     str(sample_rate)]])
            sample['wav'] = wav
            sample['label'] = sample['label'] + num_spks * speed_idx

        yield sample


def speed_perturb_expand(data, num_spks):
    """Expand each sample into original, 0.9x, and 1.1x speed variants.

        This matches the W2V-BERT recipe where speed variants are treated as
        separate samples with separate speaker-label offsets.
    """
    speeds = [1.0, 0.9, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        label = sample['label']
        for speed_idx, speed in enumerate(speeds):
            speed_sample = sample.copy()
            if speed_idx > 0:
                wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform, sample_rate,
                    [['speed', str(speed)], ['rate', str(sample_rate)]])
                speed_sample['wav'] = wav
            speed_sample['label'] = label + num_spks * speed_idx
            yield speed_sample


def get_random_chunk(data, chunk_len):
    """ Get random chunk

        Args:
            data: torch.Tensor (random len)
            chunk_len: chunk length

        Returns:
            torch.Tensor (exactly chunk_len)
    """
    data_len = len(data)
    data_shape = data.shape
    # random chunk
    if data_len >= chunk_len:
        chunk_start = random.randint(0, data_len - chunk_len)
        data = data[chunk_start:chunk_start + chunk_len]
        # re-clone the data to avoid memory leakage
        if type(data) == torch.Tensor:
            data = data.clone()
        else:  # np.array
            data = data.copy()
    else:
        # padding
        repeat_factor = chunk_len // data_len + 1
        repeat_shape = repeat_factor if len(data_shape) == 1 else (
            repeat_factor, 1)
        if type(data) == torch.Tensor:
            data = data.repeat(repeat_shape)
        else:  # np.array
            data = np.tile(data, repeat_shape)
        data = data[:chunk_len]

    return data


def filter(data,
           min_num_frames=100,
           max_num_frames=800,
           frame_shift=10,
           data_type='shard/raw/feat'):
    """ Filter the utterance with very short duration and random chunk the
        utterance with very long duration.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            min_num_frames: minimum number of frames of acoustic features
            max_num_frames: maximum number of frames of acoustic features
            frame_shift: the frame shift of the acoustic features (ms)
        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'key' in sample

        if data_type == 'feat':
            assert 'feat' in sample
            feat = sample['feat']
            if len(feat) < min_num_frames:
                continue
            elif len(feat) > max_num_frames:
                feat = get_random_chunk(feat, max_num_frames)
            sample['feat'] = feat
        else:
            assert 'sample_rate' in sample
            assert 'wav' in sample
            sample_rate = sample['sample_rate']
            wav = sample['wav'][0]

            min_len = int(frame_shift / 1000 * min_num_frames * sample_rate)
            max_len = int(frame_shift / 1000 * max_num_frames * sample_rate)

            if len(wav) < min_len:
                continue
            elif len(wav) > max_len:
                wav = get_random_chunk(wav, max_len)
            sample['wav'] = wav.unsqueeze(0)

        yield sample


def random_chunk(data, chunk_len, data_type='shard/raw/feat'):
    """ Random chunk the data into chunk_len

        Args:
            data: Iterable[{key, wav/feat, label}]
            chunk_len: chunk length for each sample

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert 'key' in sample

        if data_type == 'feat':
            assert 'feat' in sample
            feat = sample['feat']
            feat = get_random_chunk(feat, chunk_len)
            sample['feat'] = feat
        else:
            assert 'wav' in sample
            wav = sample['wav'][0]
            wav = get_random_chunk(wav, chunk_len)
            sample['wav'] = wav.unsqueeze(0)
        yield sample


# Default SNR ranges (dB) per MUSAN category, as used by the legacy path.
DEFAULT_SNR_RANGES = {'noise': [0, 15], 'speech': [10, 30], 'music': [5, 15]}

# Per-sample augmentation types of aug_mode 'mixed' (order matters only for
# the cumulative draw in draw_aug_type).
AUG_TYPES = ('clean', 'noise', 'reverb', 'reverb_noise', 'overlap')


def _snr_range_for_key(key, snr_ranges):
    if key.startswith('noise'):
        return snr_ranges.get('noise', DEFAULT_SNR_RANGES['noise'])
    elif key.startswith('speech'):
        return snr_ranges.get('speech', DEFAULT_SNR_RANGES['speech'])
    elif key.startswith('music'):
        return snr_ranges.get('music', DEFAULT_SNR_RANGES['music'])
    return snr_ranges.get('noise', DEFAULT_SNR_RANGES['noise'])


def _decode_rir(rir_data, resample_rate):
    rir_sr, rir_audio = wavfile.read(io.BytesIO(rir_data))
    rir_audio = rir_audio.astype(np.float32)
    if rir_sr != resample_rate:
        rir_audio = signal.resample(
            rir_audio, int(len(rir_audio) / rir_sr * resample_rate))
    rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2))
    return rir_audio


def _rir_same_room(reverb_source, key):
    """ Random RIR key from the same room directory as `key` (e.g.
        smallroom/Room155/...), i.e. another source/receiver position in the
        same simulated room. Falls back to any RIR. The room index is cached
        on the source object.
    """
    index = getattr(reverb_source, '_room_index', None)
    if index is None:
        index = {}
        for k in reverb_source.keys:
            index.setdefault(k.rsplit('/', 1)[0], []).append(k)
        reverb_source._room_index = index
    room = index.get(key.rsplit('/', 1)[0])
    if not room:
        return reverb_source.random_one()
    other = random.choice(room)
    with reverb_source.db.begin(write=False) as txn:
        value = txn.get(other.encode())
    return other, value


def apply_reverb(audio, reverb_source, resample_rate=16000, rir_item=None):
    """ Convolve `audio` (1-D np.ndarray) with a random (or the given) RIR,
        keep the input length. Returns (reverberant audio, rir key).
    """
    if rir_item is None:
        rir_item = reverb_source.random_one()
    key, rir_data = rir_item
    rir_audio = _decode_rir(rir_data, resample_rate)
    out = signal.convolve(audio, rir_audio, mode='full')[:audio.shape[0]]
    return out.astype(np.float32), key


def apply_noise(audio, noise_source, resample_rate=16000, snr_ranges=None):
    """ Add a random MUSAN-style noise at a random SNR drawn from the range of
        its category. Returns (noisy audio, noise key, snr).
    """
    snr_ranges = snr_ranges or {}
    audio_len = audio.shape[0]
    audio_db = 10 * np.log10(np.mean(audio**2) + 1e-4)
    key, noise_data = noise_source.random_one()
    snr_range = _snr_range_for_key(key, snr_ranges)
    noise_sr, noise_audio = wavfile.read(io.BytesIO(noise_data))
    # chunk BEFORE the float conversion (MUSAN files are minutes long);
    # same values and RNG use as converting first
    if noise_sr != resample_rate:
        noise_audio = get_random_chunk(
            noise_audio, int(audio_len / resample_rate * noise_sr))
        noise_audio = noise_audio.astype(np.float32) / (1 << 15)
        noise_audio = signal.resample(noise_audio, audio_len)
    else:
        noise_audio = get_random_chunk(noise_audio, audio_len)
        noise_audio = noise_audio.astype(np.float32) / (1 << 15)
    noise_snr = random.uniform(snr_range[0], snr_range[1])
    noise_db = 10 * np.log10(np.mean(noise_audio**2) + 1e-4)
    noise_audio = np.sqrt(10**(
        (audio_db - noise_db - noise_snr) / 10)) * noise_audio
    return audio + noise_audio, key, noise_snr


def random_chunk_per_batch(data,
                           chunk_lens,
                           weights,
                           batch_size,
                           data_type='shard/raw/feat',
                           seed=0,
                           epoch=0,
                           sync_ranks=True):
    """ Random chunk with ONE chunk length per batch (variable crop length).

        torch DataLoader (IterableDataset, automatic batching) builds every
        batch from `batch_size` CONSECUTIVE samples of a single worker's
        iterator, so redrawing the length every `batch_size` samples of this
        stage keeps every batch rectangular. All later stages must therefore
        be 1:1 and order-preserving (aug / interferer / fbank are).

        sync_ranks: the length sequence is drawn from a private RNG seeded by
        (seed, epoch, worker_id), which does not depend on the rank. DDP step
        s on every rank is served by worker s % num_workers, so all ranks crop
        the same length at the same step (no rank waits for a longer batch).
        If False the private RNG is seeded from the global `random` stream
        (per-rank lengths).

        Args:
            chunk_lens: candidate chunk lengths (samples or frames)
            weights: their probabilities
            batch_size: dataloader batch size (per rank)
    """
    assert len(chunk_lens) == len(weights) and batch_size > 0
    worker_info = torch.utils.data.get_worker_info()
    worker_id = 0 if worker_info is None else worker_info.id
    if sync_ranks:
        rng = random.Random('crop-{}-{}-{}'.format(seed, epoch,
                                                       worker_id))
    else:
        rng = random.Random(random.getrandbits(64))
    chunk_len = None
    for n, sample in enumerate(data):
        assert 'key' in sample
        if n % batch_size == 0:
            chunk_len = rng.choices(chunk_lens, weights=weights)[0]
        if data_type == 'feat':
            sample['feat'] = get_random_chunk(sample['feat'], chunk_len)
        else:
            wav = get_random_chunk(sample['wav'][0], chunk_len)
            sample['wav'] = wav.unsqueeze(0)
        yield sample


def draw_aug_type(data, aug_probs):
    """ Draw one augmentation type per sample (aug_mode 'mixed') and store it
        as sample['aug_type']. Consumed by add_interferer (type 'overlap')
        and add_reverb_noise.

        Args:
            aug_probs: {clean, noise, reverb, reverb_noise, overlap} -> prob;
                absolute probabilities, missing keys are 0; they must sum to
                <= 1 and the remainder goes to 'clean' (so the rate of every
                non-clean type is exactly what is written).
    """
    unknown = set(aug_probs) - set(AUG_TYPES)
    assert not unknown, 'unknown aug types {}'.format(unknown)
    weights = [float(aug_probs.get(t, 0.0)) for t in AUG_TYPES]
    assert min(weights) >= 0 and sum(weights) <= 1.0 + 1e-6, aug_probs
    weights[0] += max(0.0, 1.0 - sum(weights))  # remainder -> clean
    cum, acc = [], 0.0
    for w in weights:
        acc += w
        cum.append(acc)
    cum[-1] = 1.0
    for sample in data:
        u = random.random()
        for t, c in zip(AUG_TYPES, cum):
            if u < c:
                sample['aug_type'] = t
                break
        yield sample


def add_interferer(data,
                   prob=0.0,
                   sir_db=(0, 10),
                   cover=(0.25, 1.0),
                   reverb_both=True,
                   reverb_source=None,
                   resample_rate=16000,
                   buffer_size=64,
                   neg_sir_prob=0.0,
                   neg_sir_db=(-3, 0),
                   use_aug_type=False,
                   debug=False):
    """ Overlap augmentation: mix a second (interfering) speaker into a
        random contiguous part of the target crop. Place after cropping.

        The interferer is taken from a FIFO buffer of the last `buffer_size`
        clean crops seen by this stage (no extra IO; the stream is already
        shuffled), restricted to a DIFFERENT speaker (compared on
        sample['spk'], so speed-perturbed copies of the same speaker are
        excluded too). The label stays the target's. Order-preserving and
        1:1, so per-batch crop lengths are kept.

        SIR (dB) = 10 log10( mean(target^2 over the whole crop)
                             / mean(interferer^2 over its active segment) ),
        measured after reverberation when reverb_both.

        Args:
            prob: probability per sample (ignored when use_aug_type)
            sir_db: [lo, hi] uniform SIR range
            cover: [lo, hi] fraction of the crop covered by the interferer
            reverb_both: convolve target and interferer with two RIRs of the
                same room (needs reverb_source)
            neg_sir_prob / neg_sir_db: with this prob draw the SIR from
                neg_sir_db instead (dominant interferer cases)
            use_aug_type: mix iff sample['aug_type'] == 'overlap' (aug_mode
                mixed); when no different-speaker crop is buffered yet the
                sample falls back to aug_type 'reverb'
            debug: store mixing details in sample['interferer_info']
    """
    from collections import deque
    buf = deque(maxlen=buffer_size)
    for sample in data:
        assert 'wav' in sample
        wav = sample['wav']
        spk = sample.get('spk', sample.get('label'))
        if use_aug_type:
            do_mix = sample.get('aug_type') == 'overlap'
        else:
            do_mix = prob > 0 and prob > random.random()
        if do_mix:
            cands = [b for b in buf if b[0] != spk]
            if not cands:
                if use_aug_type:
                    sample['aug_type'] = 'reverb' if reverb_both else 'clean'
            else:
                i_spk, i_wav = random.choice(cands)
                target = wav.numpy()[0].astype(np.float32)
                T = target.shape[0]
                seg_len = max(1, int(round(random.uniform(*cover) * T)))
                seg_len = min(seg_len, T)
                start = random.randint(0, T - seg_len)
                interf = get_random_chunk(
                    i_wav.numpy()[0].astype(np.float32), seg_len)
                if reverb_both and reverb_source is not None:
                    target, rir_key = apply_reverb(target, reverb_source,
                                                   resample_rate)
                    interf, _ = apply_reverb(
                        interf, reverb_source, resample_rate,
                        rir_item=_rir_same_room(reverb_source, rir_key))
                if neg_sir_prob > 0 and neg_sir_prob > random.random():
                    sir = random.uniform(*neg_sir_db)
                else:
                    sir = random.uniform(*sir_db)
                p_t = np.mean(target.astype(np.float64)**2) + 1e-10
                p_i = np.mean(interf.astype(np.float64)**2) + 1e-10
                gain_i = np.sqrt(p_t / (p_i * 10**(sir / 10)))
                out = target.astype(np.float64)
                out[start:start + seg_len] += gain_i * interf
                norm = 1.0 / (np.max(np.abs(out)) + 1e-4)
                out = (out * norm).astype(np.float32)
                if debug:
                    sample['interferer_info'] = dict(
                        spk=i_spk, sir_db=sir, start=start, length=seg_len,
                        norm=norm, target=target, interferer=gain_i * interf)
                sample['wav'] = torch.from_numpy(out).unsqueeze(0)
        # buffer the CLEAN crop (the tensor object is never modified in
        # place downstream: augmentations assign a new sample['wav'])
        buf.append((spk, wav))
        yield sample


def add_reverb_noise(data,
                     reverb_source,
                     noise_source,
                     resample_rate=16000,
                     aug_prob=0.6,
                     aug_mode='legacy',
                     snr_ranges=None,
                     overlap_noise_prob=0.5):
    """ Add reverb & noise aug

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            reverb_source: reverb LMDB data source
            noise_source: noise LMDB data source
            resample_rate: resample rate for reverb/noise data
            aug_prob: aug probability (legacy mode only)
            aug_mode: 'legacy' = with prob aug_prob apply reverb XOR noise
                (50/50), bit-identical to upstream wespeaker;
                'mixed' = apply sample['aug_type'] drawn by draw_aug_type:
                clean | noise | reverb | reverb_noise (RIR then noise, the
                noise SNR is relative to the reverberant speech) | overlap
                (interferer already mixed by add_interferer; additive noise
                with prob overlap_noise_prob)
            snr_ranges: optional {noise, speech, music} -> [lo, hi] dB
                overrides of DEFAULT_SNR_RANGES (both modes)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    snr_ranges = snr_ranges or {}
    assert aug_mode in ('legacy', 'mixed'), aug_mode
    if aug_mode == 'mixed':
        for sample in data:
            assert 'wav' in sample
            aug_type = sample.get('aug_type', 'clean')
            if aug_type == 'clean':
                yield sample
                continue
            audio = sample['wav'].numpy()[0]
            if aug_type in ('reverb', 'reverb_noise'):
                audio, _ = apply_reverb(audio, reverb_source, resample_rate)
            if aug_type in ('noise', 'reverb_noise') or (
                    aug_type == 'overlap'
                    and overlap_noise_prob > random.random()):
                audio, _, _ = apply_noise(audio, noise_source, resample_rate,
                                          snr_ranges)
            # normalize into [-1, 1]
            audio = audio / (np.max(np.abs(audio)) + 1e-4)
            sample['wav'] = torch.from_numpy(
                np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
            yield sample
        return

    # ---- legacy path (unchanged upstream code; only snr_ranges lookup) ----
    for sample in data:
        assert 'wav' in sample
        assert 'key' in sample
        if aug_prob > random.random():
            aug_type = random.randint(1, 2)
            if aug_type == 1:
                # add reverberation
                audio = sample['wav'].numpy()[0]
                audio_len = audio.shape[0]

                _, rir_data = reverb_source.random_one()
                rir_sr, rir_audio = wavfile.read(io.BytesIO(rir_data))
                rir_audio = rir_audio.astype(np.float32)
                if rir_sr != resample_rate:
                    rir_audio = signal.resample(
                        rir_audio,
                        int(len(rir_audio) / rir_sr * resample_rate))
                rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2))
                out_audio = signal.convolve(audio, rir_audio,
                                            mode='full')[:audio_len]
            else:
                # add additive noise
                audio = sample['wav'].numpy()[0]
                audio_len = audio.shape[0]
                audio_db = 10 * np.log10(np.mean(audio**2) + 1e-4)

                key, noise_data = noise_source.random_one()
                snr_range = _snr_range_for_key(key, snr_ranges)
                noise_sr, noise_audio = wavfile.read(io.BytesIO(noise_data))
                noise_audio = noise_audio.astype(np.float32) / (1 << 15)
                if noise_sr != resample_rate:
                    # Since the noise audio could be very long, it must be
                    # chunked first before resampled (to save time)
                    noise_audio = get_random_chunk(
                        noise_audio, int(audio_len / resample_rate * noise_sr))
                    noise_audio = signal.resample(noise_audio, audio_len)
                else:
                    noise_audio = get_random_chunk(noise_audio, audio_len)
                noise_snr = random.uniform(snr_range[0], snr_range[1])
                noise_db = 10 * np.log10(np.mean(noise_audio**2) + 1e-4)
                noise_audio = np.sqrt(10**(
                    (audio_db - noise_db - noise_snr) / 10)) * noise_audio
                out_audio = audio + noise_audio

            # normalize into [-1, 1]
            out_audio = out_audio / (np.max(np.abs(out_audio)) + 1e-4)
            sample['wav'] = torch.from_numpy(out_audio).unsqueeze(0)

        yield sample


def compute_fbank(data,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=1.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          sample_frequency=sample_rate,
                          window_type='hamming',
                          use_energy=False)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def apply_cmvn(data, norm_mean=True, norm_var=False):
    """ Apply CMVN

        Args:
            data: Iterable[{key, feat, label}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'key' in sample
        assert 'feat' in sample
        assert 'label' in sample
        mat = sample['feat']
        if norm_mean:
            mat = mat - torch.mean(mat, dim=0)
        if norm_var:
            mat = mat / torch.sqrt(torch.var(mat, dim=0) + 1e-8)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def spec_aug(data, num_t_mask=1, num_f_mask=1, max_t=10, max_f=8, prob=0.6):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            prob: prob of spec_aug

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        if random.random() < prob:
            assert 'feat' in sample
            x = sample['feat']
            assert isinstance(x, torch.Tensor)
            # y = x.clone().detach()
            y = x.detach()  # inplace operation
            max_frames = y.size(0)
            max_freq = y.size(1)
            # time mask
            for i in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                y[start:end, :] = 0
            # freq mask
            for i in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                y[:, start:end] = 0
            sample['feat'] = y
        yield sample
