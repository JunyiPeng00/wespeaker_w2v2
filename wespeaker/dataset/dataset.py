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

import random

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from wespeaker.utils.file_utils import read_lists
from wespeaker.dataset.lmdb_data import LmdbData
import wespeaker.dataset.processor as processor


class Processor(IterableDataset):

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        """ Return an iterator over the source dataset processed by the
            given processor.
        """
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class EpochProcessor(Processor):
    """Processor whose function also receives the current epoch (kw
    'epoch'), set by set_epoch() in the main process before the DataLoader
    workers are (re)created for that epoch."""

    def set_epoch(self, epoch):
        self.kw['epoch'] = epoch
        super().set_epoch(epoch)


class DistributedSampler:

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(rank=self.rank,
                    world_size=self.world_size,
                    worker_id=self.worker_id,
                    num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):

    def __init__(self,
                 lists,
                 shuffle=True,
                 partition=True,
                 repeat_dataset=True):
        self.lists = lists
        self.repeat_dataset = repeat_dataset
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        if not self.repeat_dataset:
            for index in indexes:
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data
        else:
            indexes_len = len(indexes)
            counter = 0
            while True:
                index = indexes[counter % indexes_len]
                counter += 1
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data


def Dataset(data_type,
            data_list_file,
            configs,
            spk2id_dict,
            whole_utt=False,
            reverb_lmdb_file=None,
            noise_lmdb_file=None,
            repeat_dataset=True,
            train_lmdb_file=None,
            batch_size=None,
            crop_seed=0):
    """ Construct dataset from arguments

        We have two shuffle stage in the Dataset. The first is global
        shuffle at shards tar/raw/feat file level. The second is local shuffle
        at training samples level.

        Args:
            data_type(str): shard/raw/feat
            data_list_file: data list file
            configs: dataset configs
            spk2id_dict: spk2id dict
            reverb_lmdb_file: reverb data source lmdb file
            noise_lmdb_file: noise data source lmdb file
            whole_utt: use whole utt or random chunk
            repeat_dataset: True for training while False for testing
            train_lmdb_file: optional lmdb with raw audio bytes keyed by the
                "wav" field of the raw data list (raw data_type only)
            batch_size: dataloader batch size, required by
                configs['num_frms_choices'] (one crop length per batch)
            crop_seed: seed of the per-batch crop-length sequence
    """
    assert data_type in ['shard', 'raw', 'feat']
    frontend_type = configs.get('frontend', 'fbank')
    frontend_args = frontend_type + "_args"

    def _get_frame_params():
        """Get frame_shift (ms) and frame_length (ms) from config.
        Supports both fbank (frame_shift/frame_length in ms) and
        tfmel-style (hop_length/win_length in samples) parameters.
        """
        args = configs.get(frontend_args, {})
        resample_rate = configs.get('resample_rate', 16000)
        hop_length = args.get('hop_length', None)
        win_length = args.get('win_length', None)
        if hop_length is not None:
            fs = hop_length * 1000 // resample_rate
        else:
            fs = args.get('frame_shift', 10)
        if win_length is not None:
            fl = win_length * 1000 // resample_rate
        else:
            fl = args.get('frame_length', 25)
        return fs, fl

    lists = read_lists(data_list_file)
    shuffle = configs.get('shuffle', False)
    # Global shuffle
    dataset = DataList(lists, shuffle=shuffle, repeat_dataset=repeat_dataset)
    if data_type == 'shard':
        dataset = Processor(dataset, processor.url_opener)
        dataset = Processor(dataset, processor.tar_file_and_group)
    elif data_type == 'raw':
        dataset = Processor(dataset, processor.parse_raw,
                            lmdb_file_path=train_lmdb_file)
    else:
        dataset = Processor(dataset, processor.parse_feat)

    if configs.get('filter', True):
        # Filter the data with unwanted length
        filter_conf = configs.get('filter_args', {})
        frame_shift, _ = _get_frame_params()
        dataset = Processor(dataset,
                            processor.filter,
                            frame_shift=frame_shift,
                            data_type=data_type,
                            **filter_conf)

    # Local shuffle
    if shuffle:
        dataset = Processor(dataset, processor.shuffle,
                            **configs['shuffle_args'])

    # spk2id
    dataset = Processor(dataset, processor.spk_to_id, spk2id_dict)

    # num_frms_choices: [[num_frms, prob], ...] -> one crop length per
    # batch (see processor.random_chunk_per_batch); absent -> fixed num_frms
    num_frms_choices = configs.get('num_frms_choices', None)

    def _per_batch_chunk(dataset, frms_to_len, dtype):
        assert batch_size, 'num_frms_choices needs the dataloader batch_size'
        lens = [frms_to_len(int(c[0])) for c in num_frms_choices]
        weights = [float(c[1]) for c in num_frms_choices]
        return EpochProcessor(dataset, processor.random_chunk_per_batch,
                              lens, weights, batch_size, dtype,
                              seed=crop_seed,
                              sync_ranks=configs.get('crop_sync_ranks', True))

    if data_type == 'feat':
        if not whole_utt and num_frms_choices:
            dataset = _per_batch_chunk(dataset, lambda f: f, 'feat')
        elif not whole_utt:
            # random chunk
            chunk_len = num_frms = configs.get('num_frms', 200)
            dataset = Processor(dataset, processor.random_chunk, chunk_len,
                                'feat')
    else:
        # resample
        resample_rate = configs.get('resample_rate', 16000)
        dataset = Processor(dataset, processor.resample, resample_rate)
        # speed perturb
        speed_perturb_flag = configs.get('speed_perturb', True)
        if speed_perturb_flag:
            speed_perturb_mode = configs.get('speed_perturb_mode', 'random')
            if speed_perturb_mode == 'expanded':
                dataset = Processor(dataset, processor.speed_perturb_expand,
                                    len(spk2id_dict))
                if shuffle:
                    dataset = Processor(dataset, processor.shuffle,
                                        **configs['shuffle_args'])
            else:
                dataset = Processor(dataset, processor.speed_perturb,
                                    len(spk2id_dict))
        if not whole_utt:
            frame_shift, frame_length = _get_frame_params()

            def frms_to_len(num_frms):
                # w2vbert_sv: frame_shift = frame_length = 20 ms, so
                # num_frms 50/100/150/300 -> 1/2/3/6 s
                return ((num_frms - 1) * frame_shift +
                        frame_length) * resample_rate // 1000

            if num_frms_choices:
                dataset = _per_batch_chunk(dataset, frms_to_len, data_type)
            else:
                # random chunk
                num_frms = configs.get('num_frms', 200)
                chunk_len = frms_to_len(num_frms)
                dataset = Processor(dataset, processor.random_chunk,
                                    chunk_len, data_type)
        # add reverb & noise
        aug_prob = configs.get('aug_prob', 0.6)
        # aug_mode: legacy (default, upstream reverb XOR noise with prob
        # aug_prob) | mixed (per-sample type drawn from aug_probs; aug_prob
        # only acts as the on/off switch, extraction sets it to 0)
        aug_mode = configs.get('aug_mode', 'legacy')
        snr_ranges = {
            k: configs[k + '_snr']
            for k in ('noise', 'speech', 'music') if k + '_snr' in configs
        }
        # overlap / interfering-speaker aug (interferer_args, see
        # processor.add_interferer); in mixed mode its probability is
        # aug_probs['overlap'], in legacy mode interferer_args['prob']
        interferer_args = dict(configs.get('interferer_args', {}))
        if (reverb_lmdb_file and noise_lmdb_file) and (aug_prob > 0.0):
            reverb_data = LmdbData(reverb_lmdb_file)
            # noise_exclude_list: keys excluded from training
            # (never drawn in training); absent = all keys
            noise_data = LmdbData(noise_lmdb_file,
                                  exclude_keys=configs.get(
                                      'noise_exclude_list', None))
            if aug_mode == 'mixed':
                dataset = Processor(dataset, processor.draw_aug_type,
                                    configs['aug_probs'])
                if configs['aug_probs'].get('overlap', 0.0) > 0.0:
                    interferer_args.pop('prob', None)
                    dataset = Processor(dataset, processor.add_interferer,
                                        reverb_source=reverb_data,
                                        resample_rate=resample_rate,
                                        use_aug_type=True,
                                        **interferer_args)
                dataset = Processor(dataset, processor.add_reverb_noise,
                                    reverb_data, noise_data, resample_rate,
                                    aug_prob, aug_mode='mixed',
                                    snr_ranges=snr_ranges,
                                    overlap_noise_prob=configs.get(
                                        'overlap_noise_prob', 0.5))
            else:
                assert aug_mode == 'legacy', aug_mode
                if interferer_args.get('prob', 0.0) > 0.0:
                    dataset = Processor(dataset, processor.add_interferer,
                                        reverb_source=reverb_data,
                                        resample_rate=resample_rate,
                                        **interferer_args)
                dataset = Processor(dataset, processor.add_reverb_noise,
                                    reverb_data, noise_data, resample_rate,
                                    aug_prob, snr_ranges=snr_ranges)
        # compute fbank
        if frontend_type == 'fbank':
            dataset = Processor(dataset, processor.compute_fbank,
                                **configs['fbank_args'])

    # !!!IMPORTANT NOTICE!!!
    # To support different frontends (including ssl pretrained models),
    # we have to move apply_cmvn and spec_aug out of the dataset pipeline
    # which runs totally in cpus.
    # These two modules are now used in wespeaker/utils/executor.py (train)
    # and wespeaker/bin/extract.py (test), which runs in gpus.
    '''
    # apply cmvn
    dataset = Processor(dataset, processor.apply_cmvn)

    # spec augmentation
    spec_aug_flag = configs.get('spec_aug', True)
    if spec_aug_flag:
        dataset = Processor(dataset, processor.spec_aug,
                            **configs['spec_aug_args'])
    '''
    return dataset
