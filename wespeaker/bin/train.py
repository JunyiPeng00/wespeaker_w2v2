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

import glob
import os
import re
from pprint import pformat

import fire
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

import wespeaker.utils.schedulers as schedulers
from wespeaker.dataset.dataset import Dataset
from wespeaker.frontend import *
from wespeaker.models.projections import get_projection
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.frontend_utils import maybe_truncate_frontend
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.ema import ModelEMA
from wespeaker.utils.executor import run_epoch
from wespeaker.utils.train_state import (restore_for_resume,
                                         save_epoch_artifacts)
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.utils import get_logger, parse_config_or_kwargs, set_seed, \
    spk2id


def train(config='conf/config.yaml', **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    configs = parse_config_or_kwargs(config, **kwargs)
    checkpoint = configs.get('checkpoint', None)
    # NVIDIA-specific knobs (all default off to keep other recipes unchanged)
    if configs.get('tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = configs.get('cudnn_benchmark', False)
    # dist configs
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(configs['gpus'][local_rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')

    model_dir = os.path.join(configs['exp_dir'], "models")
    if rank == 0:
        # An existing (possibly pre-created) directory is fine; refuse only to
        # overwrite saved epochs without an explicit --checkpoint.
        os.makedirs(model_dir, exist_ok=True)
        if checkpoint is None and glob.glob(
                os.path.join(model_dir, 'model_*.pt')):
            print("[error] " + model_dir + " already holds model_*.pt "
                  "but no --checkpoint was given !")
            exit(1)
    dist.barrier(device_ids=[gpu])  # let the rank 0 mkdir first

    logger = get_logger(configs['exp_dir'], 'train.log')
    if world_size > 1:
        logger.info('training on multiple gpus, this gpu {}'.format(gpu))

    if rank == 0:
        logger.info("exp_dir is: {}".format(configs['exp_dir']))
        logger.info("<== Passed Arguments ==>")
        # Print arguments into logs
        for line in pformat(configs).split('\n'):
            logger.info(line)

    # seed
    set_seed(configs['seed'] + rank)

    # train data
    train_label = configs['train_label']
    train_utt_spk_list = read_table(train_label)
    spk2id_dict = spk2id(train_utt_spk_list)
    if rank == 0:
        logger.info("<== Data statistics ==>")
        logger.info("train data num: {}, spk num: {}".format(
            len(train_utt_spk_list), len(spk2id_dict)))

    # dataset and dataloader
    train_dataset = Dataset(configs['data_type'],
                            configs['train_data'],
                            configs['dataset_args'],
                            spk2id_dict,
                            reverb_lmdb_file=configs.get('reverb_data', None),
                            noise_lmdb_file=configs.get('noise_data', None),
                            train_lmdb_file=configs.get('train_lmdb') or None,
                            batch_size=configs['dataloader_args']['batch_size'],
                            crop_seed=configs['seed'])
    train_dataloader = DataLoader(train_dataset, **configs['dataloader_args'])
    batch_size = configs['dataloader_args']['batch_size']
    if configs['dataset_args'].get('sample_num_per_epoch', 0) > 0:
        sample_num_per_epoch = configs['dataset_args']['sample_num_per_epoch']
    else:
        sample_num_per_epoch = len(train_utt_spk_list)
        if (configs['data_type'] != 'feat'
                and configs['dataset_args'].get('speed_perturb', False)
                and configs['dataset_args'].get('speed_perturb_mode')
                == 'expanded'):
            sample_num_per_epoch *= 3
    epoch_iter = sample_num_per_epoch // world_size // batch_size
    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")
        logger.info('epoch iteration number: {}'.format(epoch_iter))

    # model: frontend (optional) => speaker model => projection layer
    logger.info("<== Model ==>")
    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    if frontend_type != "fbank":
        frontend_args = frontend_type + "_args"
        frontend = frontend_class_dict[frontend_type](
            **configs['dataset_args'][frontend_args],
            sample_rate=configs['dataset_args']['resample_rate'])
        configs['model_args']['feat_dim'] = frontend.output_size()
        model = get_speaker_model(configs['model'])(**configs['model_args'])
        model.add_module("frontend", frontend)
        # optional: stop the encoder after frontend_num_layers layers
        maybe_truncate_frontend(model, configs['dataset_args'],
                                configs['model_args'].get('nb_layer'), logger)
    else:
        model = get_speaker_model(configs['model'])(**configs['model_args'])
    if rank == 0:
        num_params = sum(param.numel() for param in model.parameters())
        logger.info('speaker_model size: {}'.format(num_params))
    # For model_init, only frontend and speaker model are needed !!!
    if configs['model_init'] is not None:
        logger.info('Load initial model from {}'.format(configs['model_init']))
        load_checkpoint(model, configs['model_init'])
    elif checkpoint is None:
        logger.info('Train model from scratch ...')
    # projection layer
    configs['projection_args']['embed_dim'] = configs['model_args'][
        'embed_dim']
    configs['projection_args']['num_class'] = len(spk2id_dict)
    configs['projection_args']['do_lm'] = configs.get('do_lm', False)
    if configs['data_type'] != 'feat' and configs['dataset_args'][
            'speed_perturb']:
        # diff speed is regarded as diff spk
        configs['projection_args']['num_class'] *= 3
        if configs.get('do_lm', False):
            logger.info(
                'No speed perturb while doing large margin fine-tuning')
            configs['dataset_args']['speed_perturb'] = False
    projection = get_projection(configs['projection_args'])
    model.add_module("projection", projection)
    if rank == 0:
        # print model
        for line in pformat(model).split('\n'):
            logger.info(line)
        # !!!IMPORTANT!!!
        # Try to export the model by script, if fails, we should refine
        # the code to satisfy the script export requirements
        if frontend_type == 'fbank':
            try:
                script_model = torch.jit.script(model)
                script_model.save(os.path.join(model_dir, 'init.zip'))
            except Exception as e:
                logger.warning(f"Failed to save jit script: {e}")

    # If specify checkpoint, load some info from checkpoint.
    # For checkpoint, frontend, speaker model, and projection layer
    # are all needed !!!
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
        start_epoch = int(re.findall(r"(?<=model_)\d*(?=.pt)",
                                     os.path.basename(checkpoint))[0]) + 1
        logger.info('Load checkpoint: {}'.format(checkpoint))
    else:
        start_epoch = 1
    logger.info('start_epoch: {}'.format(start_epoch))

    # ddp_model
    model.cuda()
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, **configs.get('ddp_args', {}))
    device = torch.device("cuda")

    criterion = getattr(torch.nn, configs['loss'])(**configs['loss_args'])
    if rank == 0:
        logger.info("<== Loss ==>")
        logger.info("loss criterion is: " + configs['loss'])

    if 'initial_lr' in configs['scheduler_args']:
        configs['optimizer_args']['lr'] = (
            configs['scheduler_args']['initial_lr']
        )
    optimizer = getattr(torch.optim,
                        configs['optimizer'])(ddp_model.parameters(),
                                              **configs['optimizer_args'])
    if rank == 0:
        logger.info("<== Optimizer ==>")
        logger.info("optimizer is: " + configs['optimizer'])

    # scheduler
    configs['scheduler_args']['num_epochs'] = configs['num_epochs']
    configs['scheduler_args']['epoch_iter'] = epoch_iter
    # here, we consider the batch_size 64 as the base, the learning rate will be
    # adjusted according to the batchsize and world_size used in different setup
    configs['scheduler_args']['scale_ratio'] = 1.0 * world_size * configs[
        'dataloader_args']['batch_size'] / 64
    scheduler_class = getattr(schedulers, configs['scheduler'])
    if isinstance(scheduler_class, type) and issubclass(
            scheduler_class, schedulers.BaseClass):
        # BaseClass schedulers need the unwrapped model to map optimizer
        # params to names for differential frontend lr (frontend_lr_ratio).
        scheduler = scheduler_class(optimizer, model=model,
                                    **configs['scheduler_args'])
    else:
        scheduler = scheduler_class(optimizer, **configs['scheduler_args'])
    if rank == 0:
        logger.info("<== Scheduler ==>")
        logger.info("scheduler is: " + configs['scheduler'])

    # margin scheduler
    configs['margin_update']['epoch_iter'] = epoch_iter
    margin_scheduler = getattr(schedulers, configs['margin_scheduler'])(
        model=model, **configs['margin_update'])
    if rank == 0:
        logger.info("<== MarginScheduler ==>")

    # save config.yaml
    if rank == 0:
        saved_config_path = os.path.join(configs['exp_dir'], 'config.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)

    # training
    dist.barrier(device_ids=[gpu])  # synchronize here
    if rank == 0:
        logger.info("<========== Training process ==========>")
        header = ['Epoch', 'Batch', 'Lr', 'Margin', 'Loss', "Acc"]
        for line in tp.header(header, width=10, style='grid').split('\n'):
            logger.info(line)
    dist.barrier(device_ids=[gpu])  # synchronize here

    # bfloat16 autocast needs no loss scaling; the scaler is only active for
    # float16 (amp_dtype: float16 | bfloat16, default float16).
    amp_dtype = getattr(torch, configs.get('amp_dtype', 'float16'))
    scaler = torch.amp.GradScaler(
        'cuda', enabled=configs['enable_amp'] and amp_dtype == torch.float16)
    if (configs.get('grad_norm_adam') or {}).get('enable', False):
        assert not scaler.is_enabled() and not configs.get(
            'grad_clip_on_scaled_grads', False), (
            'grad_norm_adam is the explicit form of the legacy regime: use it '
            'with bfloat16/fp32 (GradScaler off) and without the legacy flag')
        if rank == 0:
            logger.info('grad_norm_adam (explicit legacy-equivalent): %s'
                        % configs['grad_norm_adam'])
    if configs.get('grad_clip_on_scaled_grads', False):
        assert scaler.is_enabled() and configs.get('grad_clip'), (
            'grad_clip_on_scaled_grads replicates the legacy the original '
            'loop and needs enable_amp True, amp_dtype float16 and grad_clip')
        if rank == 0:
            logger.info('LEGACY grad clipping on SCALED grads (clip=%s before '
                        'unscale): effective grad norm = clip / loss_scale'
                        % configs['grad_clip'])
    # EMA of the weights (ema_decay, e.g. 0.9999), kept on rank 0 only;
    # built after load_checkpoint so it starts from the loaded weights
    ema = None
    if configs.get('ema_decay') and rank == 0:
        ema = ModelEMA(model, configs['ema_decay'],
                       warmup=configs.get('ema_warmup', False))
        logger.info('EMA decay {}'.format(configs['ema_decay']))
    # optimizer / scaler / scheduler (save_optimizer_state: true) and EMA
    # state of the resumed epoch; absent for a fresh start or a stage
    # hand-off (model_0.pt) -> fresh optimizer as before
    save_optimizer_state = configs.get('save_optimizer_state', False)
    if checkpoint is not None:
        restore_for_resume(checkpoint, optimizer, scaler, scheduler,
                           margin_scheduler, ema, save_optimizer_state,
                           logger)

    for epoch in range(start_epoch, configs['num_epochs'] + 1):
        train_dataset.set_epoch(epoch)

        run_epoch(train_dataloader,
                  epoch_iter,
                  ddp_model,
                  criterion,
                  optimizer,
                  scheduler,
                  margin_scheduler,
                  epoch,
                  logger,
                  scaler,
                  device=device,
                  configs=configs,
                  ema=ema)

        if rank == 0:
            if epoch % configs['save_epoch_interval'] == 0 or epoch > configs[
                    'num_epochs'] - configs['num_avg']:
                save_epoch_artifacts(
                    model_dir, epoch, model, optimizer, scaler, scheduler,
                    margin_scheduler, ema, save_optimizer_state,
                    configs.get('keep_all_train_states', False))

    if rank == 0:
        final_link = os.path.join(model_dir, 'final_model.pt')
        if os.path.lexists(final_link):
            os.remove(final_link)
        os.symlink('model_{}.pt'.format(configs['num_epochs']), final_link)
        logger.info(tp.bottom(len(header), width=10, style='grid'))


if __name__ == '__main__':
    fire.Fire(train)
