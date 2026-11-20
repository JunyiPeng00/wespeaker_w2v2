"""Train the recipe model.

Usage (from the recipe directory, after ``. ./path.sh``)::

    python -m torch.distributed.run --standalone --nnodes=1 \\
        --nproc_per_node=8 -m ovsb.bin.train --config conf/stage2_v9c.yaml \\
        [--set dotted.path=value ...] [--resume] [--epochs_this_run N]

    python -m ovsb.bin.train --config conf/stage2_v9c.yaml --plan

One process per GPU: under ``torchrun`` the number of processes
(``WORLD_SIZE``) must equal ``len(run.gpus)``, because the learning rate
and the steps per epoch are derived from ``run.gpus``. Without
``WORLD_SIZE`` in the environment the trainer runs as a single process.
Processes use CUDA when it is available, else the CPU.

Start and resume. A new run starts at epoch 1 from
``model.init.checkpoint``. With ``--resume`` the run continues from the
newest state in ``<run.exp_dir>/models`` (``ovsb.engine.checkpoint``): the
partial state of an interrupted epoch, else the last complete epoch, else
it starts new. Without ``--resume`` a directory that holds a state is
refused. ``--epochs_this_run N`` ends the process after N complete epochs.

Signals. SIGUSR1 or SIGTERM to the training processes ends the run after
the current step with a partial state (``ovsb.engine.signals``); the exit
status is 0 and ``--resume`` continues.

Files written to ``run.exp_dir``: ``recipe.yaml`` (the resolved recipe
config), ``config.yaml`` (the wespeaker-schema view the extraction tools
read), ``run_meta_<job>.json``, ``train.log`` and ``models/``.

``--plan`` prints, as shell assignments ``PLAN_<NAME>=<value>``, what a
launcher needs to know about the config and the saved state, and exits.
"""
import argparse
import dataclasses
import logging
import os
import shlex
import sys
from pprint import pformat
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import yaml

from ovsb import api
from ovsb.config import core_view, load
from ovsb.config.schema import Config, ConfigError
from ovsb.data.policies import PTYPES
from ovsb.engine import checkpoint
from ovsb.engine.arm_log import make_arm_logger
from ovsb.engine.checkpoint import FRESH, ResumePoint
from ovsb.engine.console import header_lines
from ovsb.engine.loop import TrainData, fit, link_final_model
from ovsb.engine.run_meta import write_run_meta
from ovsb.engine.signals import StopFlag
from wespeaker.utils.utils import get_logger

Builder = Callable[[Config, int, int, torch.device], api.TrainingObjects]
DataFactory = Callable[[Config, int, int, api.TrainingObjects], TrainData]
RECIPE_FILE, CORE_FILE, LOG_FILE = 'recipe.yaml', 'config.yaml', 'train.log'
EXIT_OK, EXIT_ERROR = 0, 1


class PipelineData:
    """The recipe's data pipeline behind the loop's data seam.

    ``loader`` sets the epoch and stream on the dataset of ``objects``
    (in the main process, before the workers of the pass start). The pass
    ``(epoch 1, stream 0)`` uses the loader ``api.build_training`` built,
    which takes its worker seed from the torch CPU generator; any other
    pass gets a loader that draws it from the generator of the pass.
    """

    def __init__(self, cfg: Config, rank: int, world: int,
                 objects: api.TrainingObjects) -> None:
        self.cfg, self.objects = cfg, objects

    def loader(self, epoch: int, stream: int,
               generator: Optional[torch.Generator]) -> Any:
        self.objects.dataset.set_epoch(epoch, stream)
        if generator is None:
            return self.objects.loader
        return api.build_loader(self.objects.dataset, self.cfg, generator)


def get_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', required=True, help='recipe config')
    parser.add_argument('--set', dest='overrides', action='append',
                        default=[], metavar='PATH=VALUE',
                        help='override, e.g. run.exp_dir=exp/trial '
                        '(repeatable)')
    parser.add_argument('--resume', action='store_true',
                        help='continue from the newest state in '
                        '<run.exp_dir>/models (a new run if there is none)')
    parser.add_argument('--epochs_this_run', type=int, default=None,
                        metavar='N', help='end this process after N '
                        'complete epochs')
    parser.add_argument('--plan', action='store_true',
                        help='print the launch plan and exit')
    return parser.parse_args(argv)


def plan(cfg: Config) -> Dict[str, Any]:
    """What a launcher needs to know (module docstring)."""
    model_dir = os.path.join(cfg.run.exp_dir, 'models')
    point = checkpoint.find_resume_point(model_dir,
                                         cfg.engine.save_optimizer_state)
    complete = checkpoint.complete_epochs(model_dir,
                                          cfg.engine.save_optimizer_state)
    world, batch = len(cfg.run.gpus), cfg.data.loader.batch_size
    return dict(
        EXP_DIR=cfg.run.exp_dir, NUM_EPOCHS=cfg.run.num_epochs,
        NUM_AVG=cfg.run.num_avg, WORLD_SIZE=world, BATCH_SIZE=batch,
        GLOBAL_BATCH=world * batch,
        SAVE_OPTIMIZER_STATE=int(cfg.engine.save_optimizer_state),
        INIT_CHECKPOINT=cfg.model.init.checkpoint,
        NOISE_LMDB=cfg.data.augment.noise_lmdb,
        REVERB_LMDB=cfg.data.augment.reverb_lmdb,
        RESUME_KIND=point.kind, EPOCH_DONE=complete[-1] if complete else 0,
        NEXT_EPOCH=point.start_epoch, NEXT_STEP=point.start_step)


def _flatten(tree: Any, prefix: str = '') -> Dict[str, Any]:
    if not isinstance(tree, dict):
        return {prefix: tree}
    flat: Dict[str, Any] = {}
    for key, value in tree.items():
        flat.update(_flatten(value, '%s.%s' % (prefix, key) if prefix
                             else str(key)))
    return flat


def write_configs(cfg: Config, logger: logging.Logger) -> None:
    """Write ``recipe.yaml`` and ``config.yaml`` into the run directory.

    A ``recipe.yaml`` of an earlier process of the run is compared first;
    every key that changed is logged as a warning (the files are then
    replaced).
    """
    recipe = os.path.join(cfg.run.exp_dir, RECIPE_FILE)
    if os.path.exists(recipe):
        try:
            old = _flatten(load.read_yaml(recipe))
        except ConfigError as exc:
            old = {}
            logger.warning('%s: %s' % (recipe, exc))
        new = _flatten(load.to_dict(cfg))
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                logger.warning('config changed since the run was started: '
                               '%s: %r -> %r' % (key, old.get(key),
                                                 new.get(key)))
    for path, text in (
            (recipe, load.dump(cfg)),
            (os.path.join(cfg.run.exp_dir, CORE_FILE),
             yaml.safe_dump(core_view.to_core_dict(cfg), sort_keys=False))):
        with open(path + '.tmp', 'w', encoding='utf8') as fout:
            fout.write(text)
        os.replace(path + '.tmp', path)


def _barrier(device: torch.device) -> None:
    if dist.is_initialized():
        if device.type == 'cuda':
            dist.barrier(device_ids=[device.index])
        else:
            dist.barrier()


def train(args: argparse.Namespace, cfg: Config, stop: StopFlag,
          build: Builder, data_factory: DataFactory) -> int:
    """Run one training process; returns its exit status."""
    engine, run = cfg.engine, cfg.run
    if engine.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = engine.cudnn_benchmark
    # processes
    distributed = 'WORLD_SIZE' in os.environ
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    if world != len(run.gpus):
        raise ConfigError(
            'run.gpus: %d process(es) are running (WORLD_SIZE), the config '
            'lists %d GPU(s) %s; the learning rate and the steps per epoch '
            'follow run.gpus' % (world, len(run.gpus), run.gpus))
    if torch.cuda.is_available():
        device = torch.device('cuda', int(run.gpus[local_rank]))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    if distributed:
        dist.init_process_group(
            backend='nccl' if device.type == 'cuda' else 'gloo')

    model_dir = os.path.join(run.exp_dir, 'models')
    has_state = bool(checkpoint.saved_epochs(model_dir)) or os.path.exists(
        os.path.join(model_dir, checkpoint.PARTIAL_STATE))
    if has_state and not args.resume:
        if rank == 0:
            print('[error] %s already holds a training state; pass --resume '
                  'to continue it' % model_dir, file=sys.stderr)
        return EXIT_ERROR
    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)
    _barrier(device)  # the directory exists before any process logs

    logger = get_logger(run.exp_dir, LOG_FILE)
    logger.setLevel(logging.INFO)  # whatever the root logger is set to
    log = logger if rank == 0 else None
    if world > 1:
        logger.info('training on multiple gpus, this process %d on %s' %
                    (rank, device))
    point = checkpoint.find_resume_point(model_dir,
                                         engine.save_optimizer_state) \
        if args.resume else ResumePoint(FRESH)
    weights = point.weights or cfg.model.init.checkpoint
    if rank == 0:
        logger.info('exp_dir is: %s' % run.exp_dir)
        logger.info('<== Passed Arguments ==>')
        for line in pformat(load.to_dict(cfg)).split('\n'):
            logger.info(line)
        logger.info('run metadata: %s' % write_run_meta(
            cfg, args.config, weights, world))
        logger.info(point.describe())
    if point.start_epoch > run.num_epochs:
        if rank == 0:
            link_final_model(model_dir, run.num_epochs)
            logger.info('all %d epochs are done, nothing to do' %
                        run.num_epochs)
        return EXIT_OK

    # Seeding (run.seed + rank), data side, model on its device, optimiser,
    # schedulers and objective, in the one order that fixes the random
    # initialisation and the worker seeds of the first epoch. Nothing below
    # draws from the torch generators before the first batch is requested.
    obj = build(cfg, rank, world, device)
    torch.backends.cudnn.benchmark = engine.cudnn_benchmark  # set_seed's
    classes = [str(name) for name in obj.table.class_names]
    cfg = load.resolve(cfg, num_class=len(classes))
    epoch_iter = cfg.derived.epoch_iter
    data = data_factory(cfg, rank, world, obj)
    model, optimizer, bundle = obj.model, obj.optimizer, obj.bundle
    scheduler, margin_scheduler = obj.scheduler, obj.margin_scheduler
    if rank == 0:
        logger.info('<== Data statistics ==>')
        logger.info('spk num: %d' % len(classes))
        logger.info('epoch iteration number: %d' % epoch_iter)
        logger.info('<== Model ==>')
        logger.info('speaker_model size: %d' % sum(
            p.numel() for p in model.parameters()))
    checkpoint.load_init(model, weights, classes, model_dir,
                         write=rank == 0, logger=log)
    if rank == 0:
        logger.info('start_epoch: %d' % point.start_epoch)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, **dataclasses.asdict(engine.ddp)) if distributed else model
    if rank == 0:
        write_configs(cfg, logger)

    _barrier(device)
    if rank == 0:
        logger.info('<========== Training process ==========>')
        for line in header_lines():
            logger.info(line)
    _barrier(device)

    checkpoint.resume(point, model_dir, optimizer, scheduler,
                      margin_scheduler, bundle.center, rank, world,
                      epoch_iter, engine.save_optimizer_state, log)
    result = fit(ddp_model, bundle, cfg, device, optimizer, scheduler,
                 margin_scheduler, data, point, rank, world, log, stop,
                 args.epochs_this_run,
                 make_arm_logger(cfg, PTYPES, device))
    if rank == 0:
        logger.info('training process ends: %s (next: epoch %d step %d)' %
                    (result.reason, result.epoch, result.step))
    return EXIT_OK


def main(argv: Optional[List[str]] = None,
         build: Builder = api.build_training,
         data_factory: DataFactory = PipelineData) -> int:
    """Command-line entry.

    Args:
        argv: arguments (default: ``sys.argv[1:]``).
        build: ``(cfg, rank, world, device) -> TrainingObjects``, the one
            construction path of a training process.
        data_factory: ``(cfg, rank, world, objects) -> TrainData``, the
            data seam of ``ovsb.engine.loop`` over the built objects;
            ``cfg`` is the resolved config.

    Returns:
        0 when the process ended as intended (all epochs done, a stop
        request or ``--epochs_this_run`` reached), 1 when the run was
        refused, 2 for an invalid config.
    """
    args = get_args(argv)
    try:
        cfg = load.load(args.config, args.overrides)
    except ConfigError as exc:
        print('INVALID %s\n%s' % (args.config, exc), file=sys.stderr)
        return 2
    if args.plan:
        for name, value in plan(cfg).items():
            print('PLAN_%s=%s' % (name, shlex.quote(str(value))))
        return EXIT_OK
    stop = StopFlag().install()  # before the slow start-up
    root = logging.getLogger('Pyobj, f')  # the logger of get_logger
    handlers = list(root.handlers)
    try:
        return train(args, cfg, stop, build, data_factory)
    except ConfigError as exc:
        print('INVALID %s\n%s' % (args.config, exc), file=sys.stderr)
        return 2
    finally:
        stop.restore()
        for handler in [h for h in root.handlers if h not in handlers]:
            root.removeHandler(handler)
            handler.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    sys.exit(main())
