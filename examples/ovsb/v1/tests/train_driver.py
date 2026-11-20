"""Run ``ovsb.bin.train`` with the in-memory data of ``loop_helpers``
(``fake_training`` builds the process, ``FakeData`` feeds the loop).

    python tests/train_driver.py <arguments of ovsb.bin.train>

Used by tests/test_train_entry.py for runs in separate processes (signals,
two distributed processes on the CPU). With ``OVSB_TEST_TRACE`` set, every
process also saves what it restored before the first step of the process to
``restored_rank<r>.pt`` in that directory: the running-centre state, the
optimiser's step count and the position of the learning-rate scheduler.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import RECIPE_DIR  # noqa: E402  (sets the import paths)
from loop_helpers import TRACE_ENV, FakeData, fake_training  # noqa: E402

from ovsb.bin import train as train_mod  # noqa: E402


def main():
    assert os.path.isdir(os.path.join(RECIPE_DIR, 'ovsb'))
    torch.set_num_threads(2)
    trace = os.environ.get(TRACE_ENV)
    real_fit = train_mod.fit

    def fit(model, bundle, cfg, device, optimizer, scheduler, margin, data,
            start, rank, *args, **kwargs):
        if trace:
            steps = [int(s['step']) for s in optimizer.state.values()]
            torch.save({'center': bundle.center.state_dict(),
                        'optimizer_steps': sorted(set(steps)),
                        'scheduler_iter': scheduler.current_iter,
                        'start': (start.start_epoch, start.start_step,
                                  start.start_stream)},
                       os.path.join(trace, 'restored_rank%d.pt' % rank))
        return real_fit(model, bundle, cfg, device, optimizer, scheduler,
                        margin, data, start, rank, *args, **kwargs)

    train_mod.fit = fit
    return train_mod.main(sys.argv[1:], build=fake_training,
                          data_factory=FakeData)


if __name__ == '__main__':
    sys.exit(main())
