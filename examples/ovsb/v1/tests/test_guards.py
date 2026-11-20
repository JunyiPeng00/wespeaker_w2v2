"""Non-finite step guard and the rank-symmetric finite check."""
import logging

import pytest
import torch

from ovsb.engine.guards import (NONFINITE_ABORT, NonFiniteGuard,
                                all_ranks_finite)


@pytest.mark.parametrize('value, finite', [
    (1.5, True), (0.0, True), (-3e30, True),
    (float('nan'), False), (float('inf'), False), (float('-inf'), False)])
def test_all_ranks_finite_single_process(value, finite):
    scalar = torch.tensor(value, requires_grad=True) * 1.0
    assert all_ranks_finite(scalar) is finite


def test_all_ranks_finite_reduces_with_min(monkeypatch):
    """Another process reporting 'not finite' wins over a local finite."""
    import torch.distributed as dist
    calls = []

    def fake_all_reduce(tensor, op=None):
        calls.append(op)
        tensor.fill_(0)

    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'all_reduce', fake_all_reduce)
    assert all_ranks_finite(torch.tensor(1.0)) is False
    assert calls == [dist.ReduceOp.MIN]


def test_skip_continue_and_break_at_epoch_end():
    guard = NonFiniteGuard()
    assert guard.skip(0, 10) == 'continue'
    assert guard.skip(8, 10) == 'continue'
    assert guard.skip(9, 10) == 'break'   # last step: the epoch must end
    assert guard.skip(12, 10) == 'break'
    assert (guard.run, guard.total) == (4, 4)


def test_ok_resets_the_consecutive_count_only():
    guard = NonFiniteGuard(limit=3)
    guard.skip(0, 100)
    guard.skip(1, 100)
    guard.ok()
    assert (guard.run, guard.total) == (0, 2)
    guard.skip(3, 100)
    guard.skip(4, 100)
    with pytest.raises(RuntimeError, match='3 consecutive non-finite loss'):
        guard.skip(5, 100, epoch=2)
    assert guard.total == 5


def test_default_limit_and_message():
    guard = NonFiniteGuard()
    for i in range(NONFINITE_ABORT - 1):
        guard.skip(i, 1000, 'grad norm')
    with pytest.raises(RuntimeError) as err:
        guard.skip(NONFINITE_ABORT - 1, 1000, 'grad norm', epoch=7)
    assert '20 consecutive non-finite grad norm steps (epoch 7, batch 20)' \
        in str(err.value)


def test_abort_wins_over_break():
    guard = NonFiniteGuard(limit=1)
    with pytest.raises(RuntimeError):
        guard.skip(9, 10)


def test_warning_text(caplog):
    guard = NonFiniteGuard()
    with caplog.at_level(logging.WARNING):
        guard.skip(41, 100, 'loss', epoch=3,
                   logger=logging.getLogger('guard-test'))
    assert caplog.messages == [
        'non-finite loss at epoch 3 batch 42, step skipped on all ranks '
        '(1 consecutive, 1 in this epoch)']
