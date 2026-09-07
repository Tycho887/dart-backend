from datetime import UTC, datetime, timedelta

from dart.controller.acquisition import (
    AcquisitionConfig,
    AcquisitionState,
    AppliedOffset,
    BinaryLockAcquisition,
    LockSample,
)

NOW = datetime(2026, 8, 31, 10, tzinfo=UTC)


def config() -> AcquisitionConfig:
    return AcquisitionConfig(-2, 2, 1, 1, 10, 2, 2)


def readback(value: float, command_id: int) -> AppliedOffset:
    return AppliedOffset("c", "a", "e", str(command_id), value, NOW, NOW)


def lock(value: bool, age_s: float = 0) -> LockSample:
    return LockSample("c", "a", "e", value, NOW, NOW - timedelta(seconds=age_s))


def started() -> tuple[BinaryLockAcquisition, object]:
    acquisition = BinaryLockAcquisition("c", "a", "e", config())
    acquisition.precheck()
    acquisition.wait()
    return acquisition, acquisition.start(0, 100)


def confirm(acquisition, command, monotonic_s):
    acquisition.acknowledge(command.command_id)
    acquisition.confirm_applied(
        readback(command.requested_offset_s, command.command_id), monotonic_s
    )


def score(acquisition, locked, monotonic_s):
    assert acquisition.observe_lock(lock(locked), monotonic_s, NOW) is None
    return acquisition.observe_lock(lock(locked), monotonic_s + 0.1, NOW)


def test_delayed_readback_and_settling_precede_scoring():
    acquisition, command = started()
    acquisition.acknowledge(command.command_id)

    assert acquisition.observe_lock(lock(False), 1, NOW) is None
    acquisition.confirm_applied(readback(-2, command.command_id), 2)
    assert acquisition.observe_lock(lock(False), 2.5, NOW) is None
    assert acquisition.probes == []


def test_old_command_readback_cannot_confirm_new_command():
    acquisition, command = started()
    acquisition.acknowledge(command.command_id)

    acquisition.confirm_applied(readback(-2, command.command_id + 1), 2)

    assert acquisition.outstanding == command


def test_finds_both_edges_and_confirms_midpoint():
    acquisition, command = started()
    for index, locked in enumerate((False, True, True, False)):
        moment = index * 3.0
        confirm(acquisition, command, moment)
        command = score(acquisition, locked, moment + 1.0)
        assert command is not None

    assert command.reason == "lock_window_midpoint"
    assert command.requested_offset_s == -0.5
    confirm(acquisition, command, 13)
    assert score(acquisition, True, 14) is None
    assert acquisition.state == AcquisitionState.TRACKING
    assert len(acquisition.probes) == 5


def test_lock_chatter_and_stale_samples_do_not_form_edge():
    acquisition, command = started()
    confirm(acquisition, command, 0)

    assert acquisition.observe_lock(lock(True, age_s=3), 2, NOW) is None
    assert acquisition.observe_lock(lock(True), 2, NOW) is None
    assert acquisition.observe_lock(lock(False), 2.1, NOW) is None
    assert acquisition.observe_lock(lock(True), 2.2, NOW) is None
    assert acquisition.probes == []


def test_window_touching_sweep_bound_is_rejected_and_reset():
    acquisition, command = started()
    confirm(acquisition, command, 0)

    assert acquisition.observe_lock(lock(True), 1, NOW) is None
    reset = acquisition.observe_lock(lock(True), 1.1, NOW)

    assert reset is not None
    assert reset.requested_offset_s == 0
    assert reset.reason == "left_edge_censored"
    assert acquisition.state == AcquisitionState.RESETTING


def test_deadline_resets_confirmed_nonzero_offset():
    acquisition, command = started()
    confirm(acquisition, command, 0)

    reset = acquisition.tick(100)

    assert reset is not None
    assert reset.requested_offset_s == 0
    confirm(acquisition, reset, 101)
    assert acquisition.state == AcquisitionState.COMPLETE


def test_confirmation_timeout_faults_without_blind_retry():
    acquisition, _ = started()

    assert acquisition.tick(11) is None
    assert acquisition.state == AcquisitionState.FAULT


def test_loss_of_lock_resets_then_restarts_acquisition():
    acquisition, command = started()
    for index, locked in enumerate((False, True, True, False)):
        moment = index * 3.0
        confirm(acquisition, command, moment)
        command = score(acquisition, locked, moment + 1.0)
    confirm(acquisition, command, 13)
    score(acquisition, True, 14)

    assert acquisition.observe_lock(lock(False), 15, NOW) is None
    reset = acquisition.observe_lock(lock(False), 15.1, NOW)
    assert reset is not None and reset.requested_offset_s == 0
    confirm(acquisition, reset, 16)

    command = acquisition.outstanding
    assert command is not None
    assert command.reason == "reacquire"
    assert acquisition.state == AcquisitionState.REACQUIRING
