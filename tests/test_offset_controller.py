from datetime import UTC, datetime, timedelta

from dart.controller.acquisition import AppliedOffset
from dart.controller.controller import (
    ControllerConfig,
    ControlState,
    DecisionAction,
    LiveWriteAuthorization,
    TimeOffsetController,
    WriteAcknowledgement,
)
from dart.controller.UKF import FilterIdentity, TimeOffsetEstimate

NOW = datetime(2026, 8, 31, 10, tzinfo=UTC)
IDENTITY = FilterIdentity("contact", "antenna", "ephemeris")


class Writer:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.requests = []

    def write_offset(self, request):
        self.requests.append(request)
        return WriteAcknowledgement(
            request.command_id,
            request.issued_at,
            self.accepted,
            "test acknowledgement",
        )


class FailingWriter(Writer):
    def write_offset(self, request):
        raise TimeoutError("ambiguous timeout")


def config(live=True):
    return ControllerConfig(
        live_writes_enabled=live,
        absolute_semantics_confirmed=live,
        kp=0.2,
        ki=0.1,
        maximum_abs_offset_s=2.0,
        maximum_step_s=0.5,
        maximum_rate_s_per_s=0.1,
        deadband_s=0.01,
        minimum_update_interval_s=1,
        maximum_dt_s=5,
        confirmation_timeout_s=10,
        maximum_estimate_age_s=3,
        maximum_covariance_s2=0.1,
        maximum_nis=9,
        minimum_samples=3,
        authorization_max_age_s=60,
    )


def estimate(
    target=1.0,
    *,
    identity=IDENTITY,
    covariance=0.01,
    nis=1.0,
    samples=4,
    converged=True,
    age_s=0,
):
    source = NOW - timedelta(seconds=age_s)
    return TimeOffsetEstimate(
        identity=identity,
        observed_at=NOW,
        source_event_at=source,
        target_offset_s=target,
        drift_s_per_s=0.0,
        covariance=((covariance, 0.0), (0.0, covariance)),
        innovation_s=0.0,
        nis=nis,
        accepted_sample_count=samples,
        converged=converged,
        source_window_start=source - timedelta(seconds=10),
        source_window_end=source,
        profile_version="profile-1",
        model_version="model-1",
    )


def controller(writer=None, live=True):
    writer = writer or Writer()
    control = TimeOffsetController(
        IDENTITY, "profile-1", "model-1", config(live), writer
    )
    control.enable_tracking(estimate(), 0.0, 0)
    if live:
        control.arm_live_writes(
            LiveWriteAuthorization(
                "operator", IDENTITY, NOW, NOW + timedelta(seconds=30)
            ),
            NOW,
        )
    return control, writer


def readback(value, command_id):
    return AppliedOffset(
        "contact", "antenna", "ephemeris", command_id, value, NOW, NOW
    )


def test_dry_run_reports_request_without_writing():
    control, writer = controller(live=False)

    decision = control.update(estimate(), 5, NOW)

    assert decision.action == DecisionAction.DRY_RUN
    assert decision.requested_offset_s == 0.5
    assert writer.requests == []


def test_live_write_waits_for_adx_before_another_command():
    control, writer = controller()

    first = control.update(estimate(), 5, NOW)
    second = control.update(estimate(1.2), 7, NOW)

    assert first.action == DecisionAction.APPLY
    assert second.reason == "write_unconfirmed"
    assert len(writer.requests) == 1
    assert control.pending is not None
    assert control.confirm_applied(
        readback(first.requested_offset_s, control.pending.request.command_id), NOW
    )
    assert control.confirmed_applied_offset_s == first.requested_offset_s


def test_live_write_requires_immediate_explicit_authorization():
    writer = Writer()
    control = TimeOffsetController(
        IDENTITY, "profile-1", "model-1", config(), writer
    )
    control.enable_tracking(estimate(), 0.0, 0)

    decision = control.update(estimate(), 5, NOW)

    assert decision.reason == "live_write_not_authorized"
    assert writer.requests == []


def test_expired_authorization_does_not_write():
    control, writer = controller()

    decision = control.update(estimate(), 5, NOW + timedelta(seconds=31))

    assert decision.reason == "live_write_not_authorized"
    assert writer.requests == []


def test_confirmation_timeout_faults_and_stops_integrating():
    control, _ = controller()
    control.update(estimate(), 5, NOW)

    control.tick(16, NOW + timedelta(seconds=11))

    assert control.state == ControlState.FAULT
    assert control.update(estimate(), 17, NOW).reason == "controller_not_tracking"


def test_covariance_nis_freshness_and_identity_are_gated():
    control, writer = controller()

    assert control.update(estimate(covariance=1), 5, NOW).reason == "excessive_covariance"
    assert control.update(estimate(nis=10), 5, NOW).reason == "excessive_nis"
    assert control.update(estimate(age_s=4), 5, NOW).reason == "stale_estimate"
    wrong = FilterIdentity("other", "antenna", "ephemeris")
    assert control.update(estimate(identity=wrong), 5, NOW).reason == "identity_mismatch"
    assert writer.requests == []
    assert control.state == ControlState.FAULT


def test_slew_saturation_and_bumpless_transfer():
    control, _ = controller(live=False)

    first = control.update(estimate(target=20), 1, NOW)
    later = control.update(estimate(target=20), 10, NOW)

    assert first.requested_offset_s == 0.1
    assert later.requested_offset_s == 0.5


def test_termination_zero_is_acknowledged_then_confirmed():
    control, writer = controller()
    control.confirmed_applied_offset_s = 0.4

    request = control.terminate(5, NOW)

    assert request is not None and request.requested_offset_s == 0
    assert control.state == ControlState.RESETTING
    assert control.pending is not None
    assert control.confirm_applied(readback(0, control.pending.request.command_id), NOW)
    assert control.state == ControlState.COMPLETE
    assert len(writer.requests) == 1


def test_transport_rejection_does_not_change_confirmed_state():
    control, _ = controller(Writer(accepted=False))

    decision = control.update(estimate(), 5, NOW)

    assert decision.reason == "write_not_acknowledged"
    assert control.confirmed_applied_offset_s == 0
    assert control.state == ControlState.FAULT


def test_ambiguous_transport_failure_faults_without_retry():
    control, _ = controller(FailingWriter())

    decision = control.update(estimate(), 5, NOW)

    assert decision.reason == "transport_failure"
    assert control.pending is None
    assert control.state == ControlState.FAULT
