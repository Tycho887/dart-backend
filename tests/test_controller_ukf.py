from datetime import UTC, datetime, timedelta

import pytest

from dart.controller.UKF import (
    FilterIdentity,
    OffsetMeasurement,
    TimeOffsetUKF,
    UkfConfig,
)

NOW = datetime(2026, 8, 31, 10, tzinfo=UTC)
IDENTITY = FilterIdentity("contact", "antenna", "ephemeris")


def measurement(value, variance=0.01, identity=IDENTITY, second=0):
    epoch = NOW + timedelta(seconds=second)
    return OffsetMeasurement(identity, epoch, epoch, value, variance, 0.0)


def ukf():
    return TimeOffsetUKF(
        UkfConfig(0.001, 0.0001, 20, 9, 0.02, 2),
        "profile-1",
        "model-1",
    )


def test_irregular_cadence_scales_prediction_and_converges():
    filter_ = ukf()
    first = filter_.update(measurement(1.0), 0)
    second = filter_.update(measurement(1.1, second=2), 2)
    third = filter_.update(measurement(1.2, second=5), 5)

    assert first.accepted_sample_count == 1
    assert second.accepted_sample_count == 2
    assert third.accepted_sample_count == 3
    assert third.source_window_start == NOW
    assert third.source_window_end == NOW + timedelta(seconds=5)
    assert third.covariance[0][0] > 0


def test_nis_outlier_is_rejected_without_incrementing_sample_count():
    filter_ = ukf()
    filter_.update(measurement(0.0), 0)
    accepted = filter_.update(measurement(0.01, second=1), 1)
    rejected = filter_.update(measurement(100.0, second=2), 2)

    assert rejected.nis > 9
    assert rejected.accepted_sample_count == accepted.accepted_sample_count
    assert rejected.target_offset_s != pytest.approx(100)


def test_identity_change_and_excessive_gap_reset_filter():
    filter_ = ukf()
    filter_.update(measurement(1.0), 0)
    changed = FilterIdentity("contact", "antenna", "new-ephemeris")

    identity_reset = filter_.update(measurement(4.0, identity=changed), 1)
    gap_reset = filter_.update(measurement(6.0, identity=changed, second=30), 30)

    assert identity_reset.accepted_sample_count == 1
    assert identity_reset.target_offset_s == 4
    assert gap_reset.accepted_sample_count == 1
    assert gap_reset.target_offset_s == 6


def test_invalid_measurement_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        ukf().update(measurement(float("nan")), 0)
