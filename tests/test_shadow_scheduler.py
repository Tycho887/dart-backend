from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dart.shadow_scheduler import (
    Contact,
    Lifecycle,
    SchedulingError,
    ShadowScheduleConfig,
    ShadowScheduler,
    shadow_identity,
)

NOW = datetime(2026, 8, 31, 10, tzinfo=UTC)


def contact(
    contact_id: str,
    start_s: int,
    *,
    duration_s: int = 300,
    system_id: str = "source",
    external_ref: str = "",
    ephemeris_id: str = "eph-1",
) -> Contact:
    start = NOW + timedelta(seconds=start_s)
    return Contact(
        id=contact_id,
        spacecraft_id="spacecraft-1",
        system_id=system_id,
        station_id="station-1",
        mission_profile_id="source-profile",
        ephemeris_id=ephemeris_id,
        start=start,
        end=start + timedelta(seconds=duration_s),
        state="scheduled",
        external_ref=external_ref,
    )


class MemoryStore:
    def __init__(self):
        self.records = []

    def append(self, record):
        self.records.append(record)


class FailOnceStore(MemoryStore):
    def append(self, record):
        if record.status == Lifecycle.BOOKED:
            raise OSError("audit unavailable")
        super().append(record)


class FakeClient:
    def __init__(self, sources, targets=()):
        self.sources = list(sources)
        self.targets = list(targets)
        self.validated = 0
        self.booked = 0
        self.assigned = []
        self.cancelled = []
        self.fail_assignment = False
        self.fail_cancellation = False
        self.fail_booking = False

    def validate_credentials(self):
        self.validated += 1

    def list_contacts(self, start, end, *, station_ids=(), system_ids=()):
        return self.targets if system_ids else self.sources

    def book_shadow(self, plan):
        if self.fail_booking:
            raise TimeoutError("booking response lost")
        self.booked += 1
        booked = contact(
            "shadow-1",
            1_000,
            system_id=plan.target_antenna_id,
            external_ref=plan.identity,
        )
        self.targets.append(booked)
        return booked

    def get_ephemeris_snapshot(self, ephemeris_id):
        return {
            "ephemeris_uuid": ephemeris_id,
            "spacecraft_uuid": "spacecraft-1",
            "inline": {"tle": "line 1\nline 2"},
        }

    def assign_ephemeris(self, contact_id, ephemeris_id):
        if self.fail_assignment:
            raise RuntimeError("assignment failed")
        self.assigned.append((contact_id, ephemeris_id))

    def cancel_contact(self, contact_id):
        if self.fail_cancellation:
            raise RuntimeError("cancellation failed")
        self.cancelled.append(contact_id)


@pytest.fixture
def config():
    return ShadowScheduleConfig(
        target_antenna_id="target",
        station_ids=("station-1",),
        mission_profile_id="profile-1",
        minimum_lead_s=600,
        search_window_s=86_400,
        minimum_tracking_s=120,
        setup_margin_s=60,
        teardown_margin_s=60,
        request_timeout_s=30,
        audit_path=Path("unused"),
    )


def test_plan_is_deterministic_and_causes_no_writes(config):
    client = FakeClient([contact("later", 2_000), contact("earlier", 1_000)])
    scheduler = ShadowScheduler(client, MemoryStore(), config)

    plan = scheduler.plan(NOW)

    assert plan.source.id == "earlier"
    assert client.validated == 1
    assert client.booked == 0


def test_conflict_skips_candidate_and_selects_next(config):
    conflict = contact("busy", 950, duration_s=500, system_id="target")
    client = FakeClient(
        [contact("first", 1_000), contact("second", 2_000)], [conflict]
    )

    assert ShadowScheduler(client, MemoryStore(), config).plan(NOW).source.id == "second"


def test_idempotent_rerun_returns_existing_without_booking(config):
    source = contact("source-1", 1_000)
    identity = shadow_identity(source.id, "target", "profile-1")
    existing = contact(
        "existing", 1_000, system_id="target", external_ref=identity
    )
    client = FakeClient([source], [existing])
    scheduler = ShadowScheduler(client, MemoryStore(), config)

    plan = scheduler.plan(NOW)
    result = scheduler.execute(plan, NOW)

    assert result.id == "existing"
    assert client.booked == 0


def test_partial_failure_is_compensated_and_audited(config):
    client = FakeClient([contact("source-1", 1_000)])
    client.fail_assignment = True
    store = MemoryStore()
    scheduler = ShadowScheduler(client, store, config)

    with pytest.raises(SchedulingError, match="after contact creation"):
        scheduler.execute(scheduler.plan(NOW), NOW)

    assert client.cancelled == ["shadow-1"]
    assert [record.status for record in store.records] == [
        Lifecycle.PLANNED,
        Lifecycle.BOOKED,
        Lifecycle.COMPENSATED,
    ]


def test_failed_compensation_requires_reconciliation(config):
    client = FakeClient([contact("source-1", 1_000)])
    client.fail_assignment = True
    client.fail_cancellation = True
    store = MemoryStore()
    scheduler = ShadowScheduler(client, store, config)

    with pytest.raises(SchedulingError):
        scheduler.execute(scheduler.plan(NOW), NOW)

    assert store.records[-1].status == Lifecycle.RECONCILIATION_REQUIRED


def test_persistence_failure_after_booking_is_compensated(config):
    client = FakeClient([contact("source-1", 1_000)])
    store = FailOnceStore()
    scheduler = ShadowScheduler(client, store, config)

    with pytest.raises(SchedulingError):
        scheduler.execute(scheduler.plan(NOW), NOW)

    assert client.cancelled == ["shadow-1"]
    assert store.records[-1].status == Lifecycle.COMPENSATED


def test_ambiguous_booking_failure_is_recorded_for_reconciliation(config):
    client = FakeClient([contact("source-1", 1_000)])
    client.fail_booking = True
    store = MemoryStore()
    scheduler = ShadowScheduler(client, store, config)

    with pytest.raises(SchedulingError, match="ambiguous"):
        scheduler.execute(scheduler.plan(NOW), NOW)

    assert store.records[-1].status == Lifecycle.RECONCILIATION_REQUIRED
    assert store.records[-1].shadow_contact_id is None


def test_incomplete_and_too_short_sources_are_rejected(config):
    client = FakeClient(
        [
            contact("short", 1_000, duration_s=30),
            contact("no-ephemeris", 2_000, ephemeris_id=""),
        ]
    )

    with pytest.raises(SchedulingError, match="no safe"):
        ShadowScheduler(client, MemoryStore(), config).plan(NOW)
