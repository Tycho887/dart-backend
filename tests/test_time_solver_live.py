"""Live integration: ADX telemetry + KOGS metadata -> filter -> time-shift solver.

Mirrors the ``scripts/forest-experiment.py`` pipeline against the live
backends instead of recorded parquet: fetch a real contact's telemetry from
ADX with the production gates, resolve station coordinates and the TLE from
KOGS, build ``Sgp4Input``, split passes, and solve each pass with
``dart.time_solver``.

Secrets come from ``DART_SECRETS_ENV`` (default
``/opt/dart/secrets/test.env``); the whole module skips when the file or the
required keys are absent. Like the ``test_live_*`` tests in
``test_azure.py`` (whose env fixtures are reused here), these hit the real
cluster/APIs and are part of the default suite. Live data varies, so
data-starved outcomes (empty frames, short passes) skip rather than fail;
the assertions are strict on plumbing, filtering, and the result contract.
"""

import polars as pl
import pytest

from dart.io.azure import TrackingContext
from dart.io.kogs import get_TLE, parse_ephemeris
from dart.loaders.leo import (
    build_sgp4_input,
    station_from_kogs,
    tle_from_inline,
)
from dart.schema import SCHEMA_VERSION, Sgp4Input
from dart.time_solver import solve, split_passes

from test_azure import (  # noqa: F401  (shared fixtures)
    LIVE_TEST_TIMEOUT_SECONDS,
    _reservation_window,
    azure_with_env,
    kogs_auth,
    live_reservation,
    secrets_env,
    test_contact_id,
)

pytestmark = pytest.mark.timeout(LIVE_TEST_TIMEOUT_SECONDS)

#: passes with fewer post-filter rows than this cannot constrain a 2-parameter
#: fit meaningfully; they are skipped (same spirit as MIN_PASS_MEASUREMENTS in
#: the forest experiment)
MIN_PASS_OBS = 50


def _strict_context(contact_id: str, reservation) -> TrackingContext:
    """Case-2 query with the production-style gates (forest-experiment filters)."""
    start_time, end_time = _reservation_window(reservation)
    return TrackingContext(
        spacecraft_uuid=None,
        start_time_iso=start_time,
        end_time_iso=end_time,
        contact_uuid_list=contact_id,
        ephemeris_id=None,
        mode="sgp4",
        lock_requirement=True,
        min_elevation=5.0,
        minimum_ebn0=3.0,
        min_doppler=-9e4,  # signed KQL gates; keep both doppler signs
        max_doppler=9e4,
    )


def _tle_for_contact(auth: str, contact_id: str, reservation):
    if not reservation.ephemeris_id:
        pytest.skip(f"contact {contact_id} has no ephemeris_id in KOGS")
    return tle_from_inline(
        parse_ephemeris(get_TLE(auth, reservation.ephemeris_id)).inline_tle
    )


# ---------------------------------------------------------------------------
# KOGS plumbing
# ---------------------------------------------------------------------------


def test_live_kogs_metadata(kogs_auth, test_contact_id, live_reservation):
    """Contact -> reservation -> ephemeris TLE, and antenna -> station coords."""
    reservation = live_reservation
    assert reservation.ephemeris_id, "test contact carries no ephemeris_id"
    assert reservation.system_id, "test contact carries no system_id"

    tle = _tle_for_contact(kogs_auth, test_contact_id, reservation)
    assert tle.line1.startswith("1 ") and tle.line2.startswith("2 ")

    station = station_from_kogs(kogs_auth, reservation.system_id)
    assert -90.0 <= station.lat_deg <= 90.0
    assert -180.0 <= station.lon_deg <= 180.0
    assert station.alt_km >= 0.0


# ---------------------------------------------------------------------------
# ADX filtering
# ---------------------------------------------------------------------------


def test_live_filtered_fetch_applies_gates(
    azure_with_env, test_contact_id, live_reservation
):
    """The strict query gates really filter: elevation, lock state, doppler."""
    df = azure_with_env.fetch_tracking_data(
        _strict_context(test_contact_id, live_reservation)
    )
    assert isinstance(df, pl.DataFrame)
    if df.is_empty():
        pytest.skip("strict gates excluded every live row for the test contact")

    assert df["antenna1_position_elevation"].min() >= 5.0
    assert set(df["lr1_receiver1_carrierLockState"].unique().to_list()) == {"Locked"}
    assert df["lr1_receiver1_ebN0"].min() >= 3.0
    assert df["lr1_receiver1_actualCarrierFrequencyOffset"].abs().max() <= 9e4
    # the query orders by timestamp asc — the loader relies on it
    assert df["timestamp"].is_sorted()


# ---------------------------------------------------------------------------
# full pipeline: ADX + KOGS -> Sgp4Input -> split -> time-shift solve
# ---------------------------------------------------------------------------


def test_live_time_solver_pipeline(
    azure_with_env, kogs_auth, test_contact_id, live_reservation
):
    """forest-experiment, live: gated telemetry + KOGS metadata through the
    single-pass time-shift solver, checking the result contract per pass."""
    df = azure_with_env.fetch_tracking_data(
        _strict_context(test_contact_id, live_reservation)
    )
    if df.is_empty():
        pytest.skip("strict gates excluded every live row for the test contact")

    stations = {
        sid: station_from_kogs(kogs_auth, sid)
        for sid in df["system_id"].drop_nulls().unique().to_list()
    }
    tle = _tle_for_contact(kogs_auth, test_contact_id, live_reservation)
    spacecraft_ids = df["spacecraft_id"].drop_nulls().unique().to_list()

    inp = build_sgp4_input(
        df,
        tle=tle,
        stations=stations,
        fit_model="mean_anomaly_mean_motion",  # time shift + per-pass bias
        spacecraft_id=spacecraft_ids[0] if len(spacecraft_ids) == 1 else "",
    )
    assert isinstance(inp, Sgp4Input)
    assert inp.epoch_unix > 0.0  # TLE epoch parsed from line 1
    if len(inp.observations) < MIN_PASS_OBS:
        pytest.skip(f"only {len(inp.observations)} live observations after the gates")

    passes = [p for p in split_passes(inp) if len(p.observations) >= MIN_PASS_OBS]
    if not passes:
        pytest.skip("no live pass with enough post-filter observations")

    results = [solve(p) for p in passes]
    for sub, res in zip(passes, results):
        assert res.schema_version == SCHEMA_VERSION
        assert res.mode == "sgp4"
        assert res.fitted_tle is None
        assert res.parameter_names == ("time_shift_s", "pass_bias_hz")
        assert len(res.parameters) == 2
        assert len(res.parameter_covariance) == 4
        assert len(res.residuals) == len(sub.observations)
        # the shift must stay inside the configured search window
        assert -120.0 <= res.parameters[0] <= 120.0

    assert any(r.success for r in results), (
        "no live pass converged; shifts/rms per pass: "
        + ", ".join(f"{r.parameters[0]:.2f}s/{r.rms:.0f}Hz" for r in results)
    )
