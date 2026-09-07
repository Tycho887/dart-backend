"""Loader for the SGP4 (LEO) mode: backend telemetry + KOGS metadata →
``Sgp4Input``.

v1 assumption: single-TLE batches. The TLE is resolved from the first
contact present in the telemetry; per-pass TLEs are a later concern. The TLE
travels as plain strings — propagation uses satkit's SGP4 downstream.
"""

from __future__ import annotations

import polars as pl

from dart.io.azure import fetch_tracking_data
from dart.io.kogs import (
    get_antenna,
    get_contact,
    get_TLE,
    parse_ephemeris,
    parse_reservation,
    parse_response,
)
from dart.io.common import observations_from_frame, tle_epoch_unix
from dart.io.schema import (
    FitParameter,
    Sgp4FitOptions,
    Sgp4Input,
    SolverOptions,
    Station,
    Tle,
)


def tle_from_inline(inline_tle: str | None) -> Tle:
    """Extract the two TLE lines from a multi-line blob (with or without a
    leading name line)."""
    if not inline_tle:
        raise ValueError("ephemeris has no inline TLE")
    lines = [ln.strip() for ln in inline_tle.strip().splitlines() if ln.strip()]
    if len(lines) == 2:
        return Tle(line1=lines[0], line2=lines[1])
    if len(lines) >= 3:
        # first line is usually a name/header
        return Tle(line1=lines[-2], line2=lines[-1])
    raise ValueError(f"inline TLE has {len(lines)} lines; expected 2 (or 3 with a name line)")


def station_from_kogs(auth: str, system_id: str) -> Station:
    """KOGS system antenna → schema Station (altitude converted m → km)."""
    ant = parse_response(get_antenna(auth, system_id))
    if None in (ant.latitude, ant.longitude, ant.altitude):
        raise ValueError(f"incomplete antenna coordinates for system {system_id}")
    return Station(
        id=system_id,
        name=ant.station_name or ant.antenna_name,
        lat_deg=float(ant.latitude),
        lon_deg=float(ant.longitude),
        alt_km=float(ant.altitude) / 1000.0,
    )


def tle_from_kogs(auth: str, ephemeris_id: str) -> Tle:
    """KOGS ephemeris → schema Tle."""
    return tle_from_inline(parse_ephemeris(get_TLE(auth, ephemeris_id)).inline_tle)


def build_sgp4_input(
    telemetry: pl.DataFrame,
    *,
    tle: Tle,
    stations: dict[str, Station],
    nominal_center_frequency_hz: float = 2.0e9,
    fit_model: str = "mean_anomaly",
    spacecraft_id: str = "",
) -> Sgp4Input:
    """Pure normalization: telemetry frame + metadata → Sgp4Input.

    The reference epoch is the TLE epoch — the state a fit is expressed at.
    """
    if telemetry.is_empty():
        raise ValueError("telemetry contains no observations")
    observations = observations_from_frame(telemetry, stations)
    pass_ids = list(dict.fromkeys(obs.contact_id for obs in observations))
    bias_spec = FitParameter(
        initial=0.0,
        lower=-15_000.0,
        upper=15_000.0,
        scale=2_000.0,
        finite_difference_step=1e-2,
    )
    return Sgp4Input(
        spacecraft_id=spacecraft_id,
        epoch_unix=tle_epoch_unix(tle.line1),
        tle=tle,
        stations=list(stations.values()),
        observations=observations,
        options=SolverOptions(ref_frame="TEME"),
        fit=Sgp4FitOptions(
            model=fit_model,
            pass_ids=pass_ids,
            nominal_center_frequency_hz=nominal_center_frequency_hz,
            pass_biases=[bias_spec for _ in pass_ids],
        ),
    )


def load_sgp4_input(auth: str, ctx) -> Sgp4Input:
    """End-to-end: fetch telemetry (ADX) + metadata (KOGS) → Sgp4Input.

    Stations are resolved for every ``system_id`` in the telemetry; the TLE
    comes from the first contact's ephemeris.
    """
    df = fetch_tracking_data(ctx)

    system_ids = df["system_id"].drop_nulls().unique().to_list()
    stations = {sid: station_from_kogs(auth, sid) for sid in system_ids}

    contact_id = df["contact_id"].drop_nulls()[0]
    reservation = parse_reservation(get_contact(auth, contact_id)["contact"])
    if not reservation.ephemeris_id:
        raise ValueError(f"contact {contact_id} has no ephemeris_id")
    tle = tle_from_kogs(auth, reservation.ephemeris_id)

    spacecraft_ids = df["spacecraft_id"].drop_nulls().unique().to_list()
    spacecraft_id = spacecraft_ids[0] if len(spacecraft_ids) == 1 else ""

    return build_sgp4_input(df, tle=tle, stations=stations, spacecraft_id=spacecraft_id)
