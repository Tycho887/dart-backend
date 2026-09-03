# """Loader for the RK89 (cislunar) mode: backend telemetry + KOGS metadata →
# ``Rk89Input``.

# The initial ECI state is taken from the first point of the ephemeris's
# inline CCSDS OEM. v1 parses the common OEM KVN layout
# (``EPOCH X Y Z VX VY VZ`` per state line); other layouts raise loudly.
# """

# from __future__ import annotations

# import datetime

# import polars as pl

# from dart.io.azure import fetch_tracking_data
# from dart.io.kogs import (
#     get_contact,
#     get_TLE,
#     parse_ephemeris,
#     parse_reservation,
# )
# from dart.loaders.common import observations_from_frame
# from dart.loaders.leo import station_from_kogs
# from dart.schema import ForceModel, Rk89Input, SolverOptions, Station

# State = tuple[float, tuple[float, float, float], tuple[float, float, float]]


# def _oem_epoch_to_unix(iso: str) -> float | None:
#     """Parse an OEM EPOCH as unix seconds.

#     CCSDS ``TIME_SYSTEM = UTC``: a naive epoch (no offset) means UTC, so the
#     result must not depend on the machine's local timezone.
#     """
#     try:
#         dt = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
#     except ValueError:
#         return None
#     if dt.tzinfo is None:
#         dt = dt.replace(tzinfo=datetime.UTC)
#     return dt.timestamp()


# def parse_oem_state(inline_oem: str) -> State:
#     """First state point of an inline OEM: ``(epoch_unix, pos_km, vel_km_s)``.

#     Recognizes state lines in the CCSDS OEM KVN layout — either
#     ``EPOCH X Y Z VX VY VZ`` (8 tokens) or a bare ``X Y Z VX VY VZ``
#     (7 tokens) — and ignores everything else (META_* headers, comments).
#     Raises ``ValueError`` when no recognizable state line exists.
#     """
#     epoch_unix: float | None = None
#     pos: tuple[float, float, float] | None = None
#     vel: tuple[float, float, float] | None = None

#     for raw in inline_oem.splitlines():
#         fields = raw.strip().split()
#         if len(fields) == 8 and fields[0] == "EPOCH":
#             epoch_str, nums = fields[1], fields[2:]
#         elif len(fields) == 7:
#             epoch_str, nums = fields[0], fields[1:]
#         else:
#             continue
#         epoch = _oem_epoch_to_unix(epoch_str)
#         if epoch is None:
#             continue
#         try:
#             pos = (float(nums[0]), float(nums[1]), float(nums[2]))
#             vel = (float(nums[3]), float(nums[4]), float(nums[5]))
#         except ValueError:
#             continue
#         epoch_unix = epoch
#         break

#     if epoch_unix is None or pos is None or vel is None:
#         raise ValueError("no recognizable OEM state line (expected `EPOCH X Y Z VX VY VZ`)")
#     return epoch_unix, pos, vel


# def build_rk89_input(
#     telemetry: pl.DataFrame,
#     *,
#     epoch_unix: float,
#     pos_km: tuple[float, float, float],
#     vel_km_s: tuple[float, float, float],
#     stations: dict[str, Station],
#     force_model: ForceModel | None = None,
# ) -> Rk89Input:
#     """Pure normalization: telemetry frame + state + metadata → Rk89Input."""
#     if telemetry.is_empty():
#         raise ValueError("telemetry contains no observations")
#     observations = observations_from_frame(telemetry, stations)
#     return Rk89Input(
#         epoch_unix=float(epoch_unix),
#         pos_km=pos_km,
#         vel_km_s=vel_km_s,
#         force_model=force_model or ForceModel(),
#         stations=list(stations.values()),
#         observations=observations,
#         options=SolverOptions(ref_frame="EME2000"),
#     )


# def load_rk89_input(auth: str, ctx, *, force_model: ForceModel | None = None) -> Rk89Input:
#     """End-to-end: fetch telemetry (ADX) + state/metadata (KOGS) → Rk89Input.

#     The initial state is parsed from the first contact's inline OEM.
#     """
#     df = fetch_tracking_data(ctx)

#     system_ids = df["system_id"].drop_nulls().unique().to_list()
#     stations = {sid: station_from_kogs(auth, sid) for sid in system_ids}

#     contact_id = df["contact_id"].drop_nulls()[0]
#     reservation = parse_reservation(get_contact(auth, contact_id)["contact"])
#     if not reservation.ephemeris_id:
#         raise ValueError(f"contact {contact_id} has no ephemeris_id")
#     ephemeris = parse_ephemeris(get_TLE(auth, reservation.ephemeris_id))
#     if not ephemeris.inline_oem:
#         raise ValueError(
#             f"ephemeris {ephemeris.ephemeris_uuid} has no inline OEM; "
#             "RK89 needs an OEM initial state"
#         )
#     epoch_unix, pos_km, vel_km_s = parse_oem_state(ephemeris.inline_oem)

#     return build_rk89_input(
#         df,
#         epoch_unix=epoch_unix,
#         pos_km=pos_km,
#         vel_km_s=vel_km_s,
#         stations=stations,
#         force_model=force_model,
#     )
