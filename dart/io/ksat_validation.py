"""Independent structural validator for rendered KSAT CCSDS TDM KVN text."""

from __future__ import annotations

import datetime as dt
import math
import re

from dart.io.ksat_tdm import KsatProduct

_KV = re.compile(r"^(?P<key>[A-Z][A-Z0-9_]*) = (?P<value>.+)$")
_OBSERVATIONS = {
    KsatProduct.TRACK: frozenset({"RANGE", "TRANSMIT_FREQ_1", "RECEIVE_FREQ_1"}),
    KsatProduct.ANGLE: frozenset({"ANGLE_1", "ANGLE_2"}),
    KsatProduct.SIGMET: frozenset({"CARRIER_POWER", "PC_N0", "PR_N0"}),
}
_REQUIRED_METADATA = {
    KsatProduct.TRACK: frozenset(
        {
            "TIME_SYSTEM",
            "PARTICIPANT_1",
            "PARTICIPANT_2",
            "MODE",
            "TRANSMIT_BAND",
            "RECEIVE_BAND",
            "INTEGRATION_INTERVAL",
            "INTEGRATION_REF",
            "CORRECTIONS_APPLIED",
            "PATH",
        }
    ),
    KsatProduct.ANGLE: frozenset(
        {"TIME_SYSTEM", "PARTICIPANT_1", "PARTICIPANT_2", "RECEIVE_BAND", "ANGLE_TYPE"}
    ),
    KsatProduct.SIGMET: frozenset(
        {
            "TIME_SYSTEM",
            "PARTICIPANT_1",
            "PARTICIPANT_2",
            "TRANSMIT_BAND",
            "RECEIVE_BAND",
        }
    ),
}


def _validate_observation(line: str, allowed: frozenset[str]) -> None:
    match = _KV.fullmatch(line)
    if match is None or match.group("key") not in allowed:
        raise ValueError(f"invalid observation line {line!r}")
    parts = match.group("value").split()
    if len(parts) != 2:
        raise ValueError(f"observation must contain epoch and value: {line!r}")
    try:
        dt.datetime.fromisoformat(parts[0].replace("Z", "+00:00"))
        numeric = float(parts[1])
    except ValueError as exc:
        raise ValueError(f"invalid observation value {line!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"non-finite observation value {line!r}")


def validate_ksat_tdm_text(text: str, product: KsatProduct) -> None:
    """Reject malformed block structure or values outside one KSAT product grammar."""

    if not isinstance(text, str):
        raise TypeError("TDM text must be a string")
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("KSAT TDM text must be ASCII") from exc
    if not text.endswith("\n"):
        raise ValueError("KSAT TDM text must end with a newline")
    lines = text.splitlines()
    if not lines or lines[0] != "CCSDS_TDM_VERS = 2.0":
        raise ValueError("KSAT TDM must start with CCSDS_TDM_VERS = 2.0")
    first_meta = lines.index("META_START") if "META_START" in lines else -1
    if first_meta < 0:
        raise ValueError("KSAT TDM contains no metadata segment")
    header = lines[:first_meta]
    if "ORIGINATOR = KSAT" not in header:
        raise ValueError("KSAT TDM header must contain ORIGINATOR = KSAT")
    if sum(line.startswith("CREATION_DATE = ") for line in header) != 1:
        raise ValueError("KSAT TDM header must contain one CREATION_DATE")

    state = "between"
    metadata: dict[str, str] = {}
    has_tracking_mode = False
    observation_count = 0
    segment_count = 0
    allowed = _OBSERVATIONS[product]
    for line in lines[first_meta:]:
        if not line:
            continue
        if line == "META_START":
            if state != "between":
                raise ValueError("nested or misplaced META_START")
            state = "meta"
            metadata = {}
            has_tracking_mode = False
            continue
        if line == "META_STOP":
            if state != "meta":
                raise ValueError("misplaced META_STOP")
            missing = sorted(_REQUIRED_METADATA[product] - metadata.keys())
            if missing:
                raise ValueError("metadata segment is missing " + ", ".join(missing))
            if metadata["TIME_SYSTEM"] != "UTC":
                raise ValueError("KSAT TDM metadata TIME_SYSTEM must be UTC")
            if product is KsatProduct.TRACK:
                if metadata["MODE"] != "SEQUENTIAL":
                    raise ValueError("KSAT TRACK MODE must be SEQUENTIAL")
                if metadata["INTEGRATION_REF"] != "END":
                    raise ValueError("KSAT TRACK INTEGRATION_REF must be END")
                if metadata["CORRECTIONS_APPLIED"] != "NO":
                    raise ValueError("KSAT TRACK corrections must remain unapplied")
                if metadata["PATH"] != "1,2,1":
                    raise ValueError("KSAT TRACK PATH must be 1,2,1")
            if product is KsatProduct.ANGLE and not has_tracking_mode:
                raise ValueError("KSAT ANGLE metadata requires TRACKING_MODE comment")
            state = "await_data"
            continue
        if line == "DATA_START":
            if state != "await_data":
                raise ValueError("misplaced DATA_START")
            state = "data"
            observation_count = 0
            continue
        if line == "DATA_STOP":
            if state != "data" or observation_count == 0:
                raise ValueError("empty or misplaced DATA_STOP")
            state = "between"
            segment_count += 1
            continue
        if state == "meta":
            if line.startswith("COMMENT "):
                if line.startswith("COMMENT TRACKING_MODE = "):
                    mode = line.removeprefix("COMMENT TRACKING_MODE = ")
                    if mode not in {"AUTO", "PROGRAM", "SCAN"}:
                        raise ValueError(f"unsupported KSAT tracking mode {mode!r}")
                    has_tracking_mode = True
                continue
            match = _KV.fullmatch(line)
            if match is None:
                raise ValueError(f"invalid metadata line {line!r}")
            metadata[match.group("key")] = match.group("value")
            continue
        if state == "data":
            _validate_observation(line, allowed)
            observation_count += 1
            continue
        raise ValueError(f"unexpected line outside a segment: {line!r}")
    if state != "between" or segment_count == 0:
        raise ValueError("KSAT TDM contains an incomplete segment")


__all__ = ["validate_ksat_tdm_text"]
