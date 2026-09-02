"""Shared helpers for backend payload normalization (ported from lib/IO/utils.py)."""
import datetime


def _safe_str(value) -> str | None:
    if value is None:
        return None
    return str(value)


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_bool(value) -> bool:
    return bool(value)


def _join_field(items, key) -> str | None:
    if not items:
        return None
    vals = [str(item.get(key)) for item in items if key in item and item.get(key) is not None]
    return ",".join(vals) if vals else None


def _join_list(items) -> str | None:
    if items is None:
        return None
    if isinstance(items, list):
        vals = [str(x) for x in items if x is not None]
        return ",".join(vals) if vals else None
    return _safe_str(items)


def _iso_to_unix(iso_str) -> float | None:
    if not iso_str:
        return None
    s = iso_str
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
    ]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return dt.timestamp()
        except Exception:
            continue
    try:
        dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def utc(value: object, label: str) -> datetime.datetime:
    """Coerce a string or datetime to an aware UTC datetime."""
    if isinstance(value, str):
        parsed = datetime.datetime.fromisoformat(value)
    elif isinstance(value, datetime.datetime):
        parsed = value
    else:
        raise ValueError(f"invalid {label} epoch {value!r}")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=datetime.UTC)
    return parsed.astimezone(datetime.UTC)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_utc(value: datetime.datetime) -> datetime.datetime:
    """Reject naive or non-UTC timestamps and return an aware UTC datetime."""
    if value.tzinfo is None or value.utcoffset() != datetime.timedelta(0):
        raise ValueError("timestamp must be timezone-aware UTC")
    return value.astimezone(datetime.UTC)


def parse_utc(value: object) -> datetime.datetime:
    """Parse an ISO-8601 timestamp (naive means UTC) into aware UTC."""
    return require_utc(
        datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )


def utc_text(value: datetime.datetime) -> str:
    """Format an aware UTC datetime as a Z-suffixed ISO-8601 string."""
    return require_utc(value).isoformat().replace("+00:00", "Z")



