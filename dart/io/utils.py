"""Shared helpers for backend payload normalization (ported from lib/IO/utils.py)."""
import datetime
import logging
import os

LOGGER_NAME = "logs/dart_production"


def setup_logger():
    """Sets up the logger to write only to a file."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d"
            " - %(funcName)s() - %(message)s"
        )
        os.makedirs(os.path.dirname(LOGGER_NAME), exist_ok=True)
        file_handler = logging.FileHandler(f"{LOGGER_NAME}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


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


def create_api_auth(api_key: str) -> str:
    """Creates auth string for using API key authorization."""
    return f"KSAT1-PLAIN {api_key}"
