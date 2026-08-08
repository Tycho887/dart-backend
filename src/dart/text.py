"""Small text-safety helpers shared by production transports and workers."""

from __future__ import annotations

TRUNCATION_SUFFIX = " [truncated]"


def bounded_text(value: object, maximum_length: int, fallback: str) -> str:
    """Return non-empty, UTF-8-safe text that fits one contract field."""

    try:
        text = str(value)
    except Exception:
        text = fallback
    if not text:
        text = fallback
    text = text.encode("utf-8", "replace").decode("utf-8")
    if len(text) <= maximum_length:
        return text
    if maximum_length <= len(TRUNCATION_SUFFIX):
        return TRUNCATION_SUFFIX[:maximum_length]
    return text[: maximum_length - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX
