"""Bounded raw-telemetry access used by KSAT TDM exporters."""

from dart.io import azure


def fetch_contact_columns(*args, **kwargs):
    """Delegate to the shared ADX helper while preserving one bounded query path."""
    return azure.fetch_contact_columns(*args, **kwargs)

__all__ = ["fetch_contact_columns"]
