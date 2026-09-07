"""Public interface for contact resolution and forward-model observation ingestion."""

from .contact import (
    BatchEvaluationResult,
    ContactMetadata,
    ForwardModelContext,
    ForwardObservation,
    MeasurementKind,
    load_contact_metadata,
)

__all__ = [
    "BatchEvaluationResult",
    "ContactMetadata",
    "ForwardModelContext",
    "ForwardObservation",
    "MeasurementKind",
    "load_contact_metadata",
]