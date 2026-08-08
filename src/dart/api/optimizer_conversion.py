"""Solver API conversions between its local wire model and the semantic domain."""

from __future__ import annotations

from pydantic import BaseModel

from .. import contracts as domain
from ..wire import solver as solver_wire


def _payload(value: BaseModel) -> object:
    """Serialize one validated model for a boundary conversion."""

    return value.model_dump(mode="json")


def solver_request_to_domain(value: solver_wire.BatchRequest) -> domain.BatchRequest:
    """Apply semantic validation to a solver wire request."""

    return domain.BatchRequest.model_validate(_payload(value))


def solver_result_to_wire(value: domain.BatchResult) -> solver_wire.BatchResult:
    """Prepare a semantic solver result for its local HTTP response."""

    return solver_wire.BatchResult.model_validate(_payload(value))
