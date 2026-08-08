"""Postprocessor API conversions between its local wire model and semantic domain."""

from __future__ import annotations

from pydantic import BaseModel

from .. import contracts as domain
from ..wire import postprocessor as postprocessor_wire


def _payload(value: BaseModel) -> object:
    """Serialize one validated model for a boundary conversion."""

    return value.model_dump(mode="json")


def postprocessor_request_to_domain(
    value: postprocessor_wire.QualityRequest,
) -> domain.QualityRequest:
    """Apply semantic validation to a postprocessor wire request."""

    return domain.QualityRequest.model_validate(_payload(value))


def postprocessor_result_to_wire(value: domain.QualityResult) -> postprocessor_wire.QualityResult:
    """Prepare a semantic quality result for its local HTTP response."""

    return postprocessor_wire.QualityResult.model_validate(_payload(value))
