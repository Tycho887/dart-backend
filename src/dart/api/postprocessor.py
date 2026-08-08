"""Stateless residual-metric and CCSDS OEM postprocessor service."""

from fastapi import Depends, FastAPI
from pydantic import ValidationError

from ..quality import assess_quality
from ..wire import postprocessor as postprocessor_wire
from .openapi import install_contract_openapi_rules
from .postprocessor_conversion import postprocessor_request_to_domain, postprocessor_result_to_wire
from .problems import domain_problem, install_problem_handlers, problem_responses
from .security import require_internal_bearer

app = FastAPI(title="DART Postprocessor", version="0.1")
install_problem_handlers(app)
install_contract_openapi_rules(app, "postprocessor-v0.1.json")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v0/postprocess",
    response_model=postprocessor_wire.QualityResult,
    dependencies=[Depends(require_internal_bearer)],
    responses=problem_responses(401, 422, 503),
)
def postprocess(
    request: postprocessor_wire.QualityRequest,
) -> postprocessor_wire.QualityResult:
    """Validate one postprocessor wire request and calculate its metrics."""

    try:
        domain_request = postprocessor_request_to_domain(request)
        return postprocessor_result_to_wire(assess_quality(domain_request))
    except (ValidationError, ValueError) as exc:
        raise domain_problem(str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run("dart.api.postprocessor:app", host="0.0.0.0", port=8002)
