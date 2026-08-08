"""Stateless optimizer HTTP service."""

from fastapi import Depends, FastAPI
from pydantic import ValidationError

from ..services.optimizer import solve_batch
from ..wire import solver as solver_wire
from .openapi import install_contract_openapi_rules
from .optimizer_conversion import solver_request_to_domain, solver_result_to_wire
from .problems import domain_problem, install_problem_handlers, problem_responses
from .security import require_internal_bearer

app = FastAPI(title="DART Solver", version="0.1")
install_problem_handlers(app)
install_contract_openapi_rules(app, "solver-v0.1.json")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v0/solve/batch",
    response_model=solver_wire.BatchResult,
    dependencies=[Depends(require_internal_bearer)],
    responses=problem_responses(401, 422, 503),
)
def batch(
    request: solver_wire.BatchRequest,
) -> solver_wire.BatchResult:
    """Validate one solver wire request and run its explicitly selected model."""

    try:
        return solver_result_to_wire(solve_batch(solver_request_to_domain(request)))
    except (ValidationError, ValueError) as exc:
        raise domain_problem(str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run("dart.api.optimizer:app", host="0.0.0.0", port=8001)
