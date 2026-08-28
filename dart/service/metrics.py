"""Low-cardinality service metrics; job IDs belong in logs and events."""

from prometheus_client import Counter, Gauge, Histogram

JOBS_SUBMITTED = Counter(
    "dart_jobs_submitted_total", "Submitted DART jobs", ["solver_kind"]
)
JOB_OUTCOMES = Counter(
    "dart_job_outcomes_total", "Terminal DART job outcomes", ["outcome"]
)
JOB_RETRIES = Counter("dart_job_retries_total", "DART job retries", ["error_code"])
EXTERNAL_FAILURES = Counter(
    "dart_external_failures_total",
    "External service failures",
    ["service", "error_code"],
)
STAGE_LATENCY = Histogram(
    "dart_job_stage_seconds", "DART worker stage latency", ["stage"]
)
QUEUE_DEPTH = Gauge("dart_queue_depth", "Queued DART jobs")
QUEUE_AGE = Gauge("dart_oldest_queued_job_seconds", "Age of oldest queued DART job")
SOLVER_CONVERGENCE = Counter(
    "dart_solver_convergence_total",
    "Solver convergence results",
    ["solver_kind", "converged"],
)
