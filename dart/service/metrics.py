"""Low-cardinality metrics; estimate identifiers belong in audit records."""

from prometheus_client import Counter, Gauge, Histogram

JOBS_SUBMITTED = Counter("dart_jobs_submitted_total", "Submitted estimates", ["model"])
JOB_OUTCOMES = Counter("dart_job_outcomes_total", "Job outcomes", ["outcome"])
STAGE_LATENCY = Histogram("dart_job_stage_seconds", "Worker stage latency", ["stage"])
QUEUE_DEPTH = Gauge("dart_queue_depth", "Queued estimates")
QUEUE_AGE = Gauge("dart_oldest_queued_job_seconds", "Oldest queued estimate age")
