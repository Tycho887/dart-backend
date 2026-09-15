"""Allowlisted exception diagnostics; never format provider exception messages."""

import json
import logging
import re
from traceback import walk_tb

from .database import ClaimedJob


def exception_chain(exc: BaseException) -> list[BaseException]:
    """Return the visible chain, outermost first, without following cycles."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return chain


def exception_summary(exc: BaseException) -> dict[str, str]:
    cause = exception_chain(exc)[-1]
    summary = {"exception_type": type(cause).__name__}
    if isinstance(cause, ModuleNotFoundError) and cause.name:
        if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", cause.name, re.ASCII):
            summary["missing_module"] = cause.name
    return summary


def _stack(exc: BaseException) -> dict[str, object]:
    # Walking frames directly avoids source lookups, local values, exception
    # notes, and __str__ calls (all of which can expose request credentials).
    return {
        "exception_type": type(exc).__name__,
        "frames": [
            {
                "file": frame.f_code.co_filename,
                "function": frame.f_code.co_name,
                "line": line,
            }
            for frame, line in walk_tb(exc.__traceback__)
        ],
    }


def log_failure(
    logger: logging.Logger,
    exc: BaseException,
    *,
    phase: str,
    worker_id: str,
    job: ClaimedJob | None = None,
    level: int = logging.ERROR,
) -> None:
    context: dict[str, str | int] = {
        "event": "worker_failure",
        "worker_id": worker_id,
        "phase": phase,
        **exception_summary(exc),
        "exception_type": type(exc).__name__,
    }
    if job is not None:
        context.update(
            estimate_uuid=str(job.estimate_uuid),
            job_id=str(job.id),
            run_id=str(job.run_id),
            attempt=job.attempt_count,
        )
    # One log record keeps correlation fields and all frames together, even
    # with concurrent heartbeats or multiple workers. No raw exc_info is sent.
    logger.log(
        level,
        "%s",
        json.dumps({**context, "traceback": [_stack(e) for e in exception_chain(exc)]}),
        extra=context,
    )
