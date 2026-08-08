from dataclasses import asdict
from typing import Any, Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from lib.getTelemetry import augment_telemetry_dataframe
from lib.azure import fetch_tracking_data
from lib.load import TrackingContext
from lib.metadata_proxy import (
    MetadataLookupError,
    MetadataNotFound,
    fetch_contact_metadata,
    fetch_ephemeris_metadata,
    log_metadata_event,
)
from lib.parseKogs import EphemerisData, ReservationData
from lib.settings import cors_origins
from processor import process_telemetry_batch

app = FastAPI(
    title="DASH Telemetry Processor",
    description="Fetches, enriches, and fits spacecraft tracking telemetry.",
)

allowed_origins = cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=["X-Request-ID"],
)


class ContactLookupRequest(BaseModel):
    contact_id: str = Field(min_length=1, max_length=200)


class EphemerisLookupRequest(BaseModel):
    ephemeris_id: str = Field(min_length=1, max_length=200)


def _metadata_lookup(
    fetcher: Callable[..., Any],
    identifier: str,
    request: Request,
    response: Response,
    body: dict[str, str] | None = None,
):
    request_id = str(uuid4())
    response.headers["X-Request-ID"] = request_id
    request_packet = {
        "method": request.method,
        "path": request.url.path,
        "headers": dict(request.headers),
        "body": body,
    }
    log_metadata_event(
        "info",
        100,
        "proxy_request",
        request_id,
        **request_packet,
    )

    try:
        result = fetcher(identifier, request_id=request_id)
    except ValueError as exc:
        log_metadata_event(
            "warning",
            400,
            "proxy_response",
            request_id,
            status=400,
            body={"detail": str(exc)},
        )
        raise HTTPException(
            status_code=400,
            detail=str(exc),
            headers={"X-Request-ID": request_id},
        ) from exc
    except MetadataNotFound as exc:
        log_metadata_event(
            "warning",
            404,
            "proxy_response",
            request_id,
            status=404,
            body={"detail": str(exc)},
        )
        raise HTTPException(
            status_code=404,
            detail=str(exc),
            headers={"X-Request-ID": request_id},
        ) from exc
    except MetadataLookupError as exc:
        log_metadata_event(
            "error",
            502,
            "proxy_response",
            request_id,
            status=502,
            body={"detail": str(exc)},
        )
        raise HTTPException(
            status_code=502,
            detail=str(exc),
            headers={"X-Request-ID": request_id},
        ) from exc

    response_body = asdict(result)
    log_metadata_event(
        "info",
        200,
        "proxy_response",
        request_id,
        status=200,
        headers={"X-Request-ID": request_id},
        body=response_body,
    )
    return result


@app.get(
    "/api/metadata/contact/{contact_id}",
    response_model=ReservationData,
    tags=["metadata"],
)
def get_contact_metadata(
    contact_id: str,
    request: Request,
    response: Response,
) -> ReservationData:
    """Return normalized contact metadata for one KOGS contact UUID."""

    return _metadata_lookup(fetch_contact_metadata, contact_id, request, response)


@app.post(
    "/api/metadata/contact",
    response_model=ReservationData,
    tags=["metadata"],
)
def post_contact_metadata(
    lookup: ContactLookupRequest,
    request: Request,
    response: Response,
) -> ReservationData:
    """Dashboard-friendly contact lookup with a JSON request body."""

    return _metadata_lookup(
        fetch_contact_metadata,
        lookup.contact_id,
        request,
        response,
        body={"contact_id": lookup.contact_id},
    )


@app.get(
    "/api/metadata/ephemeris/{ephemeris_id}",
    response_model=EphemerisData,
    tags=["metadata"],
)
def get_ephemeris_metadata(
    ephemeris_id: str,
    request: Request,
    response: Response,
) -> EphemerisData:
    """Return normalized ephemeris metadata for one KOGS ephemeris UUID."""

    return _metadata_lookup(fetch_ephemeris_metadata, ephemeris_id, request, response)


@app.post(
    "/api/metadata/ephemeris",
    response_model=EphemerisData,
    tags=["metadata"],
)
def post_ephemeris_metadata(
    lookup: EphemerisLookupRequest,
    request: Request,
    response: Response,
) -> EphemerisData:
    """Dashboard-friendly ephemeris lookup with a JSON request body."""

    return _metadata_lookup(
        fetch_ephemeris_metadata,
        lookup.ephemeris_id,
        request,
        response,
        body={"ephemeris_id": lookup.ephemeris_id},
    )


@app.post(
    "/api/tracking",
    responses={
        400: {"description": "Invalid selection or solver options"},
        404: {"description": "No matching telemetry"},
        502: {"description": "Upstream metadata service failure"},
    },
)
def receive_spacecraft_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Process one Grafana tracking request.

    FastAPI runs this synchronous endpoint in a worker thread, so the blocking
    ADX, metadata, and PostgreSQL clients do not stall the event loop.
    """

    try:
        context = TrackingContext.from_payload(payload)
        telemetry = fetch_tracking_data(context)
        if telemetry.is_empty():
            raise HTTPException(status_code=404, detail="No telemetry matched the request")

        augmented = augment_telemetry_dataframe(telemetry)
        if not context.spacecraft_uuid:
            context.spacecraft_uuid = augmented["spacecraft_id"][0]

        results = process_telemetry_batch(augmented, context=context)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    completed = len(results["time_offset_passes"]) + len(results["mean_elements_windows"])
    if not results["errors"]:
        status, message = "success", "Telemetry processed successfully"
    elif completed:
        status, message = "partial", "Telemetry processed with some stage failures"
    else:
        status, message = "failed", "No telemetry processing stage completed"

    return {
        "status": status,
        "message": message,
        "results": results,
    }
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
