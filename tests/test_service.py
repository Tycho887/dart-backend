"""Behavioral contract tests for profiles, replay, API and normalization."""

from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from service_fixtures import (
    CONTACTS,
    EPHEMERIS,
    configuration,
    frame_fixture,
    prior_fixture,
)

from dart.od import fit
from dart.service.api import create_app
from dart.service.config import ServiceSettings
from dart.service.models import EstimateAccepted, EstimateRequest
from dart.service.profiles import (
    forward_model_profiles,
    optimizer_profiles,
    resolve_configuration,
)
from dart.service.resolver import prepare_prior
from dart.service.serialization import (
    document,
    prior_document,
    restore_optimizer,
    restore_prior,
    result_rows,
)
from dart.service.worker import run_estimate


@pytest.mark.parametrize(
    "model", ["lofi-time", "lofi-time-frequency", "lofi-elements", "hifi"]
)
def test_frozen_fit_replay_and_parameter_units(model):
    config = configuration(model)
    prior = prior_fixture(config)
    output, optimizer, scan = run_estimate(prior, config)
    replay = fit(
        restore_prior(prior_document(prior)), restore_optimizer(document(optimizer))
    )
    np.testing.assert_allclose(output.parameters, replay.parameters, atol=1e-8)
    rows, diagnostics = result_rows(prior, config, optimizer, output)
    assert [p["parameter_name"] for p in rows] == list(output.parameter_names)
    assert [p["contact_id"] for p in rows[-2:]] == [str(cid) for cid in CONTACTS]
    assert rows[-1]["unit"] == "Hz"
    assert all(p["standard_uncertainty"] is None for p in rows)
    assert diagnostics["covariance"] is None
    assert diagnostics["residual_rms_hz"] == pytest.approx(
        2 * diagnostics["whitened_residual_rms"]
    )
    assert diagnostics["residual_rms_hz"] < 0.05
    assert not scan


def test_profiles_preserve_zero_overrides_and_reject_incompatible_initialization():
    config = configuration()
    body = config.request.model_dump(mode="json")
    body["measurement_selection"]["min_elevation_deg"] = 0
    request = EstimateRequest.model_validate(body)
    assert request.measurement_selection.min_elevation_deg == 0
    with pytest.raises(ValueError, match="incompatible"):
        configuration("hifi", "timing-scan")
    with pytest.raises(ValidationError):
        EstimateRequest.model_validate({**body, "regularization": "gaussian"})
    with pytest.raises(ValidationError, match="unique"):
        EstimateRequest.model_validate({**body, "contact_ids": [str(CONTACTS[0])] * 2})


def test_prepare_prior_identity_selection_and_explicit_epoch():
    config = configuration()
    prior = prior_fixture(config)
    contacts = list(prior.observations.contacts.values())
    frame = frame_fixture(prior)
    result = prepare_prior(config, contacts, frame, prior.ephemeris, 400e6)
    assert result.epoch == prior.epoch
    assert list(result.observations.contact_to_pass_idx) == [
        str(cid) for cid in CONTACTS
    ]
    with pytest.raises(ValueError, match="one spacecraft"):
        prepare_prior(
            config,
            [replace(contacts[0], spacecraft_id="different"), contacts[1]],
            frame,
            prior.ephemeris,
            400e6,
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        prepare_prior(
            config,
            contacts,
            frame,
            replace(prior.ephemeris, ephemeris_id="wrong"),
            400e6,
        )
    with pytest.raises(ValueError, match="insufficient"):
        prepare_prior(config, contacts, frame.head(1), prior.ephemeris, 400e6)


class ProfileDatabase:
    def resolve_profiles(self, m, mv, o, ov):
        return (
            {(p.name, p.version): p for p in forward_model_profiles()}[m, mv],
            {(p.name, p.version): p for p in optimizer_profiles()}[o, ov],
        )

    def list_profiles(self, family):
        values = (
            forward_model_profiles()
            if family == "forward_model"
            else optimizer_profiles()
        )
        return [p.model_dump(mode="json") for p in values]

    def submit_estimate(self, configuration, actor_id, actor_type, key):
        assert actor_id == "grafana:1:7" and actor_type == "human"
        assert key == "test-submission"
        return EstimateAccepted(
            job_id=EPHEMERIS,
            estimate_uuid=CONTACTS[0],
            status="queued",
            idempotent_replay=False,
        )


def test_api_gateway_validation_and_submission():
    settings = ServiceSettings("unused", gateway_token="test-gateway-only")
    with TestClient(
        create_app(settings=settings, database=ProfileDatabase())
    ) as client:
        body = configuration().request.model_dump(mode="json")
        assert client.post("/v1/estimate-jobs", json=body).status_code == 403
        headers = {
            "X-DART-Gateway-Token": "test-gateway-only",
            "X-DART-Actor-ID": "grafana:1:7",
            "X-DART-Actor-Type": "human",
            "Idempotency-Key": "test-submission",
        }
        response = client.post("/v1/estimate-jobs/validate", json=body, headers=headers)
        assert (
            response.status_code == 200
            and response.json()["input_validation"] == "deferred_to_worker"
        )
        assert (
            client.post("/v1/estimate-jobs", json=body, headers=headers).status_code
            == 202
        )
        body["optimizer"]["name"] = "missing"
        response = client.post("/v1/estimate-jobs", json=body, headers=headers)
        assert (
            response.status_code == 422
            and response.json()["code"] == "profile_not_found"
        )
        assert (
            client.post("/v1/solve-jobs", json={}, headers=headers).status_code == 410
        )


def test_rejects_ineffective_overrides_and_invalid_covariance():
    config = configuration()
    request = config.request.model_copy(
        update={
            "optimizer_overrides": config.request.optimizer_overrides.model_copy(
                update={"loss_scale": 2.0}
            )
        }
    )
    with pytest.raises(ValueError, match="only effective"):
        resolve_configuration(request, config.forward_model, config.optimizer)
    prior = prior_fixture(config)
    output, optimizer, _ = run_estimate(prior, config)
    covariance = -np.eye(len(output.parameters))
    with pytest.raises(ValueError, match="negative parameter variance"):
        result_rows(prior, config, optimizer, replace(output, covariance=covariance))


@pytest.mark.parametrize(
    "status,retryable", [(400, False), (401, False), (429, True), (503, True)]
)
def test_provider_retries_do_not_persist_provider_secrets(status, retryable):
    import requests

    from dart.io.load import LoadError
    from dart.service.worker import _failure

    response = requests.Response()
    response.status_code = status
    original = requests.HTTPError("secret=do-not-store", response=response)
    error = LoadError("KOGS", "contact", "provider failure")
    error.__cause__ = original
    diagnostic, actual = _failure(error)
    assert actual is retryable
    assert "do-not-store" not in str(diagnostic)
