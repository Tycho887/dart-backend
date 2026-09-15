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
    assert diagnostics["covariance_method"] is None
    assert diagnostics["covariance_rank"] is None
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


def test_v2_profiles_cover_every_parameter_with_covariance_priors():
    expected_counts = {"sgp4": 9, "full_state": 8}
    profiles = [profile for profile in forward_model_profiles() if profile.version == 2]
    assert len(profiles) == 4
    for profile in profiles:
        parameters = [*profile.parameters, profile.pass_bias]
        names = [parameter.name for parameter in parameters]
        assert len(profile.parameters) == expected_counts[profile.model]
        assert len(names) == len(set(names))
        assert all(
            parameter.role in {"estimate", "consider"} for parameter in parameters
        )
        assert all(
            parameter.prior_standard_uncertainty is not None
            and parameter.prior_standard_uncertainty > 0
            for parameter in parameters
        )


@pytest.mark.parametrize(
    "model,optimizer",
    [
        ("lofi-time", "least-squares"),
        ("lofi-time", "robust"),
        ("lofi-time", "timing-scan"),
        ("lofi-time-frequency", "least-squares"),
        ("lofi-elements", "phase-scan"),
        ("hifi", "least-squares"),
    ],
)
def test_v2_service_methods_publish_full_consider_covariance(model, optimizer):
    config = configuration(model, optimizer, model_version=2)
    prior = prior_fixture(config)
    output, initialized, _ = run_estimate(prior, config)
    rows, diagnostics = result_rows(prior, config, initialized, output)
    assert output.success
    assert output.covariance is not None
    assert output.covariance.shape == (len(output.parameters),) * 2
    assert diagnostics["covariance_method"] == "classical_consider_v1"
    assert diagnostics["parameter_order"] == output.parameter_names
    assert all(row["standard_uncertainty"] is not None for row in rows)
    assert any(row["role"] == "consider" for row in rows)
    np.testing.assert_allclose(
        [row["standard_uncertainty"] for row in rows],
        np.sqrt(np.diag(output.covariance)),
    )
    for row, spec in zip(rows, initialized.parameters, strict=True):
        if row["role"] == "consider":
            assert row["value"] == spec.initial
            assert row["standard_uncertainty"] == pytest.approx(
                spec.prior_standard_uncertainty
            )


@pytest.mark.parametrize("optimizer", ["least-squares", "robust"])
def test_v2_covariance_does_not_change_point_estimate(optimizer):
    legacy = configuration("lofi-time", optimizer, model_version=1)
    enabled = configuration("lofi-time", optimizer, model_version=2)
    prior = prior_fixture(legacy)
    old, _, _ = run_estimate(prior, legacy)
    new, _, _ = run_estimate(prior, enabled)
    new_values = dict(zip(new.parameter_names, new.parameters, strict=True))
    np.testing.assert_allclose(
        old.parameters,
        [new_values[name] for name in old.parameter_names],
        rtol=0,
        atol=1e-10,
    )


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


@pytest.mark.parametrize("prior_mode", ["explicit", "omitted", "null"])
def test_api_gateway_validation_and_submission(prior_mode):
    settings = ServiceSettings("unused", gateway_token="test-gateway-only")
    with TestClient(
        create_app(settings=settings, database=ProfileDatabase())
    ) as client:
        body = configuration().request.model_dump(mode="json")
        if prior_mode == "omitted":
            body.pop("ephemeris_id")
        elif prior_mode == "null":
            body["ephemeris_id"] = None
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
        assert response.json()["configuration"]["request"]["ephemeris_id"] == body.get(
            "ephemeris_id"
        )
        capabilities = client.get("/v1/capabilities", headers=headers).json()
        assert capabilities["explicit_prior_required"] is False
        assert capabilities["default_prior_source"] == "latest_contact"
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
