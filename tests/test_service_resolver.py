"""Automatic prior selection uses contact chronology and preserves provenance."""

from dataclasses import replace
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import UUID

import pytest
import requests
from service_fixtures import configuration, frame_fixture, prior_fixture

from dart.io import kogs
from dart.io.load import LoadError
from dart.service import resolver as resolver_module
from dart.service.config import ServiceSettings
from dart.service.resolver import InputResolver, InputValidationError
from dart.service.worker import _failure


@pytest.fixture
def inputs(monkeypatch):
    config = configuration()
    prior = prior_fixture(config)
    contacts = [
        replace(
            contact,
            ephemeris_id=str(UUID(int=10 + index)),
            ephemeris=replace(prior.ephemeris, ephemeris_id=str(UUID(int=10 + index))),
        )
        for index, contact in enumerate(prior.observations.contacts.values())
    ]
    load = AsyncMock(return_value=(contacts, frame_fixture(prior)))
    get_ephemeris = Mock(return_value=prior.ephemeris)
    monkeypatch.setattr(resolver_module, "load_passes", load)
    monkeypatch.setattr(resolver_module.adx, "client_from_env", MagicMock())
    monkeypatch.setattr(resolver_module.kogs, "get_ephemeris", get_ephemeris)
    resolver = InputResolver(ServiceSettings("unused", kogs_api_key="test"))
    return resolver, config, contacts, load, get_ephemeris


@pytest.mark.parametrize("offset", [-60, 0, 60])
@pytest.mark.parametrize("reverse", [False, True])
def test_automatic_prior_uses_latest_start_independent_of_order(
    inputs, offset, reverse
):
    resolver, config, contacts, load, get_ephemeris = inputs
    contacts[1] = replace(
        contacts[1], start=contacts[1].start + timedelta(seconds=offset)
    )
    expected = contacts[1] if offset > 0 else contacts[0]
    if reverse:
        contacts.reverse()
    request = config.request.model_copy(
        update={
            "ephemeris_id": None,
            "contact_ids": [UUID(contact.contact_id) for contact in contacts],
        }
    )
    config = config.model_copy(update={"request": request})
    prepared = resolver.prepare(config)
    assert prepared.prior.ephemeris == expected.ephemeris
    assert prepared.provenance["prior_selection"] == {
        "source": "latest_contact",
        "contact_id": expected.contact_id,
        "contact_start": expected.start.isoformat(),
        "ephemeris_id": expected.ephemeris_id,
    }
    assert [*prepared.prior.observations.contacts] == [c.contact_id for c in contacts]
    get_ephemeris.assert_not_called()
    load.assert_awaited_once()
    assert config.request.ephemeris_id is None


def test_single_contact_uses_its_associated_ephemeris(inputs):
    resolver, config, contacts, load, get_ephemeris = inputs
    contact = contacts[0]
    frame = load.return_value[1].filter(
        resolver_module.pl.col("contact_id") == contact.contact_id
    )
    load.return_value = ([contact], frame)
    request = config.request.model_copy(
        update={"ephemeris_id": None, "contact_ids": [UUID(contact.contact_id)]}
    )
    prepared = resolver.prepare(config.model_copy(update={"request": request}))
    assert prepared.prior.ephemeris == contact.ephemeris
    get_ephemeris.assert_not_called()


def test_explicit_prior_overrides_contact_ephemerides(inputs):
    resolver, config, contacts, load, get_ephemeris = inputs
    prepared = resolver.prepare(config)
    assert prepared.prior.ephemeris == get_ephemeris.return_value
    assert prepared.provenance["prior_selection"] == {
        "source": "request",
        "ephemeris_id": str(config.request.ephemeris_id),
    }
    get_ephemeris.assert_called_once_with(
        "test", str(config.request.ephemeris_id), timeout_seconds=30
    )


@pytest.mark.parametrize(
    "problem,match",
    [
        ("spacecraft", "one spacecraft"),
        ("tle", "TLE prior"),
        ("identity", "identity mismatch"),
    ],
)
def test_automatic_prior_rejects_unusable_latest_without_fallback(
    inputs, problem, match
):
    resolver, config, contacts, load, get_ephemeris = inputs
    latest = contacts[0]
    if problem == "spacecraft":
        contacts[1] = replace(contacts[1], spacecraft_id="another-spacecraft")
    elif problem == "tle":
        contacts[0] = replace(latest, ephemeris=replace(latest.ephemeris, tle=None))
    else:
        contacts[0] = replace(latest, ephemeris_id="")
    request = config.request.model_copy(update={"ephemeris_id": None})
    with pytest.raises(InputValidationError, match=match):
        resolver.prepare(config.model_copy(update={"request": request}))
    get_ephemeris.assert_not_called()


def test_missing_linked_ephemeris_has_safe_actionable_error(inputs):
    resolver, config, contacts, load, get_ephemeris = inputs
    error = LoadError("KOGS", contacts[0].contact_id, "secret-provider-detail")
    error.__cause__ = kogs.KogsError("secret-provider-detail")
    load.side_effect = error
    with pytest.raises(InputValidationError, match="associated ephemeris") as captured:
        resolver.prepare(config)
    assert "secret-provider-detail" not in _failure(captured.value)[0]["detail"]


def test_transient_provider_failure_still_retries(inputs):
    resolver, config, contacts, load, get_ephemeris = inputs
    error = LoadError("KOGS", contacts[0].contact_id, "secret-provider-detail")
    error.__cause__ = requests.Timeout("secret-provider-detail")
    load.side_effect = error
    with pytest.raises(LoadError) as captured:
        resolver.prepare(config)
    assert _failure(captured.value)[1] is True
