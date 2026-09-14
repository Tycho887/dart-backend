"""Frozen, credential-free inputs for the estimate service."""

from datetime import timedelta
from uuid import UUID

import numpy as np
import polars as pl
import satkit as sk

from dart.forward_models import (
    evaluate_full_state_augmented,
    evaluate_sgp4_augmented,
    tle_state_gcrf,
)
from dart.io import ContactMetadata, EphemerisMetadata, ForwardModelContext
from dart.od import PriorStateData
from dart.service.models import EstimateRequest, ProfileRef
from dart.service.profiles import (
    forward_model_profiles,
    optimizer_profiles,
    resolve_configuration,
)

TLE = (
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
)
CONTACTS = [UUID(int=2), UUID(int=1)]
EPHEMERIS = UUID(int=3)


def configuration(model="lofi-time", optimizer="least-squares", multipass=True):
    request = EstimateRequest(
        contact_ids=CONTACTS if multipass else CONTACTS[:1],
        ephemeris_id=EPHEMERIS,
        forward_model=ProfileRef(name=model),
        optimizer=ProfileRef(name=optimizer),
        nominal_center_frequency_hz=400e6,
    )
    models = {p.name: p for p in forward_model_profiles()}
    optimizers = {p.name: p for p in optimizer_profiles()}
    return resolve_configuration(request, models[model], optimizers[optimizer])


def prior_fixture(config):
    epoch = sk.TLE.from_lines(list(TLE)).epoch
    start = epoch.as_datetime()
    ephemeris = EphemerisMetadata(
        str(EPHEMERIS),
        "test-spacecraft",
        "TLE",
        "fixture",
        None,
        start,
        None,
        None,
        None,
        "\n".join(TLE),
        None,
        None,
        False,
        None,
    )
    context = ForwardModelContext(400e6)
    for i, cid in enumerate(config.request.contact_ids):
        contact = ContactMetadata(
            "test-spacecraft",
            f"system-{i}",
            f"station-{i}",
            str(EPHEMERIS),
            f"ANT{i}",
            "test",
            60 - 80 * i,
            10 + 100 * i,
            0,
            (1, 2, 3),
            "TEST",
            "1998-067A",
            "25544",
            start,
            start + timedelta(hours=1),
            str(cid),
            ephemeris,
        )
        context.register_contact(contact)
    for n in range(30):
        for i, cid in enumerate(config.request.contact_ids):
            context.add_observation(
                epoch.as_unixtime() + 60 + n * 30, 0, 4, f"system-{i}", str(cid)
            )
    prior = PriorStateData(
        context,
        ephemeris,
        min(o.time for o in context.observations) - sk.duration(seconds=1),
    )
    passes = len(config.request.contact_ids)
    if config.forward_model.model == "sgp4":
        target = np.zeros(9 + passes)
        target[9:] = 8
        evaluation = evaluate_sgp4_augmented(target, TLE, context)
    else:
        nominal = tle_state_gcrf(TLE, prior.epoch)
        target = np.zeros(8 + passes)
        target[8:] = 8
        evaluation = evaluate_full_state_augmented(
            target, nominal, prior.epoch, context
        )
    for observation, value in zip(
        context.observations, evaluation.residuals * 2, strict=True
    ):
        observation.observed[0] = float(value)
    return prior


def frame_fixture(prior):
    return pl.DataFrame(
        [
            {
                "timestamp": o.time.as_datetime(),
                "contact_id": o.contact_id,
                "spacecraft_id": "test-spacecraft",
                "system_id": prior.observations.contacts[o.contact_id].system_id,
                "antenna_name": "TEST",
                "tracking_epoch_offset_s": 0.0,
                "azimuth_deg": 20.0,
                "elevation_deg": 40.0,
                "carrier_lock": "Locked",
                "ebn0": 10.0,
                "doppler_hz": o.observed[0],
            }
            for o in prior.observations.observations
        ]
    )
