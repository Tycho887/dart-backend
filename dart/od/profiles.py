"""Reusable orbit/bias study profiles, in physical units."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

from . import _FULL_STATE_PARAMETER_NAMES, _SGP4_PARAMETER_NAMES
from .schema import OptimizerContext, OrbitModel, ParameterSpec

FOREST_VARIANCE_HZ2 = 500.0**2


def forest_profile(
    profile: OptimizerContext,
    *,
    variance_hz2: float = FOREST_VARIANCE_HZ2,
    loss_scale_hz: float = 700.0,
) -> OptimizerContext:
    """FOREST noise/loss and optimizer settings, retaining model-specific scales."""
    import math

    if not all(math.isfinite(v) and v > 0 for v in (variance_hz2, loss_scale_hz)):
        raise ValueError("variance and loss scale must be finite and positive")
    return replace(
        profile,
        parameters=tuple(
            replace(p, lower_bound=-100000, upper_bound=100000, scale=5000)
            if p.name.startswith("pass_bias_hz:")
            else p
            for p in profile.parameters
        ),
        loss="soft_l1",
        loss_scale=loss_scale_hz / math.sqrt(variance_hz2),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        x_scale="profile",
    )


def time_offset_profile(
    contact_id: str, *, max_evaluations: int = 1000, bound_s: float = 120.0
) -> OptimizerContext:
    """Single-pass measurement-time shift, including station geometry."""
    return forest_profile(
        OptimizerContext(
            OrbitModel.SGP4,
            (
                ParameterSpec("time_offset_s", 0, -bound_s, bound_s, 30),
                ParameterSpec(f"pass_bias_hz:{contact_id}", 0, -100000, 100000, 5000),
            ),
            max_evaluations=max_evaluations,
        )
    )


# The existing burst-radio six/hifi study uses these same bounds and scales.
# They are experiment configuration, not estimates of prior uncertainty.
ORBIT_SCALES = {
    OrbitModel.SGP4: (0.001, 0.001, 0.001, 0.001, 0.001, 0.1),
    OrbitModel.FULL_STATE: (1e4, 1e4, 1e4, 10.0, 10.0, 10.0),
}
ORBIT_BOUNDS = {
    OrbitModel.SGP4: (0.2, 0.1, 0.1, 0.1, 0.1, 30.0),
    OrbitModel.FULL_STATE: (1e6, 1e6, 1e6, 1000.0, 1000.0, 1000.0),
}

Sgp4ParameterSet = Literal["L", "L+n", "six"]


def sgp4_bias_profile(
    parameter_set: Sgp4ParameterSet,
    contact_id: str | Sequence[str],
    *,
    robust: bool = False,
    max_evaluations: int = 1000,
) -> OptimizerContext:
    """Selected orbit corrections plus one bias; omitted corrections stay zero.

    The robust transition is 200 Hz for unit-variance Doppler observations.
    Bounds and scales are identical to the six-parameter control.
    """
    from dataclasses import replace

    contacts = (contact_id,) if isinstance(contact_id, str) else tuple(contact_id)
    if not contacts or len(set(contacts)) != len(contacts):
        raise ValueError("contact IDs must be nonempty and unique")
    selected = {
        "L": {"mean_longitude_deg"},
        "L+n": {"mean_longitude_deg", "mean_motion_rev_per_day"},
        "six": set(_SGP4_PARAMETER_NAMES[:6]),
    }[parameter_set] | {f"pass_bias_hz:{cid}" for cid in contacts}
    control = orbit_bias_profile(
        OrbitModel.SGP4, contacts, max_evaluations=max_evaluations
    )
    return replace(
        control,
        parameters=tuple(p for p in control.parameters if p.name in selected),
        loss="soft_l1" if robust else "linear",
        loss_scale=200.0 if robust else 1.0,
    )


def orbit_bias_profile(
    model: OrbitModel,
    contact_ids: Sequence[str],
    *,
    max_evaluations: int = 1000,
    robust: bool = False,
) -> OptimizerContext:
    """Six orbit corrections and one bias/contact; other corrections fixed at zero."""
    names = {
        OrbitModel.SGP4: _SGP4_PARAMETER_NAMES[:6],
        OrbitModel.FULL_STATE: _FULL_STATE_PARAMETER_NAMES,
    }[model]
    orbit = tuple(
        ParameterSpec(name, 0, -bound, bound, scale)
        for name, bound, scale in zip(
            names, ORBIT_BOUNDS[model], ORBIT_SCALES[model], strict=True
        )
    )
    biases = tuple(
        ParameterSpec(f"pass_bias_hz:{cid}", 0, -150000, 150000, 200)
        for cid in contact_ids
    )
    return OptimizerContext(
        model,
        orbit + biases,
        loss="soft_l1" if robust else "linear",
        loss_scale=200.0 if robust else 1.0,
        max_evaluations=max_evaluations,
    )


def sgp4_epoch_bias_profile(
    contact_ids: Sequence[str],
    *,
    bound_s: float = 600.0,
    robust: bool = False,
    max_evaluations: int = 1000,
) -> OptimizerContext:
    """Fit E' = E + offset plus pass biases; observation clocks stay fixed."""
    from dataclasses import replace

    control = sgp4_bias_profile(
        "L", contact_ids, robust=robust, max_evaluations=max_evaluations
    )
    biases = tuple(p for p in control.parameters if p.name.startswith("pass_bias_hz:"))
    epoch = ParameterSpec("tle_epoch_offset_s", 0, -bound_s, bound_s, 1)
    return replace(control, parameters=(epoch, *biases))
