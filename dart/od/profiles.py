"""Shared six-orbit-parameter study configuration, in physical units."""

from collections.abc import Sequence

from . import _FULL_STATE_PARAMETER_NAMES, _SGP4_PARAMETER_NAMES
from .schema import OptimizerContext, OrbitModel, ParameterSpec

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


def orbit_bias_profile(
    model: OrbitModel, contact_ids: Sequence[str], *, max_evaluations: int = 1000
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
        model, orbit + biases, loss="linear", max_evaluations=max_evaluations
    )


def time_offset_profile(
    contact_id: str, *, max_evaluations: int = 1000
) -> OptimizerContext:
    """Single-pass SGP4 timing/bias fit; use observation variance 500**2 Hz².

    All orbit elements and the center-frequency correction stay at zero.
    Whitened soft-L1 scale 1.4 gives the historical 700 Hz transition.
    """
    return OptimizerContext(
        OrbitModel.SGP4,
        (
            ParameterSpec("time_offset_s", 0, -120, 120, 30),
            ParameterSpec(f"pass_bias_hz:{contact_id}", 0, -100000, 100000, 5000),
        ),
        loss="soft_l1",
        loss_scale=1.4,
        max_evaluations=max_evaluations,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
