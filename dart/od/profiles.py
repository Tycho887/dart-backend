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
