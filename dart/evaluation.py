"""Reference-state comparisons, independent of fitting and data acquisition."""

from dataclasses import dataclass

import numpy as np

from dart.orbit import FloatArray, StateHistory


@dataclass(frozen=True)
class OrbitError:
    sample_count: int
    position_rms_m: float
    velocity_rms_m_s: float
    position_max_m: float
    velocity_max_m_s: float
    differences: FloatArray


def compare_states(predicted: StateHistory, reference: StateHistory) -> OrbitError:
    """Compute sample-weighted predicted-minus-reference errors in GCRF/SI.

    Both inputs must identify the same object and exactly the same epochs. No
    alignment, interpolation, time-offset fitting, or extrapolation occurs here.
    """
    if predicted.object_id != reference.object_id:
        raise ValueError("state history object identities differ")
    if predicted.epochs != reference.epochs:
        raise ValueError("comparison requires identical epochs in identical order")
    difference = predicted.states - reference.states
    position = np.linalg.norm(difference[:, :3], axis=1)
    velocity = np.linalg.norm(difference[:, 3:], axis=1)
    return OrbitError(
        len(reference.epochs),
        float(np.sqrt(np.mean(position**2))),
        float(np.sqrt(np.mean(velocity**2))),
        float(position.max()),
        float(velocity.max()),
        difference,
    )
