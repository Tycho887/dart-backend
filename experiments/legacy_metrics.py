"""Deprecation notice shared by frozen FOREST archive/regression entry points.

Do not add metrics or fitting behavior here. New studies use experiment.py v5.
"""

import warnings


def warn_legacy_metrics(feature: str) -> None:
    warnings.warn(
        f"{feature} is deprecated and retained only for historical archive/regression replay. "
        "Use experiment.py v5: TLE epoch correction and post-contact forecast-hour OEM position/velocity RMSE. "
        "See docs/benchmark.md#deprecated-experiments-and-metrics.",
        FutureWarning,
        stacklevel=3,
    )
