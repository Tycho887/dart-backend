"""Read-only ctrl-config v2 access: spacecraft link frequencies and qradio REST.

Configuration paths are rooted at the project's ``ctrl-config/v2`` directory by
default; callers that resolve configs from another deployment root pass it
explicitly. Only names that are bare file names are accepted, so a caller can
never escape the configured root through a path-traversal name.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

_V2_ROOT = Path(__file__).resolve().parents[2] / "ctrl-config" / "v2"
OBSERVED_LINK_NAME = "s_band_downlink_p1_1"


def _path(root: Path, kind: str, name: str) -> Path:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"invalid {kind} configuration name {name!r}")
    return (root / kind / f"{name}.yml").resolve()


def spacecraft_config_path(root: Path, name: str) -> Path:
    return _path(root, "spacecrafts", name)


def system_config_path(root: Path, name: str) -> Path:
    return _path(root, "system", name)


def has_spacecraft_config(root: Path, name: str) -> bool:
    return spacecraft_config_path(root, name).is_file()


def has_system_config(root: Path, name: str) -> bool:
    return system_config_path(root, name).is_file()


def qradio_rest_url(root: Path, system_name: str) -> str:
    """Find the REST address of a system's qradio from ctrl-config."""
    path = system_config_path(root, system_name)
    if not path.is_file():
        raise FileNotFoundError(f"No config found for system '{system_name}'.")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    defaults = data.get("defaults")
    if isinstance(defaults, dict) and "qradio" in defaults:
        return defaults["qradio"].get("rest")
    if isinstance(defaults, list):
        for entry in defaults:
            if isinstance(entry, dict) and "qradio" in entry:
                return entry["qradio"].get("rest")

    raise ValueError(
        f"'qradio' with 'rest' not found in the config for '{system_name}'."
    )


def get_link_frequency(
    spacecraft_name: str,
    link_name: str,
    direction: str,
    *,
    root: Path = _V2_ROOT,
) -> float:
    """Return one positive ctrl-config link frequency after checking direction."""
    if direction not in {"up", "down"}:
        raise ValueError("link direction must be 'up' or 'down'")
    path = spacecraft_config_path(root, spacecraft_name)
    if not path.is_file():
        raise FileNotFoundError(f"No config found for spacecraft {spacecraft_name!r}.")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    link = data.get("links", {}).get(link_name)
    if not isinstance(link, dict):
        raise ValueError(f"link {link_name!r} is missing from {spacecraft_name}.yml")
    if link.get("direction") != direction:
        raise ValueError(f"link {link_name!r} is not a {direction}link")
    try:
        frequency = float(link["frequency"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"link {link_name!r} has no numeric frequency") from exc
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError(f"link {link_name!r} frequency must be positive and finite")
    return frequency


def get_observed_frequency(spacecraft_name: str, *, root: Path = _V2_ROOT) -> float:
    """Return the primary S-band downlink frequency for one spacecraft."""
    return get_link_frequency(spacecraft_name, OBSERVED_LINK_NAME, "down", root=root)
