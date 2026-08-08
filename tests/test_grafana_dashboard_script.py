import argparse
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "tools" / "grafana" / "dashboard.py"
SPEC = importlib.util.spec_from_file_location("grafana_dashboard_script", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
grafana_dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grafana_dashboard)


def _dashboard(version: int) -> dict:
    return {
        "dashboard": {
            "panels": [],
            "title": "Test dashboard",
            "uid": "test-dashboard",
            "version": version,
        },
        "meta": {},
    }


def _args(file: Path, backup: Path) -> argparse.Namespace:
    return argparse.Namespace(
        backup=backup,
        confirm_write=True,
        file=file,
        message="Test guarded apply",
    )


def test_apply_rejects_stale_version_without_creating_a_backup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(_dashboard(4)), encoding="utf-8")
    backup = tmp_path / "backup.json"
    monkeypatch.setattr(grafana_dashboard, "request_json", lambda *_args, **_kwargs: _dashboard(5))

    with pytest.raises(SystemExit, match="Version conflict"):
        grafana_dashboard.command_apply(_args(edited, backup))

    assert not backup.exists()


def test_apply_refuses_to_overwrite_an_existing_backup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(_dashboard(5)), encoding="utf-8")
    backup = tmp_path / "backup.json"
    backup.write_text("original backup", encoding="utf-8")
    requests: list[str] = []

    def request_json(path: str, **_kwargs: object) -> dict:
        requests.append(path)
        return _dashboard(5)

    monkeypatch.setattr(grafana_dashboard, "request_json", request_json)

    with pytest.raises(SystemExit, match="Refusing to overwrite existing backup"):
        grafana_dashboard.command_apply(_args(edited, backup))

    assert backup.read_text(encoding="utf-8") == "original backup"
    assert requests == ["/api/dashboards/uid/test-dashboard"]
