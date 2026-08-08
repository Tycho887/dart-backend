#!/usr/bin/env python3
"""Inspect, validate, export, and safely apply live Grafana dashboards."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def api_url(path: str) -> str:
    base = os.getenv("GRAFANA_URL", "http://localhost:3000").rstrip("/")
    return f"{base}{path}"


def auth_headers() -> dict[str, str]:
    token = os.getenv("GRAFANA_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    user = os.getenv("GRAFANA_USER")
    password = os.getenv("GRAFANA_PASSWORD")
    if user is None or password is None:
        raise SystemExit("Set GRAFANA_TOKEN or both GRAFANA_USER and GRAFANA_PASSWORD")
    encoded = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


def request_json(
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    authenticated: bool = True,
) -> Any:
    headers = {"Accept": "application/json"}
    if authenticated:
        headers.update(auth_headers())
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(api_url(path), data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SystemExit(f"Grafana API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cannot reach Grafana: {exc.reason}") from exc
    return json.loads(body) if body else {}


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("Dashboard JSON must be an object")
    return value


def dashboard_from(value: dict[str, Any]) -> dict[str, Any]:
    dashboard = value.get("dashboard", value)
    if not isinstance(dashboard, dict):
        raise SystemExit("The dashboard field must be an object")
    return dashboard


def validate(value: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    dashboard = dashboard_from(value)
    if not dashboard.get("title"):
        errors.append("dashboard.title is required")
    panels = dashboard.get("panels")
    if not isinstance(panels, list):
        return ["dashboard.panels must be an array"], warnings
    seen_ids: set[int] = set()
    for index, panel in enumerate(panels):
        label = f"panels[{index}]"
        if not isinstance(panel, dict):
            errors.append(f"{label} must be an object")
            continue
        panel_id = panel.get("id")
        if not isinstance(panel_id, int):
            errors.append(f"{label}.id must be an integer")
        elif panel_id in seen_ids:
            errors.append(f"duplicate panel id {panel_id}")
        else:
            seen_ids.add(panel_id)
        if not panel.get("type"):
            errors.append(f"{label}.type is required")
        if not panel.get("title"):
            warnings.append(f"{label} has no title")
        grid = panel.get("gridPos")
        if not isinstance(grid, dict):
            errors.append(f"{label}.gridPos must be an object")
        else:
            positions = [grid.get(key) for key in ("x", "y", "w", "h")]
            if not all(isinstance(position, int) for position in positions):
                errors.append(f"{label}.gridPos x/y/w/h must be integers")
            else:
                x, _, width, height = positions
                if x < 0 or width < 1 or height < 1 or x + width > 24:
                    errors.append(f"{label}.gridPos is outside the 24-column grid")
        refs: set[str] = set()
        for target_index, target in enumerate(panel.get("targets", [])):
            if not isinstance(target, dict):
                errors.append(f"{label}.targets[{target_index}] must be an object")
                continue
            ref_id = target.get("refId")
            if ref_id in refs:
                errors.append(f"{label} contains duplicate target refId {ref_id!r}")
            elif isinstance(ref_id, str) and ref_id:
                refs.add(ref_id)
    return errors, list(dict.fromkeys(warnings))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_backup(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise SystemExit(f"Refusing to overwrite existing backup: {path}") from exc


def command_health(_: argparse.Namespace) -> None:
    print(json.dumps(request_json("/api/health", authenticated=False), indent=2))


def command_export(args: argparse.Namespace) -> None:
    value = request_json(f"/api/dashboards/uid/{args.uid}")
    write_json(args.output, value)
    print(f"Exported dashboard {args.uid} to {args.output}")


def command_panels(args: argparse.Namespace) -> None:
    value = request_json(f"/api/dashboards/uid/{args.uid}")
    dashboard = dashboard_from(value)
    print(f"{dashboard.get('title')} ({dashboard.get('uid')}) version {dashboard.get('version')}")
    for panel in dashboard.get("panels", []):
        datasource = panel.get("datasource") or {}
        uid = datasource.get("uid") if isinstance(datasource, dict) else datasource
        print(
            f"{panel.get('id')}\t{panel.get('type')}\t{panel.get('title')}\tdatasource={uid or '-'}"
        )


def command_datasources(_: argparse.Namespace) -> None:
    for datasource in request_json("/api/datasources"):
        print(
            f"{datasource.get('uid')}\t{datasource.get('type')}\t"
            f"{datasource.get('name')}\turl={datasource.get('url') or '-'}"
        )


def command_validate(args: argparse.Namespace) -> None:
    errors, warnings = validate(load_json(args.file))
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        raise SystemExit(1)
    print(f"Valid dashboard JSON ({len(warnings)} warning(s))")


def command_apply(args: argparse.Namespace) -> None:
    if not args.confirm_write:
        raise SystemExit("Refusing to write without --confirm-write")
    value = load_json(args.file)
    errors, warnings = validate(value)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        raise SystemExit("Refusing to apply invalid dashboard JSON: " + "; ".join(errors))
    dashboard = dashboard_from(value)
    uid = dashboard.get("uid")
    if not uid:
        raise SystemExit("Refusing to apply a dashboard without a UID")
    live = request_json(f"/api/dashboards/uid/{uid}")
    live_version = dashboard_from(live).get("version")
    if live_version != dashboard.get("version"):
        raise SystemExit(
            f"Version conflict: live={live_version}, edited={dashboard.get('version')}. "
            "Re-export and merge before applying."
        )
    write_backup(args.backup, live)
    meta = value.get("meta", {})
    folder_uid = meta.get("folderUid", "") if isinstance(meta, dict) else ""
    payload = {
        "dashboard": dashboard,
        "folderUid": folder_uid,
        "message": args.message,
        "overwrite": False,
    }
    print(json.dumps(request_json("/api/dashboards/db", method="POST", payload=payload), indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    health = commands.add_parser("health")
    health.set_defaults(function=command_health)
    export = commands.add_parser("export")
    export.add_argument("--uid", required=True)
    export.add_argument("--output", required=True, type=Path)
    export.set_defaults(function=command_export)
    panels = commands.add_parser("panels")
    panels.add_argument("--uid", required=True)
    panels.set_defaults(function=command_panels)
    datasources = commands.add_parser("datasources")
    datasources.set_defaults(function=command_datasources)
    check = commands.add_parser("validate")
    check.add_argument("--file", required=True, type=Path)
    check.set_defaults(function=command_validate)
    apply = commands.add_parser("apply")
    apply.add_argument("--file", required=True, type=Path)
    apply.add_argument("--backup", required=True, type=Path)
    apply.add_argument("--message", required=True)
    apply.add_argument("--confirm-write", action="store_true")
    apply.set_defaults(function=command_apply)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
