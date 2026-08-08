import json
import re
from pathlib import Path

DASHBOARDS = Path(__file__).parents[1] / "dashboards"
ADX_UID = "ffo40r3smd81sb"
POSTGRES_UID = "ffo3xwq8aqt4we"


def _dashboard(name: str) -> dict:
    value = json.loads((DASHBOARDS / name).read_text())
    return value["dashboard"]


def _panel(dashboard: dict, panel_id: int) -> dict:
    return next(panel for panel in dashboard["panels"] if panel["id"] == panel_id)


def test_metadata_dashboard_uses_same_origin_proxy_routes():
    dashboard = _dashboard("metadata-explorer.json")
    urls = {panel["options"]["update"]["url"] for panel in dashboard["panels"]}

    assert urls == {
        "/dart-api/v0/metadata/contact",
        "/dart-api/v0/metadata/ephemeris",
    }


def test_metadata_dashboard_reset_reports_an_unconditional_clear() -> None:
    dashboard = _dashboard("metadata-explorer.json")

    for panel in dashboard["panels"]:
        reset_action = panel["options"]["resetAction"]
        assert reset_action["mode"] == "initial"
        assert reset_action["getPayload"] == "return {}"
        assert "context.panel.response" not in reset_action["code"]
        assert "Form values cleared." in reset_action["code"]


def test_initial_processing_dashboard_uses_reviewed_run_contract():
    dashboard = _dashboard("initial-processing.json")
    form = _panel(dashboard, 1)
    elements = form["options"]["elements"]
    element_ids = {element["id"] for element in elements}
    update = form["options"]["update"]
    payload = update["getPayload"]

    assert update["url"] == "/dart-api/v0/runs"
    assert update["payloadMode"] == "custom"
    assert {
        "spacecraft_id",
        "contact_ids",
        "reference_tle_name",
        "reference_tle_line1",
        "reference_tle_line2",
        "nominal_carrier_frequency_hz",
        "candidate_models",
        "doppler_standard_deviation_hz",
        "robust_loss",
        "robust_scale_hz",
        "qmc_samples",
        "selection_criterion",
    } <= element_ids
    assert not {
        "problemType",
        "modelType",
        "pointingConstraintEnabled",
        "angleConstraint",
        "penaltyWeight",
    }.intersection(element_ids)
    assert len(element_ids) == len(elements)

    candidate_models = next(element for element in elements if element["id"] == "candidate_models")
    assert [option["value"] for option in candidate_models["options"]] == [
        "time_offset",
        "time_offset_pass_bias",
        "time_offset_frequency_pass_bias",
        "mean_elements_two_parameter",
    ]
    assert "two distinct passes" in candidate_models["tooltip"]

    assert "schema_version: '0.1'" in payload
    assert "optimizer_configuration" in payload
    assert "candidate_metaparameters" in payload
    assert "selection_criterion" in payload
    assert "orderedModels" in payload
    assert "value === null || value === undefined || value === ''" in payload
    assert "contactIds.length === 0" in payload
    assert "query.spacecraft_id = spacecraftId" in payload
    assert "require_lock: checked('require_lock')" in payload
    assert "qmc_samples: qmcSamples" in payload
    assert "problem_type" not in payload
    assert "optimizer_data" not in payload
    assert "use_qmc" not in payload


def test_result_panels_use_the_canonical_selected_result_view():
    dashboard = _dashboard("initial-processing.json")
    time_panel = _panel(dashboard, 2)
    mean_panel = _panel(dashboard, 4)
    time_sql = time_panel["targets"][0]["rawSql"]
    mean_sql = mean_panel["targets"][0]["rawSql"]
    sql = f"{time_sql}\n{mean_sql}"

    assert time_panel["datasource"]["uid"] == POSTGRES_UID
    assert mean_panel["datasource"]["uid"] == POSTGRES_UID
    assert "dart_run_results" in sql
    assert "selected_result_json" in sql
    assert "selected_metrics_json" in sql
    assert "time_offset" in time_sql
    assert "time_offset_pass_bias" in time_sql
    assert "time_offset_frequency_pass_bias" in time_sql
    assert "center_frequency_correction_hz" in time_sql
    assert "pass_biases" in time_sql
    assert "mean_elements_two_parameter" in mean_sql
    assert "corrected_tle" in mean_sql
    assert "mean_anomaly_rad" in mean_sql
    assert "mean_motion_rad_min" in mean_sql

    assert re.search(r"(?<!selected_)result_json->", sql) is None
    assert re.search(r"(?<!selected_)metrics_json->", sql) is None
    for stale_field in ("problem_type", "selected_model", "doppler_bias_hz", "mean_elements'"):
        assert stale_field not in sql


def test_direct_adx_recent_pass_panel_remains_direct_and_read_only():
    dashboard = _dashboard("initial-processing.json")
    panel = _panel(dashboard, 3)
    target = panel["targets"][0]

    assert panel["title"] == "Most Recent Passes"
    assert panel["datasource"]["uid"] == ADX_UID
    assert target["datasource"]["uid"] == ADX_UID
    assert target["queryType"] == "KQL"
    assert "/dart-api/" not in json.dumps(panel)


def test_provisioned_dashboards_match_verified_live_exports():
    for name in ("metadata-explorer.json", "initial-processing.json"):
        provisioned = json.loads((DASHBOARDS / "provisioned" / name).read_text())
        assert provisioned == _dashboard(name)
