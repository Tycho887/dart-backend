"""Build the small Estimates dashboard for the installed Business Forms plugin."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deploy/grafana"
DS = {"type": "grafana-postgresql-datasource", "uid": "dart-estimates-db"}


def element(name, title, kind="string", value="", **extra):
    return {
        "id": name,
        "uid": name,
        "title": title,
        "type": kind,
        "value": value,
        "labelWidth": 32,
        **extra,
    }


def query(sql):
    return [
        {
            "refId": "A",
            "datasource": DS,
            "format": "table",
            "rawQuery": True,
            "rawSql": sql,
        }
    ]


def table(id, title, y, height, sql):
    return {
        "id": id,
        "title": title,
        "type": "table",
        "datasource": DS,
        "gridPos": {"x": 0, "y": y, "w": 24, "h": height},
        "targets": query(sql),
        "options": {"showHeader": True, "cellHeight": "sm", "footer": {"show": False}},
        "fieldConfig": {
            "defaults": {"custom": {"align": "auto"}, "noValue": "Unavailable"},
            "overrides": [],
        },
    }


def main():
    show = "return context.panel.elements.find(e => e.id === 'advanced')?.value?.includes('show') ?? false;"
    fields = [
        element(
            "contact_ids",
            "Contact UUIDs",
            tooltip="Comma or whitespace separated; one or multiple contacts from the same spacecraft.",
        ),
        element(
            "ephemeris_id",
            "Prior ephemeris UUID (optional)",
            tooltip="Leave blank to use the ephemeris associated with the selected contact. For multiple contacts, uses the latest KOGS contact start time. An explicit UUID overrides this selection.",
        ),
        element(
            "forward_model",
            "Forward model",
            "select",
            "lofi-time@2",
            optionsSource="Code",
            getOptions="return globalThis.dartEstimateForm?.modelOptions() ?? [];",
            options=[],
        ),
        element(
            "optimizer",
            "Optimizer",
            "select",
            "least-squares@1",
            optionsSource="Code",
            getOptions="return globalThis.dartEstimateForm?.optimizerOptions(context.panel.elements) ?? [];",
            options=[],
        ),
        element("summary", "Effective settings", "disabledTextarea"),
        element("label", "Label (optional)"),
        element(
            "advanced",
            "Advanced",
            "checkboxList",
            [],
            optionsSource="Custom",
            options=[
                {
                    "id": "show",
                    "label": "Show settings",
                    "value": "show",
                    "type": "string",
                }
            ],
        ),
    ]
    advanced = [
        ("min_elevation_deg", "Min. elevation (deg)", 1),
        ("min_ebn0_db", "Min. Eb/N0 (dB)", ""),
        ("min_abs_doppler_hz", "Min. |Doppler| (Hz)", 0),
        ("max_abs_doppler_hz", "Max. |Doppler| (Hz)", 100000),
        ("min_samples_per_contact", "Min. samples/contact", 20),
        ("doppler_sigma_hz", "Doppler sigma (Hz)", 1),
        ("nominal_center_frequency_mhz", "Nominal frequency (MHz)", ""),
        ("max_evaluations", "Evaluation limit", ""),
    ]
    fields += [
        element(name, title, "string" if value == "" else "number", value, showIf=show)
        for name, title, value in advanced
    ]
    form = {
        "id": 1,
        "title": "Submit estimate",
        "type": "volkovlabs-form-panel",
        "gridPos": {"x": 0, "y": 0, "w": 24, "h": 20},
        "options": {
            "elements": fields,
            "layout": {"variant": "single", "padding": 12},
            "sync": False,
            "updateEnabled": "manual",
            "initial": {
                "method": "-",
                "code": (SOURCE / "form.js").read_text()
                + "\nreturn globalThis.dartEstimateForm.load(context);",
            },
            "elementValueChanged": "globalThis.dartEstimateForm?.change(context);",
            "update": {
                "method": "-",
                "payloadMode": "custom",
                "confirm": False,
                "code": "return globalThis.dartEstimateForm.submit(context);",
            },
            "submit": {"text": "Run estimate", "icon": "play", "variant": "primary"},
            "reset": {"variant": "hidden"},
            "saveDefault": {"variant": "hidden"},
        },
    }
    where = "WHERE estimate_uuid::text = ${estimate_uuid:sqlstring}"
    recent = table(
        2,
        "Recent estimates",
        20,
        9,
        "SELECT created_at,estimate_uuid,status,spacecraft_name,contact_ids,forward_model_name,optimizer_name,label FROM dart.estimates_v1 WHERE $__timeFilter(created_at) ORDER BY created_at DESC LIMIT 200",
    )
    recent["fieldConfig"]["overrides"] = [
        {
            "matcher": {"id": "byName", "options": "estimate_uuid"},
            "properties": [
                {
                    "id": "links",
                    "value": [
                        {
                            "title": "Inspect estimate",
                            "url": "/d/dart-estimates/estimates?var-estimate_uuid=${__value.raw}",
                        }
                    ],
                }
            ],
        }
    ]
    details = table(
        3,
        "Selected estimate",
        29,
        6,
        "SELECT estimate_uuid,job_id,status,stage,cancel_requested,spacecraft_name,prior_ephemeris_id,epoch,terminal_error FROM dart.estimates_v1 "
        + where,
    )
    parameters = table(
        4,
        "Estimated and considered parameters",
        35,
        10,
        "SELECT parameter_name,role,value,unit,standard_uncertainty,initial_value,lower_bound,upper_bound,contact_id FROM dart.estimate_parameters_v1 "
        + where
        + " ORDER BY ordinal",
    )
    parameters["fieldConfig"]["overrides"] = [
        {
            "matcher": {"id": "byName", "options": "standard_uncertainty"},
            "properties": [{"id": "displayName", "value": "1σ uncertainty"}],
        }
    ]
    diagnostics = table(
        5,
        "Fit diagnostics",
        45,
        6,
        "SELECT success,message,objective,optimality,function_evaluations,observation_count,residual_rms_hz,whitened_residual_rms,covariance_method,covariance_rank,warnings FROM dart.estimate_diagnostics_v1 "
        + where,
    )
    provenance = table(
        6,
        "Prior and profiles",
        51,
        6,
        "SELECT prior_ephemeris_id,epoch,configuration,software_version FROM dart.estimates_v1 "
        + where,
    )
    events = table(
        7,
        "Job events",
        57,
        7,
        "SELECT created_at,event_type,to_status,diagnostic FROM dart.estimate_events_v1 "
        + where
        + " ORDER BY created_at DESC,id DESC LIMIT 100",
    )
    cancel = {
        "id": 8,
        "title": "Cancel selected estimate",
        "type": "volkovlabs-form-panel",
        "datasource": DS,
        "gridPos": {"x": 0, "y": 64, "w": 24, "h": 4},
        "targets": query("SELECT job_id::text,status FROM dart.estimates_v1 " + where),
        "options": {
            "elements": [],
            "layout": {"variant": "none"},
            "updateEnabled": "manual",
            "initial": {
                "method": "-",
                "code": "const status=context.panel.data.series[0]?.fields.find(f=>f.name==='status')?.values[0]; if(['queued','resolving_inputs','loading_telemetry','running'].includes(status)) context.panel.enableSubmit(); else context.panel.disableSubmit();",
            },
            "update": {
                "method": "-",
                "confirm": False,
                "code": "return (async()=>{context.panel.disableSubmit(); try {const id=context.panel.data.series[0]?.fields.find(f=>f.name==='job_id')?.values[0]; if(!id) throw new Error('Select an active estimate'); const r=await fetch('/dart/v1/jobs/'+encodeURIComponent(id)+'/cancel',{method:'POST',credentials:'same-origin',signal:AbortSignal.timeout(30000)}); if(!r.ok) throw new Error(await r.text()); context.grafana.refresh(); context.grafana.notifySuccess(['Cancellation requested',id]);}catch(e){context.grafana.notifyError(['Cancellation failed',e.message]);}})();",
            },
            "submit": {
                "variant": "destructive",
                "text": "Cancel estimate",
                "icon": "times",
            },
            "reset": {"variant": "hidden"},
            "saveDefault": {"variant": "hidden"},
        },
    }
    dashboard = {
        "uid": "dart-estimates",
        "title": "Estimates",
        "schemaVersion": 39,
        "version": 0,
        "tags": ["DART"],
        "timezone": "utc",
        "refresh": "5s",
        "time": {"from": "now-7d", "to": "now"},
        "templating": {
            "list": [
                {
                    "name": "estimate_uuid",
                    "label": "Estimate UUID",
                    "type": "query",
                    "datasource": DS,
                    "refresh": 1,
                    "multi": False,
                    "includeAll": False,
                    "query": "WITH recent AS (SELECT estimate_uuid::text AS __value, COALESCE(label,spacecraft_name,'Estimate') || ' · ' || estimate_uuid::text AS __text, created_at FROM dart.estimates_v1 ORDER BY created_at DESC LIMIT 200) SELECT __value,__text FROM recent UNION ALL SELECT '', 'No estimates yet' WHERE NOT EXISTS (SELECT 1 FROM recent) ORDER BY __text",
                    "current": {"text": "", "value": ""},
                    "options": [],
                }
            ]
        },
        "panels": [
            form,
            recent,
            details,
            parameters,
            diagnostics,
            provenance,
            events,
            cancel,
        ],
    }
    destination = SOURCE / "dashboards/estimates.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        dashboard["id"] = previous.get("id")
        dashboard["version"] = previous.get("version", 0)
    destination.write_text(json.dumps(dashboard, indent=2) + "\n")


if __name__ == "__main__":
    main()
