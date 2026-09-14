"""Versioned model/optimizer profiles composed into the existing OD contract."""

from typing import Literal

from dart.od import OptimizerContext, OrbitModel, ParameterRole, ParameterSpec
from dart.od.profiles import orbit_bias_profile, sgp4_bias_profile

from .models import (
    EstimateRequest,
    ForwardModelProfile,
    OptimizerProfile,
    ParameterDefinition,
    ResolvedEstimateConfiguration,
)


def parameter_unit(name: str) -> str:
    if name.startswith("pass_bias_hz:"):
        return "Hz"
    units = {
        "time_offset_s": "s",
        "center_frequency_offset_hz": "Hz",
        "mean_longitude_deg": "deg",
        "mean_motion_rev_per_day": "rev/day",
        "position_x_m": "m",
        "position_y_m": "m",
        "position_z_m": "m",
        "velocity_x_m_s": "m/s",
        "velocity_y_m_s": "m/s",
        "velocity_z_m_s": "m/s",
    }
    return units[name]


def _definition(parameter: ParameterSpec) -> ParameterDefinition:
    return ParameterDefinition(
        name=parameter.name,
        unit=parameter_unit(parameter.name),
        initial=parameter.initial,
        lower_bound=parameter.lower_bound,
        upper_bound=parameter.upper_bound,
        scale=parameter.scale,
        role=parameter.role.value,
    )


def forward_model_profiles() -> list[ForwardModelProfile]:
    time = ParameterDefinition(
        name="time_offset_s", unit="s", lower_bound=-600, upper_bound=600, scale=1
    )
    frequency = ParameterDefinition(
        name="center_frequency_offset_hz",
        unit="Hz",
        lower_bound=-1e6,
        upper_bound=1e6,
        scale=1000,
    )
    bias = ParameterDefinition(
        name="pass_bias_hz:{contact_id}",
        unit="Hz",
        lower_bound=-150000,
        upper_bound=150000,
        scale=200,
    )
    element = sgp4_bias_profile("L+n", ["template"]).parameters[:-1]
    cartesian = orbit_bias_profile(OrbitModel.FULL_STATE, ["template"]).parameters[:-1]
    definitions: list[
        tuple[str, str, Literal["sgp4", "full_state"], list[ParameterDefinition], str]
    ] = [
        (
            "lofi-time",
            "lofi: time",
            "sgp4",
            [time],
            "Time offset and one bias per contact; source orbit is unchanged.",
        ),
        (
            "lofi-time-frequency",
            "lofi: time + f",
            "sgp4",
            [time, frequency],
            "Time and center-frequency offsets plus one bias per contact.",
        ),
        (
            "lofi-elements",
            "lofi: L + n",
            "sgp4",
            [_definition(p) for p in element],
            "SGP4 mean longitude and mean motion corrections plus contact biases.",
        ),
        (
            "hifi",
            "hifi",
            "full_state",
            [_definition(p) for p in cartesian],
            "Six GCRF Cartesian corrections plus contact biases; prior initialized from the selected TLE.",
        ),
    ]
    return [
        ForwardModelProfile(
            name=n,
            label=label,
            model=model,
            parameters=params,
            pass_bias=bias,
            description=description,
        )
        for n, label, model, params, description in definitions
    ]


def optimizer_profiles() -> list[OptimizerProfile]:
    names = [p.name for p in forward_model_profiles()]
    return [
        OptimizerProfile(
            name="least-squares", label="Least squares", compatible_models=names
        ),
        OptimizerProfile(
            name="robust",
            label="Robust (soft L1)",
            compatible_models=names,
            loss="soft_l1",
            loss_scale=200,
        ),
        OptimizerProfile(
            name="timing-scan",
            label="Timing scan + least squares",
            compatible_models=["lofi-time"],
            initialization="timing_scan",
        ),
        OptimizerProfile(
            name="phase-scan",
            label="Phase scan + least squares",
            compatible_models=["lofi-elements"],
            initialization="phase_scan",
        ),
    ]


def resolve_configuration(
    request: EstimateRequest,
    model: ForwardModelProfile,
    optimizer: OptimizerProfile,
) -> ResolvedEstimateConfiguration:
    if (model.name, model.version) != (
        request.forward_model.name,
        request.forward_model.version,
    ):
        raise ValueError("forward-model profile identity mismatch")
    if (optimizer.name, optimizer.version) != (
        request.optimizer.name,
        request.optimizer.version,
    ):
        raise ValueError("optimizer profile identity mismatch")
    if model.name not in optimizer.compatible_models:
        raise ValueError(
            "optimizer initialization is incompatible with the selected forward model"
        )
    if (
        request.optimizer_overrides.loss_scale is not None
        and optimizer.loss == "linear"
    ):
        raise ValueError("loss_scale is only effective for robust-loss profiles")
    effective = optimizer.model_copy(
        update=request.optimizer_overrides.model_dump(exclude_none=True)
    )
    parameters = [p.model_copy(deep=True) for p in model.parameters]
    parameters.extend(
        model.pass_bias.model_copy(update={"name": f"pass_bias_hz:{cid}"})
        for cid in request.contact_ids
    )
    return ResolvedEstimateConfiguration(
        request=request, forward_model=model, optimizer=effective, parameters=parameters
    )


def optimizer_context(configuration: ResolvedEstimateConfiguration) -> OptimizerContext:
    profile = configuration.optimizer
    parameters = tuple(
        ParameterSpec(
            p.name,
            p.initial,
            p.lower_bound,
            p.upper_bound,
            p.scale,
            ParameterRole(p.role),
        )
        for p in configuration.parameters
    )
    return OptimizerContext(
        model=OrbitModel(configuration.forward_model.model),
        parameters=parameters,
        loss=profile.loss,
        loss_scale=profile.loss_scale,
        max_evaluations=profile.max_evaluations,
        ftol=profile.ftol,
        xtol=profile.xtol,
        gtol=profile.gtol,
    )


def capabilities_document() -> dict:
    return {
        "contract_version": 1,
        "operations": ["estimate"],
        "max_contacts": 100,
        "explicit_prior_required": True,
        "same_spacecraft_required": True,
        "carrier_lock_required": True,
        "regularization": ["none"],
        "products": {"tdm": False, "oem": False},
        "result_access": "grafana_sql_views",
        "epoch_policy": "one_second_before_earliest_retained_observation",
        "override_schema": EstimateRequest.model_json_schema()["$defs"][
            "OptimizerOverrides"
        ],
    }
