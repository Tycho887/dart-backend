"""Transport schema — THE CONTRACT between Python and the Rust solver.

Rules (do not break without a schema version bump):

1. Field names are the documentation. A Python field name == msgpack key ==
   serde field name == CCSDS TDM keyword, all snake_case.
2. One unit convention, the CCSDS one: km, km/s, Hz, degrees. Epochs are
   f64 unix-seconds in UTC (floating point, not integer).
3. Nothing crosses the boundary that is not data: primitive scalars,
   strings, and lists/tuples of them only. No numpy arrays, no polars
   DataFrames, no satkit objects. Conversion happens in ``dart.loaders``.
4. ``schema_version`` is the first field of every message. Bump it on any
   incompatible change; both sides reject mismatched versions loudly.

The Rust mirror of these dataclasses lives in
``crates/dart_solver/src/schema.rs`` and must be kept in lockstep; the
round-trip fixture tests (``tests/test_codec.py`` and
``crates/dart_solver/tests/roundtrip.rs``) enforce that contract.
"""

from dataclasses import dataclass, field

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Station:
    """Ground station, geodetic WGS-84. ``id`` keys ``Observation.station_id``."""

    id: str
    name: str | None = None
    lat_deg: float = 0.0
    lon_deg: float = 0.0
    alt_km: float = 0.0


@dataclass(frozen=True)
class Observation:
    """One tracking observation at a station.

    ``range_km`` is optional and only expected for cislunar (RK89) passes.
    """

    epoch_unix: float
    doppler_hz: float
    azimuth_deg: float
    elevation_deg: float
    station_id: str
    contact_id: str
    range_km: float | None = None


@dataclass(frozen=True)
class Tle:
    """Two-line element set as plain strings (never parsed at the boundary)."""

    line1: str
    line2: str


@dataclass(frozen=True)
class ForceModel:
    """RK89 (cislunar) force model selection. ``gravity_deg`` is the central
    body harmonic cutoff: 0 = point mass, 4 = zonal/tesseral up to degree 4."""

    gravity_deg: int = 4
    third_body: bool = True  # Sun + Moon point masses
    srp: bool = False
    step_s: float = 60.0  # initial integrator step


@dataclass(frozen=True)
class SolverOptions:
    """Numeric control for the least-squares fit, shared by both modes."""

    max_iterations: int = 10
    tolerance: float = 1e-8
    ref_frame: str = "TEME"  # "TEME" for sgp4, "EME2000" for rk89


@dataclass(frozen=True)
class FitParameter:
    """One bounded optimizer variable, in the physical units named by its use."""

    initial: float = 0.0
    lower: float = -1.0
    upper: float = 1.0
    scale: float = 1.0
    finite_difference_step: float = 1e-6


@dataclass(frozen=True)
class Sgp4FitOptions:
    """Configuration for the bounded mean-element Doppler fit.

    ``model`` is one of ``mean_anomaly``, ``mean_anomaly_mean_motion`` or
    ``mean_anomaly_mean_motion_frequency``.  Pass-bias specs are aligned with
    the ordered, unique ``pass_ids`` list.
    """

    model: str = "mean_anomaly"
    pass_ids: list[str] = field(default_factory=list)
    nominal_center_frequency_hz: float = 0.0
    mean_anomaly: FitParameter = FitParameter(
        initial=0.0, lower=-0.05, upper=0.05, scale=0.02, finite_difference_step=1e-6
    )
    mean_motion: FitParameter = FitParameter(
        initial=0.0,
        lower=-3.3333333333333335e-5,
        upper=3.3333333333333335e-5,
        scale=1.6666666666666667e-5,
        finite_difference_step=1.6666666666666667e-9,
    )
    center_frequency: FitParameter = FitParameter(
        initial=0.0,
        lower=-5e8,
        upper=5e8,
        scale=5e8,
        finite_difference_step=1e4,
    )
    pass_biases: list[FitParameter] = field(default_factory=list)
    doppler_sigma_hz: float = 1e3
    loss: str = "soft_l1"  # linear, huber, soft_l1, log_cosh
    loss_scale: float = 1.0
    max_evaluations: int = 200
    ftol_rel: float = 1e-6
    xtol_rel: float = 1e-6


@dataclass(frozen=True)
class Sgp4Input:
    """LEO batch: a single TLE propagated over station observations.

    ``epoch_unix`` is the reference epoch the fitted state is expressed at
    (normally the TLE epoch). Propagation itself runs through satkit's SGP4
    in Rust through satkit; this struct is what the solver consumes.
    """

    schema_version: int = SCHEMA_VERSION
    mode: str = "sgp4"
    spacecraft_id: str = ""
    epoch_unix: float = 0.0
    tle: Tle = Tle("", "")
    stations: list[Station] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    options: SolverOptions = SolverOptions()
    fit: Sgp4FitOptions = Sgp4FitOptions()


@dataclass(frozen=True)
class Rk89Input:
    """Cislunar batch: initial ECI state (``ref_frame``) integrated with an
    8(9) adaptive Runge-Kutta over the station observations."""

    schema_version: int = SCHEMA_VERSION
    mode: str = "rk89"
    spacecraft_id: str = ""
    epoch_unix: float = 0.0
    pos_km: tuple[float, float, float] = (0.0, 0.0, 0.0)
    vel_km_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    force_model: ForceModel = ForceModel()
    stations: list[Station] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    options: SolverOptions = SolverOptions()


@dataclass(frozen=True)
class SolverResult:
    """Solver output, mirror-shaped so Python can process it directly.

    ``pos_km``/``vel_km_s`` are the fitted (or final propagated) state in
    ``ref_frame``; ``covariance`` is the flattened 6x6 (position then
    velocity), row-major; ``residuals`` is one entry per input observation,
    in input order. SGP4 fits additionally report the ordered physical fit
    parameters and their row-major robust sandwich covariance.
    """

    schema_version: int = SCHEMA_VERSION
    mode: str = ""
    success: bool = False
    message: str = ""
    converged: bool = False
    iterations: int = 0
    rms: float = 0.0
    epoch_unix: float = 0.0
    pos_km: tuple[float, float, float] = (0.0, 0.0, 0.0)
    vel_km_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    covariance: tuple[float, ...] = ()  # 36 entries, row-major 6x6
    residuals: tuple[float, ...] = ()  # per-observation, input order
    objective: float = 0.0
    function_evaluations: int = 0
    gradient_evaluations: int = 0
    parameter_names: tuple[str, ...] = ()
    parameters: tuple[float, ...] = ()
    parameter_covariance: tuple[float, ...] = ()
    covariance_rank: int = 0
    fitted_tle: Tle | None = None
