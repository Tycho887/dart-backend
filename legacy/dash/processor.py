import polars as pl
import numpy as np
import satkit as sk
from typing import Any
from lib.utils import Data, Config, Result
from lib.time_model import solve_time_shift
from lib.load import TrackingContext
import psycopg2
from psycopg2.extras import Json
from datetime import datetime
from lib.db_logger import ProcessLogger
# Import mean element solver and TLE utilities
from sgp4.api import Satrec
from lib.mean_element_model import solve_mean_elements, update_TLE
from contextlib import closing
from lib.settings import database_config

logger = ProcessLogger("processor")

def compute_metrics(result: Result) -> dict:
    """
    Computes qualitative and quantitative residual metrics to evaluate fit quality.
    """
    logger.debug(300, "Computing residual metrics")
    residuals = result.residuals
    if residuals is None or len(residuals) == 0:
        return {
            "rmse": None, "mean": None, "acf_lag1": None, 
            "pacf_lag1": None, "threshold": None, "is_white_noise": None,
            "num_samples": 0
        }
        
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    
    res_mean = float(np.mean(residuals))
    res_rmse = float(np.sqrt(np.mean(residuals**2)))
    threshold = float(1.96 / np.sqrt(n)) if n > 0 else 0.0
    
    if n > 1:
        centered_res = residuals - res_mean
        variance = np.sum(centered_res**2)
        if variance > 0:
            covariance_lag1 = np.sum(centered_res[1:] * centered_res[:-1])
            acf_lag1 = float(covariance_lag1 / variance)
            pacf_lag1 = acf_lag1
        else:
            acf_lag1, pacf_lag1 = 0.0, 0.0
    else:
        acf_lag1, pacf_lag1 = 0.0, 0.0

    is_white_noise = bool(abs(acf_lag1) < threshold) if n > 1 else False

    return {
        "rmse": res_rmse,
        "mean": res_mean,
        "acf_lag1": acf_lag1,
        "pacf_lag1": pacf_lag1,
        "threshold": threshold,
        "is_white_noise": is_white_noise,
        "num_samples": n
    }

def write_result_to_db(
    spacecraft_id: str,
    result: Result, 
    pass_timestamp: datetime, 
    original_fc_hz: float
):
    """
    Writes time-offset optimization results to TimescaleDB using context managers.
    Failures propagate natively to the caller.
    """
    logger.info(200, f"Writing result to DB for contact {result.contact_id}")
    
    x_vals = getattr(result, "x", [0.0, 0.0, 0.0])
    cov_matrix = getattr(result, "cov", np.zeros((3, 3)))
    ssr_val = getattr(result, "ssr", 0.0)
    
    time_shift_s = float(x_vals[0])
    doppler_bias_hz = float(x_vals[1])
    fitted_fc_ghz = float(x_vals[2])
    fitted_fc_hz = fitted_fc_ghz * 1e9
    freq_error_hz = fitted_fc_hz - original_fc_hz
    
    var_time_shift = float(cov_matrix[0, 0]) if cov_matrix.shape == (3, 3) else 0.0
    var_doppler_bias = float(cov_matrix[1, 1]) if cov_matrix.shape == (3, 3) else 0.0
    var_freq_error = float(cov_matrix[2, 2]) if cov_matrix.shape == (3, 3) else 0.0
    
    is_3_param_model = var_freq_error > 0.0
    model_name_full = result.message.split(" | Model: ")[-1] if " | Model: " in result.message else "Unknown"
    
    metrics = compute_metrics(result)
    metadata = {
        "success": bool(result.success),
        "message": str(result.message),
        "passes_found": getattr(result, "passes_found", None),
        "full_covariance": cov_matrix.tolist() if hasattr(cov_matrix, "tolist") else []
    }

    with closing(psycopg2.connect(**database_config())) as conn:
        with conn:  # Automatically commits on success, rolls back on exception
            with conn.cursor() as cursor:
                
                logger.debug(301, "Upserting spacecraft registry")
                upsert_registry_query = """
                    INSERT INTO spacecraft_registry 
                        (spacecraft_id, name, center_frequency_hz, track, last_updated)
                    VALUES 
                        (%s, %s, %s, TRUE, NOW())
                    ON CONFLICT (spacecraft_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        center_frequency_hz = CASE 
                            WHEN %s = TRUE THEN EXCLUDED.center_frequency_hz 
                            ELSE spacecraft_registry.center_frequency_hz 
                        END,
                        frequency_confidence = CASE 
                            WHEN %s = TRUE THEN 'VERIFIED_3PARAM' 
                            ELSE spacecraft_registry.frequency_confidence 
                        END,
                        last_updated = NOW();
                """
                cursor.execute(
                    upsert_registry_query, 
                    (
                        spacecraft_id, result.name, fitted_fc_hz,
                        is_3_param_model and result.success, is_3_param_model and result.success
                    )
                )

                logger.debug(302, "Inserting BLS time offset result")
                insert_offset_query = """
                    INSERT INTO bls_time_offset (
                        timestamp, spacecraft_id, name, contact_id, model_type,
                        time_shift_s, doppler_bias_hz, fitted_frequency_hz, freq_error_hz,
                        var_time_shift, var_doppler_bias, var_freq_error,
                        ssr, num_samples,
                        residual_rmse, residual_mean, residual_acf_lag1, 
                        residual_pacf_lag1, residual_significance_threshold, is_white_noise,
                        metadata_json
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (timestamp, spacecraft_id, model_type) DO UPDATE SET
                        time_shift_s = EXCLUDED.time_shift_s,
                        doppler_bias_hz = EXCLUDED.doppler_bias_hz,
                        fitted_frequency_hz = EXCLUDED.fitted_frequency_hz,
                        freq_error_hz = EXCLUDED.freq_error_hz,
                        var_time_shift = EXCLUDED.var_time_shift,
                        var_doppler_bias = EXCLUDED.var_doppler_bias,
                        var_freq_error = EXCLUDED.var_freq_error,
                        ssr = EXCLUDED.ssr,
                        num_samples = EXCLUDED.num_samples,
                        residual_rmse = EXCLUDED.residual_rmse,
                        residual_mean = EXCLUDED.residual_mean,
                        residual_acf_lag1 = EXCLUDED.residual_acf_lag1,
                        residual_pacf_lag1 = EXCLUDED.residual_pacf_lag1,
                        residual_significance_threshold = EXCLUDED.residual_significance_threshold,
                        is_white_noise = EXCLUDED.is_white_noise,
                        metadata_json = EXCLUDED.metadata_json;
                """
                cursor.execute(
                    insert_offset_query,
                    (
                        pass_timestamp, spacecraft_id, result.name, result.contact_id, model_name_full[:20],
                        time_shift_s, doppler_bias_hz, fitted_fc_hz, freq_error_hz,
                        var_time_shift, var_doppler_bias, var_freq_error,
                        float(ssr_val), metrics["num_samples"],
                        metrics["rmse"], metrics["mean"], metrics["acf_lag1"],
                        metrics["pacf_lag1"], metrics["threshold"], metrics["is_white_noise"],
                        Json(metadata)
                    )
                )
    
    logger.info(200, f"Result successfully written to DB for contact {result.contact_id}")


def write_mean_elements_to_db(
    spacecraft_id: str,
    result: Result,
    pass_timestamp: datetime,
    original_line1: str,
    original_line2: str
):
    """
    Writes updated TLE lines and multi-pass mean elements results to TimescaleDB.
    Guarantees spacecraft registry exists prior to insert.
    """
    logger.info(200, f"Writing mean element result to DB for contacts {result.contact_id}")
    
    x_vals = getattr(result, "x", [])
    cov_matrix = getattr(result, "cov", np.zeros((2, 2)))
    ssr_val = getattr(result, "ssr", 0.0)
    
    mean_anomaly = float(x_vals[0]) if len(x_vals) > 0 else 0.0
    mean_motion = float(x_vals[1]) if len(x_vals) > 1 else 0.0
    doppler_bias_hz = float(x_vals[-1]) if len(x_vals) > 2 else 0.0
    
    var_mean_anomaly = float(cov_matrix[0, 0]) if cov_matrix.shape[0] > 0 else 0.0
    var_mean_motion = float(cov_matrix[1, 1]) if cov_matrix.shape[0] > 1 else 0.0
    var_doppler_bias = float(cov_matrix[-1, -1]) if cov_matrix.shape[0] > 2 else 0.0
    
    sat = Satrec.twoline2rv(original_line1, original_line2)
    base_kep = np.array([sat.inclo, sat.nodeo, sat.ecco, sat.argpo, sat.mo, sat.no_kozai])
    kep_mod = base_kep.copy()
    kep_mod[4] = mean_anomaly
    kep_mod[5] = mean_motion
    
    updated_line1, updated_line2 = update_TLE(kep_mod, original_line1, original_line2)
    
    metrics = compute_metrics(result)
    metadata = {
        "success": bool(result.success),
        "message": str(result.message),
        "passes_found": getattr(result, "passes_found", None),
        "full_covariance": cov_matrix.tolist() if hasattr(cov_matrix, "tolist") else []
    }

    with closing(psycopg2.connect(**database_config())) as conn:
        with conn:
            with conn.cursor() as cursor:
                
                logger.debug(301, "Upserting spacecraft registry before mean elements insert")
                placeholder_hz = 0.0 
                upsert_registry_query = """
                    INSERT INTO spacecraft_registry 
                        (spacecraft_id, name, center_frequency_hz, track, last_updated)
                    VALUES 
                        (%s, %s, %s, TRUE, NOW())
                    ON CONFLICT (spacecraft_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        last_updated = NOW();
                """
                cursor.execute(
                    upsert_registry_query, 
                    (spacecraft_id, result.name, placeholder_hz)
                )
                
                insert_query = """
                    INSERT INTO bls_tle_2param (
                        timestamp, spacecraft_id, name, contact_id, model_type,
                        mean_anomaly, mean_motion, doppler_bias_hz,
                        var_mean_anomaly, var_mean_motion, var_doppler_bias,
                        ssr, num_samples,
                        residual_rmse, residual_mean, residual_acf_lag1,
                        residual_pacf_lag1, residual_significance_threshold, is_white_noise,
                        updated_tle_line1, updated_tle_line2,
                        metadata_json
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (timestamp, spacecraft_id, model_type) DO UPDATE SET
                        mean_anomaly = EXCLUDED.mean_anomaly,
                        mean_motion = EXCLUDED.mean_motion,
                        doppler_bias_hz = EXCLUDED.doppler_bias_hz,
                        var_mean_anomaly = EXCLUDED.var_mean_anomaly,
                        var_mean_motion = EXCLUDED.var_mean_motion,
                        var_doppler_bias = EXCLUDED.var_doppler_bias,
                        ssr = EXCLUDED.ssr,
                        num_samples = EXCLUDED.num_samples,
                        residual_rmse = EXCLUDED.residual_rmse,
                        residual_mean = EXCLUDED.residual_mean,
                        residual_acf_lag1 = EXCLUDED.residual_acf_lag1,
                        residual_pacf_lag1 = EXCLUDED.residual_pacf_lag1,
                        residual_significance_threshold = EXCLUDED.residual_significance_threshold,
                        is_white_noise = EXCLUDED.is_white_noise,
                        updated_tle_line1 = EXCLUDED.updated_tle_line1,
                        updated_tle_line2 = EXCLUDED.updated_tle_line2,
                        metadata_json = EXCLUDED.metadata_json;
                """
                cursor.execute(
                    insert_query,
                    (
                        pass_timestamp, spacecraft_id, result.name, str(result.contact_id), "MEAN_ELEMENT_2PARAM",
                        mean_anomaly, mean_motion, doppler_bias_hz,
                        var_mean_anomaly, var_mean_motion, var_doppler_bias,
                        float(ssr_val), metrics["num_samples"],
                        metrics["rmse"], metrics["mean"], metrics["acf_lag1"],
                        metrics["pacf_lag1"], metrics["threshold"], metrics["is_white_noise"],
                        updated_line1, updated_line2,
                        Json(metadata)
                    )
                )

    logger.info(200, f"Mean element results successfully written to DB for contacts {result.contact_id}")

def context_to_config(ctx: TrackingContext) -> Config:
    """Translate request options into solver configuration."""

    base_config = Config()
    if ctx.criterion is not None:
        base_config.criterion = ctx.criterion
    if ctx.f_scale is not None:
        base_config.f_scale = ctx.f_scale
    if ctx.loss_function is not None:
        base_config.loss = ctx.loss_function
    if ctx.method is not None:
        base_config.method = ctx.method
    if ctx.model_type is not None:
        base_config.model_type = ctx.model_type
    if ctx.angle_constraint is not None:
        base_config.N_degrees = ctx.angle_constraint
    if ctx.penalty_weight is not None:
        base_config.penalty_weight = ctx.penalty_weight
    if ctx.qmc_samples is not None:
        base_config.qmc_samples = ctx.qmc_samples
    if ctx.use_qmc is not None:
        base_config.use_qmc = ctx.use_qmc

    return base_config

def create_od_data(pass_df: pl.DataFrame) -> tuple[Data, float]:
    """Maps a single pass DataFrame into the solver's Data dataclass."""
    if pass_df.is_empty():
        raise ValueError("Cannot create solver data from an empty DataFrame")

    contact_id = pass_df["contact_id"][0]
    sc_id = pass_df["spacecraft_id"][0]
    pass_logger = logger.bind(contact_id=contact_id, spacecraft_id=sc_id)
    
    pass_logger.info(201, "Creating OD data object from pass DataFrame")
    
    spacecraft_name = pass_df["spacecraft_name"][0]
    tle_line1 = pass_df['tle_line_1'][0]
    tle_line2 = pass_df['tle_line_2'][0]
    sat_tle = sk.TLE.from_lines([spacecraft_name, tle_line1, tle_line2])
    
    lat = pass_df['station_lat'][0]
    lon = pass_df['station_lon'][0]
    alt = pass_df['station_alt'][0]  
    
    fc_hz = pass_df['satellite_frequency'][0]
    fc_GHz = fc_hz / 1e9
    
    if pass_df['timestamp'].dtype == pl.Utf8:
        pass_df = pass_df.with_columns(pl.col('timestamp').str.to_datetime())
        
    py_datetimes = pass_df['timestamp'].to_list()
    time_array = np.array([sk.time.from_datetime(dt) for dt in py_datetimes])
    
    obs_doppler = pass_df['lr1_receiver1_actualCarrierFrequencyOffset'].to_numpy()
    
    pass_logger.debug(303, f"Loaded TLE for {spacecraft_name}, fc={fc_GHz} GHz, n_samples={len(time_array)}")
    
    # ---------------------------------------------------------
    # POINTING VECTOR TRANSFORMATION: Topocentric -> ITRF -> GCRF
    # ---------------------------------------------------------
    
    # 1. Extract Azimuth and Elevation
    az_deg = pass_df['antenna1_position_azimuth'].to_numpy()
    el_deg = pass_df['antenna1_position_elevation'].to_numpy()
    
    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)
    
    # 2. Calculate Topocentric ENU unit vectors
    E = np.sin(az_rad) * np.cos(el_rad)
    N = np.cos(az_rad) * np.cos(el_rad)
    U = np.sin(el_rad)
    
    enu_vectors = np.column_stack((E, N, U))
    
    # 3. Transform ENU to ITRF (Earth-Centered, Earth-Fixed)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    
    R_enu_to_itrf = np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [ cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
        [     0.0,            cos_lat,            sin_lat]
    ])
    
    itrf_vectors = (R_enu_to_itrf @ enu_vectors.T).T 
    
    # 4. Transform ITRF to GCRF using satkit's Earth rotation models
    obs_pointing = np.zeros((len(time_array), 3))
    
    for i, t in enumerate(time_array):
        q_itrf2gcrf = sk.frametransform.qitrf2gcrf(t)
        obs_pointing[i] = q_itrf2gcrf * itrf_vectors[i]
        
    # ---------------------------------------------------------
        
    contact_ids = pass_df["contact_id"].cast(pl.Utf8).to_numpy()
    pass_tles = {}
    pass_stations = {}
    for contact in pass_df["contact_id"].unique().to_list():
        contact_rows = pass_df.filter(pl.col("contact_id") == contact)
        contact_key = str(contact)
        pass_tles[contact_key] = sk.TLE.from_lines([
            contact_rows["spacecraft_name"][0],
            contact_rows["tle_line_1"][0],
            contact_rows["tle_line_2"][0],
        ])
        pass_stations[contact_key] = (
            contact_rows["station_lat"][0],
            contact_rows["station_lon"][0],
            contact_rows["station_alt"][0],
        )

    od_data = Data(
        time_array=time_array,
        tle=sat_tle,
        spacecraft_name=spacecraft_name,
        contact_ids=contact_ids,
        pass_tles=pass_tles,
        pass_stations=pass_stations,
        lat=lat,
        lon=lon,
        alt=alt,
        obs_doppler=obs_doppler,
        obs_pointing=obs_pointing
    )
    
    pass_logger.info(202, "OD data object created successfully")
    return od_data, fc_GHz

def process_telemetry_batch(df: pl.DataFrame, context: TrackingContext) -> dict[str, Any]:
    """
    Processes time-offset per pass independently and applies a multi-pass 
    mean-element solver over grouped observations.
    """
    logger.info(200, f"Starting telemetry batch processing: {len(df)} total rows")
    results: dict[str, list[Any]] = {
        "time_offset_passes": [],
        "mean_elements_windows": [],
        "errors": [],
    }
    if df.is_empty():
        return results

    config = context_to_config(context)
    
    # ---------------------------------------------------------
    # STAGE 1: Per-pass time offset calculation
    # ---------------------------------------------------------
    pass_groups = df.partition_by("contact_id")
    logger.info(200, f"Partitioned into {len(pass_groups)} contact passes for time-offset solving")
    
    for pass_df in pass_groups:
        contact_id = pass_df['contact_id'][0]
        sc_id = pass_df['spacecraft_id'][0]

        pass_logger = logger.bind(contact_id=contact_id, spacecraft_id=sc_id)
        pass_start_time = pass_df['timestamp'][0]
        original_frequency_hz = pass_df['satellite_frequency'][0]
                    
        pass_logger.info(203, f"Processing pass with {len(pass_df)} samples")
        try:
            data_obj, fc_GHz = create_od_data(pass_df)
            pass_result = solve_time_shift(data=data_obj, config=config, fc_GHz=fc_GHz)
            if not pass_result.success:
                pass_logger.warning(422, f"Time shift solver failed: {pass_result.message}")
            else:
                pass_logger.info(205, f"Time shift solver converged: {pass_result.message}")
            
            write_result_to_db(sc_id, pass_result, pass_start_time, original_frequency_hz)
            results["time_offset_passes"].append(contact_id)
            
        except Exception as exc:
            pass_logger.error(500, f"Critical execution failure handling time shift for {contact_id}: {exc}")
            results["errors"].append(
                {"stage": "time_offset", "contact_id": contact_id, "error": str(exc)}
            )

    # ---------------------------------------------------------
    # STAGE 2: Multi-pass Mean-Element calculation
    # ---------------------------------------------------------
    sc_groups = df.partition_by("spacecraft_id")
    for sc_df in sc_groups:
        sc_id = sc_df['spacecraft_id'][0]
        sc_logger = logger.bind(spacecraft_id=sc_id)

        # Retrieve maximum timestamp per contact_id to sort the latest passes
        pass_summaries = sc_df.group_by("contact_id").agg(pl.col("timestamp").max().alias("max_ts"))
        pass_summaries = pass_summaries.sort("max_ts", descending=True)
        top_contacts = pass_summaries["contact_id"].head(3).to_list()

        sc_logger.info(208, f"Selected {len(top_contacts)} most recent passes for mean-element solving.")

        # Re-filter dataset for the target contacts and mandate chronological ordering
        multipass_df = sc_df.filter(pl.col("contact_id").is_in(top_contacts)).sort("timestamp")
        
        if len(multipass_df) == 0:
            continue

        base_line1 = multipass_df['tle_line_1'][0]
        base_line2 = multipass_df['tle_line_2'][0]
        latest_timestamp = multipass_df['timestamp'].max()

        try:
            multipass_data_obj, fc_GHz = create_od_data(multipass_df)
            me_result = solve_mean_elements(
                data=multipass_data_obj, 
                config=config, 
                line1=base_line1, 
                line2=base_line2, 
                fc_GHz=fc_GHz
            )

            if me_result.success:
                sc_logger.info(209, f"Mean elements solver converged: {me_result.message}")
                write_mean_elements_to_db(
                    spacecraft_id=sc_id,
                    result=me_result,
                    pass_timestamp=latest_timestamp,
                    original_line1=base_line1,
                    original_line2=base_line2
                )
                results["mean_elements_windows"].append(me_result.contact_id)
            else:
                sc_logger.warning(423, f"Mean elements solver failed: {me_result.message}")

        except Exception as exc:
            sc_logger.error(500, f"Critical execution failure handling mean elements for {sc_id}: {exc}")
            results["errors"].append(
                {"stage": "mean_elements", "spacecraft_id": sc_id, "error": str(exc)}
            )

    logger.info(200, "Telemetry batch processing complete")
    return results
