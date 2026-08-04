import os
import glob
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import satkit as sk
import time
import optuna

from models.common import Data, Config
from models.time_model import solve_time_shift, simulate_doppler_curve

# =============================================================================
# Configuration
# =============================================================================
ROOT_DIR = "/home/tycho/leop/doppler_parquet/"

GPS_TLES = {
    "18": (
        "1 90918U 00000AAA 26124.10084548  .00000000  00000-0  44216-3 0  9992",
        "2 90918  97.7369  22.3788 0001905  77.7453  86.9927 14.91984340    02"
    ),
    "19": (
        "1 90919U 00000AAA 26124.11166446  .00000000  00000-0  83669-3 0  9997",
        "2 90919  97.7414  22.3835 0001997 127.8106  95.3615 14.92130954    03"
    )
}

# Known Hardware Delay
HW_OFFSET_SECONDS = 0.35

# =============================================================================
# Data Pipeline
# =============================================================================

def load_and_clean_data(root_dir: str, sat_id: str) -> pl.DataFrame:
    """Loads and filters parquet files using Polars lazy execution."""
    search_pattern = os.path.join(root_dir, f"forest{sat_id}*.parquet")
    
    if not glob.glob(search_pattern):
        raise FileNotFoundError(f"No parquet files found matching {search_pattern}")
        
    doppler_col = 'lr1_receiver1_actualCarrierFrequencyOffset'
    
    lf = pl.scan_parquet(search_pattern)
    
    cleaned_df = lf.filter(
        (pl.col('antenna1_position_elevation') > 1.0) &
        (pl.col('antenna1_position_elevation') < 89.0) &
        (pl.col(doppler_col).is_not_null()) &
        (pl.col(doppler_col) != 0.0) &
        (pl.col(doppler_col).abs() >= 0.1)
    ).collect()
    
    return cleaned_df


def create_od_data(pass_df: pl.DataFrame) -> tuple[Data, float]:
    """Maps a single pass DataFrame into the solver's Data dataclass."""
    tle_line1 = pass_df['tle_line1'][0]
    tle_line2 = pass_df['tle_line2'][0]
    sat_tle = sk.TLE.from_lines([tle_line1, tle_line2])
    
    lat = pass_df['station_lat'][0]
    lon = pass_df['station_lon'][0]
    alt = pass_df['station_alt'][0]  
    
    fc_hz = pass_df['expected_frequency'][0]
    fc_GHz = fc_hz / 1e9
    
    if pass_df['timestamp'].dtype == pl.Utf8:
        pass_df = pass_df.with_columns(pl.col('timestamp').str.to_datetime())
        
    py_datetimes = pass_df['timestamp'].to_list()
    time_array = np.array([sk.time.from_datetime(dt) for dt in py_datetimes])
    
    obs_doppler = pass_df['lr1_receiver1_actualCarrierFrequencyOffset'].to_numpy()
    
    # ---------------------------------------------------------
    # POINTING VECTOR TRANSFORMATION: Topocentric -> ITRF -> GCRF
    # ---------------------------------------------------------
    
    # 1. Extract Azimuth and Elevation
    az_deg = pass_df['antenna1_position_azimuth'].to_numpy()
    el_deg = pass_df['antenna1_position_elevation'].to_numpy()
    
    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)
    
    # 2. Calculate Topocentric ENU unit vectors
    # Azimuth is measured clockwise from North (0 rad) to East (pi/2 rad)
    E = np.sin(az_rad) * np.cos(el_rad)
    N = np.cos(az_rad) * np.cos(el_rad)
    U = np.sin(el_rad)
    
    enu_vectors = np.column_stack((E, N, U))
    
    # 3. Transform ENU to ITRF (Earth-Centered, Earth-Fixed)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    
    # Rotation matrix from local tangent plane to Earth-fixed frame
    R_enu_to_itrf = np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [ cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
        [     0.0,            cos_lat,            sin_lat]
    ])
    
    # Apply rotation (dot product broadcasts over the N vectors)
    itrf_vectors = (R_enu_to_itrf @ enu_vectors.T).T 
    
    # 4. Transform ITRF to GCRF using satkit's Earth rotation models
    obs_pointing = np.zeros((len(time_array), 3))
    
    for i, t in enumerate(time_array):
        # Fetch the Earth rotation quaternion for the specific timestamp
        q_itrf2gcrf = sk.frametransform.qitrf2gcrf(t)
        
        # Apply quaternion rotation to the ITRF pointing vector
        obs_pointing[i] = q_itrf2gcrf * itrf_vectors[i]
        
    # ---------------------------------------------------------
        
    od_data = Data(
        time_array=time_array,
        tle=sat_tle,
        lat=lat,
        lon=lon,
        alt=alt,
        obs_doppler=obs_doppler,
        obs_pointing=obs_pointing
    )
    
    return od_data, fc_GHz

# =============================================================================
# Execution
# =============================================================================

def objective(trial, full_df, sat_id, truth_tle):
    """Optuna objective function to minimize Cartesian error on a holdout set."""
    
    # 1. Suggest Hyperparameters
    loss_type = trial.suggest_categorical("loss", ["linear", "soft_l1", "huber", "cauchy", "arctan"])
    solver_method = trial.suggest_categorical("method", ["trf", "dogbox"])
    
    if loss_type == "linear":
        f_scale_val = 1.0
    else:
        f_scale_val = trial.suggest_float("f_scale", 50.0, 5000.0, log=True)

    reg_t = trial.suggest_float("reg_t", 1e-7, 1e-2, log=True)
    reg_b = trial.suggest_float("reg_b", 1e-6, 1e-1, log=True)
    reg_f = trial.suggest_float("reg_f", 1e-7, 1e-2, log=True)
    
    penalty_weight = trial.suggest_float("penalty_weight", 1e-4, 1e4, log=True)
    N_degrees = trial.suggest_float("N_degrees", 1.0, 10.0)
    
    # 2. Isolate the first 5 passes
    unique_contacts = full_df['contact_id'].unique().to_list()
    target_contacts = unique_contacts[:5]
    
    pass_errors = []
    
    # 3. Evaluate solver on each pass using Train/Test split
    for contact in target_contacts:
        pass_df = full_df.filter(pl.col('contact_id') == contact)
        
        if len(pass_df) < 20:
            continue
            
        # Interleaved 50/50 Split (Even/Odd index)
        pass_df = pass_df.with_row_index("index")
        train_df = pass_df.filter(pl.col("index") % 2 == 0)
        test_df = pass_df.filter(pl.col("index") % 2 != 0)
        
        train_data, fc_GHz = create_od_data(train_df)
        test_data, _ = create_od_data(test_df)
        
        config = Config(
            x0=np.array([0.0, 0.0, fc_GHz]),
            qmc_bounds=((-120.0, -1e5, fc_GHz - 0.5), (0.0, 1e5, fc_GHz + 0.5)),
            reg_weights=np.array([reg_t, reg_b, reg_f]),
            penalty_weight=penalty_weight,
            N_degrees=N_degrees,  
            method=solver_method,
            loss=loss_type,
            f_scale=f_scale_val,
            model_type="2-param",
            criterion="BIC",
            use_qmc=True,
            qmc_samples=64  
        )
        
        try:
            # Fit solver strictly on the 50% training set
            result = solve_time_shift(train_data, config, fc_GHz)
            
            # Evaluate Cartesian Error against Truth TLE using the 50% testing set
            dt_seconds = float(result.x[0])
            
            # Incorporate +0.35s hardware offset into the final evaluation
            t_shifted = test_data.time_array + sk.duration(seconds=dt_seconds + HW_OFFSET_SECONDS)
            
            p_true, _ = sk.sgp4(truth_tle, test_data.time_array)
            p_corr, _ = sk.sgp4(test_data.tle, t_shifted)
            
            mean_err_km = np.mean(np.linalg.norm(p_true - p_corr, axis=1)) / 1000.0
            pass_errors.append(mean_err_km)
            
        except Exception as e:
            return 9999.0 
            
    if not pass_errors:
        return 9999.0
        
    return float(np.mean(pass_errors))


if __name__ == "__main__":
    sat_id = "18"
    
    truth_line1, truth_line2 = GPS_TLES[sat_id]
    truth_tle = sk.TLE.from_lines([truth_line1, truth_line2])
    
    full_df = load_and_clean_data(ROOT_DIR, sat_id)
    
    study = optuna.create_study(direction="minimize")
    
    print(f"Starting Hyperparameter Optimization for {sat_id}...")
    study.optimize(lambda trial: objective(trial, full_df, sat_id, truth_tle), n_trials=1000)
    
    print("\nOptimization Complete.")
    print("Best Trial:")
    print(f"  Cartesian Error (Test Set): {study.best_value:.2f} km")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")