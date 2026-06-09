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
    "16": (
        "1 90916U 00000AAA 26124.11399935  .00000000  00000-0  43981-3 0  9998",
        "2 90916  97.7435  22.3832 0001203 162.4543  72.6511 14.91902713    02"
    ),
    "17": (
        "1 90917U 00000AAA 26124.11249883  .00000000  00000-0  44072-3 0  9997",
        "2 90917  97.7385  22.3866 0005222 306.9095 276.5605 14.90519438    02"
    ),
    "18": (
        "1 90918U 00000AAA 26124.10084548  .00000000  00000-0  44216-3 0  9992",
        "2 90918  97.7369  22.3788 0001905  77.7453  86.9927 14.91984340    02"
    ),
    "19": (
        "1 90919U 00000AAA 26124.11166446  .00000000  00000-0  83669-3 0  9997",
        "2 90919  97.7414  22.3835 0001997 127.8106  95.3615 14.92130954    03"
    )
}
# =============================================================================
# Data Pipeline
# =============================================================================

def load_and_clean_data(root_dir: str, sat_id: str) -> pl.DataFrame:
    """Loads and filters parquet files using Polars lazy execution."""
    search_pattern = os.path.join(root_dir, f"forest{sat_id}*.parquet")
    
    # Verify files exist before scanning
    if not glob.glob(search_pattern):
        raise FileNotFoundError(f"No parquet files found matching {search_pattern}")
        
    doppler_col = 'lr1_receiver1_actualCarrierFrequencyOffset'
    
    # Lazy load and filter
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
    # 1. Extract metadata (assuming constant across the pass)
    tle_line1 = pass_df['tle_line1'][0]
    tle_line2 = pass_df['tle_line2'][0]
    sat_tle = sk.TLE.from_lines([tle_line1, tle_line2])
    
    lat = pass_df['station_lat'][0]
    lon = pass_df['station_lon'][0]
    alt = pass_df['station_alt'][0]  # Assumes meters
    
    fc_hz = pass_df['expected_frequency'][0]
    fc_GHz = fc_hz / 1e9
    
    # 2. Format Timestamps
    # Ensure timestamp column is datetime type, then convert to standard python datetimes
    if pass_df['timestamp'].dtype == pl.Utf8:
        pass_df = pass_df.with_columns(pl.col('timestamp').str.to_datetime())
        
    py_datetimes = pass_df['timestamp'].to_list()
    
    print(py_datetimes[0])
    
    # Convert standard datetimes to satkit time objects
    time_array = np.array([sk.time.from_datetime(dt) for dt in py_datetimes])
    
    # 3. Extract Observations
    obs_doppler = pass_df['lr1_receiver1_actualCarrierFrequencyOffset'].to_numpy()
    
    # Mute pointing requirements: pass zero vectors
    obs_pointing = np.zeros((len(time_array), 3))
    
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


if __name__ == "__main__":
    for sat_id in GPS_TLES.keys():
        print(f"\n{'='*50}\nProcessing Satellite {sat_id}\n{'='*50}")
        
        try:
            full_df = load_and_clean_data(ROOT_DIR, sat_id)
        except FileNotFoundError:
            print(f"Skipping {sat_id} - no data found.")
            continue
            
        if full_df.is_empty():
            print(f"Skipping {sat_id} - no valid data post-filtering.")
            continue
            
        # Isolate the first continuous pass
        target_contact = full_df['contact_id'].unique()[0]
        pass_df = full_df.filter(pl.col('contact_id') == target_contact)
        
        print(f"Isolated Contact {target_contact} | {len(pass_df)} samples")
        
        # Build OD Data Structures
        obs_data, fc_GHz = create_od_data(pass_df)
        
        # Configure the solver
        config = Config(
            x0=np.array([0.0, 0.0, fc_GHz]),
            qmc_bounds=(
                (-60.0, -1e5, fc_GHz - 0.5),  # Lower bounds
                (0.0, 1e5, fc_GHz + 0.5)       # Upper bounds
            ),
            reg_weights=np.array([7e-5, 1e-5, 2e-7]),
            penalty_weight=5e1,
            f_scale=250.0,
            N_degrees=10.0,  
            method="trf",
            loss="huber",
            model_type="2-param",
            criterion="AIC",
            use_qmc=True,
            qmc_samples=1024
        )
        
        print("Starting time-shift least-squares optimization...")
        t0 = time.time()
        result = solve_time_shift(obs_data, config, fc_GHz)
        t1 = time.time()
        
        print(f"Optimization Time: {t1 - t0:.4f} seconds")
        print(f"Fitted State:      {result.x}")
        print(f"SSR:               {result.ssr:.2f}")
        print(f"Model Selected:    {result.message}")
        
        # Simulate the final fitted curve for plotting
        fitted_doppler, _ = simulate_doppler_curve(result.x, obs_data, fc_GHz)
        
        # =====================================================================
        # Error Comparison: Truth vs Prior (Uncorrected) vs Prior (Corrected)
        # =====================================================================
        truth_line1, truth_line2 = GPS_TLES[sat_id]
        truth_tle = sk.TLE.from_lines([truth_line1, truth_line2])
        
        t_array = obs_data.time_array
        dt_seconds = float(result.x[0])
        t_shifted = t_array + sk.duration(seconds=dt_seconds) + sk.duration(seconds=0.350)
        
        # Propagate states (Satkit returns arrays of shape [N, 3] in meters)
        p_true, _ = sk.sgp4(truth_tle, t_array)
        p_uncorr, _ = sk.sgp4(obs_data.tle, t_array)
        p_corr, _ = sk.sgp4(obs_data.tle, t_shifted)
        
        # Calculate 3D Euclidean distances in km
        err_uncorr_km = np.linalg.norm(p_true - p_uncorr, axis=1) / 1000.0
        err_corr_km = np.linalg.norm(p_true - p_corr, axis=1) / 1000.0

        # =====================================================================
        # Plot Results (3 Panels)
        # =====================================================================
        times_py = [t.as_datetime() for t in obs_data.time_array]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Panel 1: Doppler Shift
        ax1.scatter(times_py, obs_data.obs_doppler, color='blue', label='Telemetry Doppler', s=10, alpha=0.5)
        ax1.plot(times_py, fitted_doppler, color='red', linestyle='--', label='Fitted Model', linewidth=2)
        ax1.set_ylabel('Doppler (Hz)')
        ax1.set_title(f'[{sat_id}] LEOP Time-Shift Fit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Doppler Residuals
        residuals = obs_data.obs_doppler - fitted_doppler
        ax2.scatter(times_py, residuals, color='purple', s=10, alpha=0.5, label='Residuals')
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_ylabel('Residual Error (Hz)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: Cartesian Prediction Error
        ax3.plot(times_py, err_uncorr_km, color='orange', label=f'Prior Error (Uncorrected, $\Delta t=0$)', linewidth=2)
        ax3.plot(times_py, err_corr_km, color='green', label=f'Prior Error (Corrected, $\Delta t={dt_seconds:.2f}s$)', linewidth=2)
        ax3.set_ylabel('Cartesian Error (km)')
        ax3.set_xlabel('Time (UTC)')
        ax3.set_title('Prediction Error vs GPS Reference TLE')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()