import os
import glob
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Astropy and SGP4 Imports for Pointing Verification
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, TEME, CartesianRepresentation
from astropy import units as u
from sgp4.api import Satrec

# Import local modules
from model import RangeRateModel
from batch_estimate import FlexibleDopplerModel
from forced_simulator import ForcedSimulator

# =============================================================================
# Configuration
# =============================================================================
ROOT_DIR = "/home/tycho/leop/doppler_parquet/"

# Grid definitions
DT_GRID = np.arange(-500, 25, 10)          # -80 to +25 seconds, step 1
DF_GRID = np.arange(-50001, 50000, 5000) # -50 kHz to +50 kHz, step 5 kHz

REG_LAMBDA = 1000.0
DT_NORM = 120.0
DF_NORM = 10000.0

# Soft Constraint Penalty Weights
WEIGHT_HORIZON = 15000.0
WEIGHT_POINTING = 2000.0

# Reference GPS TLEs (Truth)
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
# Core Execution
# =============================================================================

def load_and_clean_data(root_dir: str, sat_id: str) -> pd.DataFrame:
    """Loads parquet files for a specific satellite and applies LEOP filtering rules."""
    search_pattern = os.path.join(root_dir, f"forest{sat_id}*.parquet")
    files = glob.glob(search_pattern)
    
    if not files:
        raise FileNotFoundError(f"No parquet files found matching {search_pattern}")
        
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).dropna()
    
    # Elevation mask (low pass capture)
    df = df[(df['antenna1_position_elevation'] > 1.0) & (df['antenna1_position_elevation'] < 89.0)]
    
    # Doppler valid mask
    doppler_col = 'lr1_receiver1_actualCarrierFrequencyOffset'
    if doppler_col in df.columns:
        df = df[(df[doppler_col] != 0) & (df[doppler_col].abs() >= 0.1)].copy()
    
    return df


def compute_penalties(track_line1: str, track_line2: str, dt_val: float, times_unix: np.ndarray, 
                      lat: float, lon: float, alt_km: float, 
                      telemetry_el: np.ndarray, telemetry_az: np.ndarray) -> tuple:
    """
    Computes soft constraint penalties for horizon violations and angle separations.
    """
    sat = Satrec.twoline2rv(track_line1, track_line2)
    t_shifted = times_unix + dt_val
    t_astro = Time(t_shifted, format='unix')
    
    err, r, _ = sat.sgp4_array(t_astro.jd1, t_astro.jd2)
    valid = (err == 0)
    
    if not np.any(valid):
        return 1e6, 1e6  # High penalty for failed orbit propagation
        
    teme_pos = CartesianRepresentation(x=r[valid, 0]*u.km, y=r[valid, 1]*u.km, z=r[valid, 2]*u.km)
    teme = TEME(teme_pos, obstime=t_astro[valid])
    
    locs = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=alt_km*1000*u.m)
    altaz_frame = AltAz(obstime=t_astro[valid], location=locs)
    
    true_altaz = teme.transform_to(altaz_frame)
    theoretical_el = true_altaz.alt.deg
    theoretical_az = true_altaz.az.deg
    
    # Angular processing
    el1 = np.radians(theoretical_el)
    el2 = np.radians(telemetry_el[valid])
    az1 = np.radians(theoretical_az)
    az2 = np.radians(telemetry_az[valid])
    
    d_az = az1 - az2
    cos_ang = np.sin(el1)*np.sin(el2) + np.cos(el1)*np.cos(el2)*np.cos(d_az)
    ang_dist = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
    
    # 1. Horizon Penalty: Satellite must be above 0 degrees
    horizon_violations = np.maximum(0, -theoretical_el)
    print(f"theoretical_el: {theoretical_el}, horizon_violations: {horizon_violations}")
    penalty_horizon = np.sum(horizon_violations**2) * WEIGHT_HORIZON
    
    # 2. Pointing Penalty: Telemetry tracker vs Model Separation > 5 degrees
    pointing_violations = np.maximum(0, ang_dist - 2.0)
    penalty_pointing = np.sum(pointing_violations**2) * WEIGHT_POINTING
    
    return penalty_horizon, penalty_pointing


def generate_grid_metrics(df_pass: pd.DataFrame, ref_line1: str, ref_line2: str):
    """Evaluates the 1D Error and 2D Grids using a MAP estimator with known variance."""
    
    track_line1 = df_pass['tle_line1'].iloc[0]
    track_line2 = df_pass['tle_line2'].iloc[0]
    lat = df_pass['station_lat'].iloc[0]
    lon = df_pass['station_lon'].iloc[0]
    alt_km = df_pass['station_alt'].iloc[0] / 1000.0
    f_carrier = df_pass['expected_frequency'].iloc[0]
    
    telemetry_el = df_pass['antenna1_position_elevation'].values
    telemetry_az = df_pass['antenna1_position_azimuth'].values
    
    times_pd = pd.to_datetime(df_pass["timestamp"], utc=True)
    times_unix = times_pd.astype('int64').values / 10**9
    doppler_meas = df_pass['lr1_receiver1_actualCarrierFrequencyOffset'].values
    n_samples = len(doppler_meas)
    
    simulator = ForcedSimulator(ref_line1, ref_line2)
    base_model = RangeRateModel("Track", track_line1, track_line2)
    eval_model = FlexibleDopplerModel(base_model)
    
    t_tensor = torch.tensor(times_unix, dtype=torch.float64)
    z_tensor = torch.tensor(doppler_meas, dtype=torch.float64)
    fc_tensor = torch.tensor(f_carrier, dtype=torch.float64)
    
    err_1d = np.zeros(len(DT_GRID))
    shape_2d = (len(DF_GRID), len(DT_GRID))
    aic_2d = np.zeros(shape_2d)
    bic_2d = np.zeros(shape_2d)
    
    SIGMA_MEAS = 30000.0  
    SIGMA_T = 1.0        
    
    print(f"Evaluating grids for pass containing {n_samples} samples...")
    
    with torch.no_grad():
        for j, dt_val in enumerate(DT_GRID):
            dt_tensor = torch.tensor(dt_val, dtype=torch.float64)
            
            err_df = simulator.calculate_propagation_error(
                track_line1, track_line2, dt_val, times_unix
            )
            err_1d[j] = err_df['total_position_error_km'].mean()
            
            # Time regularizer and physical constraints
            penalty_t = 1e3 * (dt_val / SIGMA_T)**2
            penalty_horiz, penalty_point = compute_penalties(
                track_line1, track_line2, dt_val, times_unix, 
                lat, lon, alt_km, telemetry_el, telemetry_az
            )
            total_dt_penalty = penalty_t + penalty_horiz + penalty_point
            
            for i, df_val in enumerate(DF_GRID):
                df_tensor = torch.tensor(df_val, dtype=torch.float64)
                
                f_pred = eval_model.evaluate(dt_tensor, fc_tensor, df_tensor, t_tensor, lat, lon, alt_km)
                rss = torch.sum((z_tensor - f_pred)**2).item()
                
                chi2_data = rss / (SIGMA_MEAS**2)
                cost = chi2_data + total_dt_penalty
                
                print(f"DT={dt_val:6.1f}s, DF={df_val:6.0f}Hz -> RSS={rss:.2e}, Cost={cost:.2e}, Penalties(T={penalty_t:.2e}, H={penalty_horiz:.2e}, P={penalty_point:.2e})")
                
                k_2d = 2
                aic_2d[i, j] = cost + 2*k_2d
                bic_2d[i, j] = cost + k_2d*np.log(n_samples)
    
    print("Grid evaluation complete.")
    
    return err_1d, aic_2d, bic_2d


def plot_separated_results(err_1d, aic_2d, bic_2d, sat_id):
    """Visualizes the 1D slices from optimal bias rows and 2D heatmaps with markers."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 14))
    
    aic_min_idx = np.unravel_index(np.argmin(aic_2d), aic_2d.shape)
    bic_min_idx = np.unravel_index(np.argmin(bic_2d), bic_2d.shape)
    
    true_dt_idx = np.argmin(err_1d)
    true_df_idx = np.argmin(aic_2d[:, true_dt_idx]) 
    
    aic_1d_slice = aic_2d[aic_min_idx[0], :]
    bic_1d_slice = bic_2d[bic_min_idx[0], :]
    
    df_opt_aic = DF_GRID[aic_min_idx[0]]
    df_opt_bic = DF_GRID[bic_min_idx[0]]

    # 1D Model AIC / BIC vs Time Offset (at Optimal Bias)
    ax1 = axes[0, 0]
    ax1.plot(DT_GRID, err_1d, color='darkred', marker='o', linewidth=2)
    ax1.axvline(DT_GRID[true_dt_idx], color='green', linestyle='--', label='Physical Minimum')
    ax1.set_title(f'[{sat_id}] Physical Model: Mean Position Error', fontsize=14)
    ax1.set_xlabel('Time Offset $\Delta t$ (s)', fontsize=12)
    ax1.set_ylabel('Total Position Error (km)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2 = axes[0, 1]
    ax2.plot(DT_GRID, aic_1d_slice, label=f'AIC ($\Delta f = {df_opt_aic}$ Hz)', color='blue', marker='s', linewidth=2)
    ax2.plot(DT_GRID, bic_1d_slice, label=f'BIC ($\Delta f = {df_opt_bic}$ Hz)', color='orange', marker='^', linewidth=2)
    ax2.axvline(DT_GRID[true_dt_idx], color='green', linestyle='--', label='Physical Minimum (True $\Delta t$)')
    ax2.set_title(f'[{sat_id}] Information Criteria at Optimal Frequency Bias', fontsize=14)
    ax2.set_xlabel('Time Offset $\Delta t$ (s)', fontsize=12)
    ax2.set_ylabel('Score (Lower is Better)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)

    cmap_opt = "viridis_r"
    kwargs = {'xticklabels': DT_GRID, 'yticklabels': DF_GRID, 'cbar_kws': {'shrink': 0.8}, 'annot': False}

    ax3 = axes[1, 0]
    sns.heatmap(aic_2d, ax=ax3, cmap=cmap_opt, **kwargs)
    ax3.set_title(f'[{sat_id}] 2D Model: AIC Score', fontsize=14)
    ax3.set_xlabel('Time Offset $\Delta t$ (s)', fontsize=12)
    ax3.set_ylabel('Freq Bias $\Delta f$ (Hz)', fontsize=12)
    
    ax3.scatter(aic_min_idx[1] + 0.5, aic_min_idx[0] + 0.5, marker='*', s=400, color='red', edgecolor='white', label='AIC Minimum')
    ax3.scatter(true_dt_idx + 0.5, true_df_idx + 0.5, marker='D', s=150, color='lime', edgecolor='black', label='Physical Minimum')
    ax3.legend(loc='upper right')
    
    ax4 = axes[1, 1]
    sns.heatmap(bic_2d, ax=ax4, cmap=cmap_opt, **kwargs)
    ax4.set_title(f'[{sat_id}] 2D Model: BIC Score', fontsize=14)
    ax4.set_xlabel('Time Offset $\Delta t$ (s)', fontsize=12)
    ax4.set_ylabel('Freq Bias $\Delta f$ (Hz)', fontsize=12)
    
    ax4.scatter(bic_min_idx[1] + 0.5, bic_min_idx[0] + 0.5, marker='*', s=400, color='orange', edgecolor='white', label='BIC Minimum')
    ax4.scatter(true_dt_idx + 0.5, true_df_idx + 0.5, marker='D', s=150, color='lime', edgecolor='black', label='Physical Minimum')
    ax4.legend(loc='upper right')

    for ax in [ax3, ax4]:
        ax.set_xticks(np.arange(0, len(DT_GRID), 10) + 0.5)
        ax.set_xticklabels(DT_GRID[::10], rotation=45)

    plt.tight_layout()
    plt.show()


def plot_doppler_discrepancy(df_pass: pd.DataFrame, ref_line1: str, ref_line2: str, 
                             aic_2d: np.ndarray, bic_2d: np.ndarray, err_1d: np.ndarray, 
                             dt_grid: np.ndarray, df_grid: np.ndarray, sat_id: str):
    """Overlays the raw Doppler telemetry with simulated curves."""
    track_line1 = df_pass['tle_line1'].iloc[0]
    track_line2 = df_pass['tle_line2'].iloc[0]
    lat = df_pass['station_lat'].iloc[0]
    lon = df_pass['station_lon'].iloc[0]
    alt_km = df_pass['station_alt'].iloc[0] / 1000.0
    f_carrier = df_pass['expected_frequency'].iloc[0]
    
    times_pd = pd.to_datetime(df_pass["timestamp"], utc=True)
    times_unix = times_pd.astype('int64').values / 10**9
    doppler_meas = df_pass['lr1_receiver1_actualCarrierFrequencyOffset'].values
    
    aic_min_idx = np.unravel_index(np.argmin(aic_2d), aic_2d.shape)
    df_aic_best = df_grid[aic_min_idx[0]]
    dt_aic_best = dt_grid[aic_min_idx[1]]
    
    bic_min_idx = np.unravel_index(np.argmin(bic_2d), bic_2d.shape)
    df_bic_best = df_grid[bic_min_idx[0]]
    dt_bic_best = dt_grid[bic_min_idx[1]]
    
    dt_true_best = dt_grid[np.argmin(err_1d)]
    true_col_idx = np.argmin(err_1d)
    df_true_best = df_grid[np.argmin(aic_2d[:, true_col_idx])]
    
    
    
    base_model = RangeRateModel("Track", track_line1, track_line2)
    eval_model = FlexibleDopplerModel(base_model)
    
    t_tensor = torch.tensor(times_unix, dtype=torch.float64)
    fc_tensor = torch.tensor(f_carrier, dtype=torch.float64)
    
    with torch.no_grad():
        f_pred_aic = eval_model.evaluate(
            torch.tensor(dt_aic_best, dtype=torch.float64), fc_tensor, 
            torch.tensor(df_aic_best, dtype=torch.float64), t_tensor, lat, lon, alt_km
        ).numpy()
        
        f_pred_bic = eval_model.evaluate(
            torch.tensor(dt_bic_best, dtype=torch.float64), fc_tensor, 
            torch.tensor(df_bic_best, dtype=torch.float64), t_tensor, lat, lon, alt_km
        ).numpy()
        
        f_pred_true = eval_model.evaluate(
            torch.tensor(dt_true_best, dtype=torch.float64), fc_tensor, 
            torch.tensor(df_true_best, dtype=torch.float64), t_tensor, lat, lon, alt_km
        ).numpy()

    plt.figure(figsize=(15, 9))
    plt.scatter(times_pd, doppler_meas, color='black', alpha=0.5, s=20, label='Raw Telemetry', zorder=2)
    plt.plot(times_pd, f_pred_aic, color='red', linewidth=2.5, linestyle='--', 
             label=f'AIC Estimate ($\Delta t={dt_aic_best}$s, $\Delta f={df_aic_best}$Hz)', zorder=3)
    plt.plot(times_pd, f_pred_bic, color='orange', linewidth=2.5, linestyle=':', 
             label=f'BIC Estimate ($\Delta t={dt_bic_best}$s, $\Delta f={df_bic_best}$Hz)', zorder=4)
    plt.plot(times_pd, f_pred_true, color='green', linewidth=3.0, 
             label=f'Physical True ($\Delta t={dt_true_best}$s, $\Delta f={df_true_best}$Hz)', zorder=5)
    
    plt.title(f"[{sat_id}] Doppler Discrepancy: Information Criteria vs Physical Minimum", fontsize=15)
    plt.xlabel("Timestamp", fontsize=12)
    plt.ylabel("Frequency Offset (Hz)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.show()

# =============================================================================
# Execution Block Addition
# =============================================================================

if __name__ == "__main__":
    for sat_id, (ref_tle_1, ref_tle_2) in GPS_TLES.items():
        print(f"\n{'='*50}\nProcessing Satellite {sat_id}\n{'='*50}")
        
        try:
            full_df = load_and_clean_data(ROOT_DIR, sat_id)
        except FileNotFoundError:
            print(f"No data found for satellite {sat_id}. Skipping.")
            continue
            
        if full_df.empty:
            print(f"No valid data remaining for satellite {sat_id} after cleaning. Skipping.")
            continue
        
        # Isolate a single pass for coherent grid evaluation
        target_contact = full_df['contact_id'].unique()[0]
        pass_df = full_df[full_df['contact_id'] == target_contact].copy()
        
        # Compute the 1D and 2D Metrics
        err_1d, aic_2d, bic_2d = generate_grid_metrics(pass_df, ref_tle_1, ref_tle_2)
        
        # Render Multi-panel Plot
        print(f"Generating Separated Plots for {sat_id}...")
        plot_separated_results(err_1d, aic_2d, bic_2d, sat_id)
        
        # Generate Discrepancy Plot
        print(f"Generating Doppler Discrepancy Plot for {sat_id}...")
        plot_doppler_discrepancy(pass_df, ref_tle_1, ref_tle_2, aic_2d, bic_2d, err_1d, DT_GRID, DF_GRID, sat_id)