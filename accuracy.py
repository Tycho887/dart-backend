import os
import glob
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import satkit as sk
import time

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

# Confidence Gating Parameters
MAX_TIME_STD_SECONDS = 15  # Maximum allowable 1-sigma uncertainty

# =============================================================================
# Data Pipeline
# =============================================================================

def load_and_clean_data(root_dir: str, sat_id: str) -> pl.DataFrame:
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
    
    # Tracking list for all evaluated passes
    pass_summaries = []
    
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
            
        # Group by contact_id and filter for passes with > 300 samples
        pass_counts = full_df.group_by("contact_id").agg(pl.len().alias("count"))
        valid_contacts = pass_counts.filter(pl.col("count") > 300)["contact_id"].to_list()
        
        # Removed the slice to run ALL valid passes across the dataset
        target_contacts = valid_contacts
        
        if not target_contacts:
            print(f"Skipping {sat_id} - no passes found with > 300 samples.")
            continue
            
        for contact_idx, target_contact in enumerate(target_contacts, start=1):
            pass_df = full_df.filter(pl.col('contact_id') == target_contact)
            
            print(f"\n  -> Pass {contact_idx}/{len(target_contacts)} | Contact {target_contact} | {len(pass_df)} samples")
            
            obs_data, fc_GHz = create_od_data(pass_df)
            
            config = Config(
                x0=np.array([0.0, 0.0, fc_GHz]),
                qmc_bounds=((-60.0, -1e5, fc_GHz - 0.5), (0.0, 1e5, fc_GHz + 0.5)),
                reg_weights=np.array([2e-5, 1e-5, 1e-5]),
                penalty_weight=1.0,
                f_scale=500.0,
                N_degrees=5.0,  
                method="trf",
                loss="cauchy",
                model_type="2-param",
                criterion="AIC",
                use_qmc=True,
                qmc_samples=1024
            )
            
            result = solve_time_shift(obs_data, config, fc_GHz)
            
            time_variance = result.cov[0, 0]
            time_std = np.sqrt(time_variance) if time_variance > 0 else np.inf
            
            # Evaluate error for ALL passes unconditionally
            truth_line1, truth_line2 = GPS_TLES[sat_id]
            truth_tle = sk.TLE.from_lines([truth_line1, truth_line2])
            
            t_array = obs_data.time_array
            dt_seconds = float(result.x[0])
            t_shifted = t_array + sk.duration(seconds=dt_seconds) + sk.duration(seconds=0.35)
            
            p_true, _ = sk.sgp4(truth_tle, t_array)
            p_uncorr, _ = sk.sgp4(obs_data.tle, t_array)
            p_corr, _ = sk.sgp4(obs_data.tle, t_shifted)
            
            err_uncorr_km = np.linalg.norm(p_true - p_uncorr, axis=1) / 1000.0
            err_corr_km = np.linalg.norm(p_true - p_corr, axis=1) / 1000.0

            pass_summaries.append({
                "sat_id": sat_id,
                "contact_id": target_contact,
                "samples": len(pass_df),
                "dt": dt_seconds,
                "dt_std": time_std,
                "mean_error_uncorr": np.mean(err_uncorr_km),
                "mean_error_corr": np.mean(err_corr_km),
                "raw_err_uncorr": err_uncorr_km.tolist(),
                "raw_err_corr": err_corr_km.tolist()
            })
            
            status = "ACCEPTED" if time_std <= MAX_TIME_STD_SECONDS else "REJECTED"
            print(f"     Fit Status: {status} | SSR: {result.ssr:.2f} | dt: {result.x[0]:.3f}s ± {time_std:.3f}s")
            if status == "REJECTED":
                print(f"     [!] Note: Pass failed confidence check but error metrics were recorded.")

    # =====================================================================
    # Global Summary Reporting
    # =====================================================================
    print("\n" + "="*85)
    print(" GLOBAL SOLVER PERFORMANCE SUMMARY ".center(85, "="))
    print("="*85)

    if not pass_summaries:
        print("No passes were evaluated.")
    else:
        # 1. Unfiltered Dataset (All Passes)
        unfiltered_mean_err = np.mean([p["mean_error_corr"] for p in pass_summaries])
        
        # 2. Filtered Dataset (High Confidence Only)
        filtered_passes = [p for p in pass_summaries if p["dt_std"] <= MAX_TIME_STD_SECONDS]
        filtered_mean_err = np.mean([p["mean_error_corr"] for p in filtered_passes]) if filtered_passes else float('inf')
        
        print(f"Total Passes Evaluated : {len(pass_summaries)}")
        print(f"Passes Passing Filter  : {len(filtered_passes)} (Threshold: std <= {MAX_TIME_STD_SECONDS}s)\n")
        
        print(f"Mean Solver Error (UNFILTERED) : {unfiltered_mean_err:.3f} km")
        print(f"Mean Solver Error (FILTERED)   : {filtered_mean_err:.3f} km")
    print("="*85)

    # =====================================================================
    # Aggregate Histogram Plotting (Using FILTERED Data)
    # =====================================================================
    if filtered_passes:
        print("\nGenerating final histogram for high-confidence passes...")
        
        global_err_uncorr_filt = []
        global_err_corr_filt = []
        
        for p in filtered_passes:
            global_err_uncorr_filt.extend(p["raw_err_uncorr"])
            global_err_corr_filt.extend(p["raw_err_corr"])
            
        plt.figure(figsize=(12, 7))
        
        max_err = max(max(global_err_uncorr_filt, default=0), max(global_err_corr_filt, default=0))
        bins = np.linspace(0, max_err, 75)
        
        plt.hist(global_err_uncorr_filt, bins=bins, alpha=0.6, color='orange', edgecolor='black', label='Prior Error (Uncorrected)')
        plt.hist(global_err_corr_filt, bins=bins, alpha=0.6, color='green', edgecolor='black', label='Prior Error (Corrected)')
        
        plt.axvline(np.mean(global_err_uncorr_filt), color='darkorange', linestyle='dashed', linewidth=2, label='Mean Uncorrected')
        plt.axvline(np.mean(global_err_corr_filt), color='darkgreen', linestyle='dashed', linewidth=2, label='Mean Corrected')
        
        plt.title('Aggregate Cartesian Error Distribution Across Confirmed Passes')
        plt.xlabel('Cartesian Prediction Error (km)')
        plt.ylabel('Frequency (Number of Samples)')
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.show()
    else:
        print("\nInsufficient filtered data to generate histogram.")

# import os
# import glob
# import numpy as np
# import polars as pl
# import matplotlib.pyplot as plt
# import satkit as sk
# import time

# from models.common import Data, Config
# from models.time_model import solve_time_shift, simulate_doppler_curve

# # =============================================================================
# # Configuration
# # =============================================================================
# ROOT_DIR = "/home/tycho/leop/doppler_parquet/"

# GPS_TLES = {
#     "16": (
#         "1 90916U 00000AAA 26124.11399935  .00000000  00000-0  43981-3 0  9998",
#         "2 90916  97.7435  22.3832 0001203 162.4543  72.6511 14.91902713    02"
#     ),
#     "17": (
#         "1 90917U 00000AAA 26124.11249883  .00000000  00000-0  44072-3 0  9997",
#         "2 90917  97.7385  22.3866 0005222 306.9095 276.5605 14.90519438    02"
#     ),
#     "18": (
#         "1 90918U 00000AAA 26124.10084548  .00000000  00000-0  44216-3 0  9992",
#         "2 90918  97.7369  22.3788 0001905  77.7453  86.9927 14.91984340    02"
#     ),
#     "19": (
#         "1 90919U 00000AAA 26124.11166446  .00000000  00000-0  83669-3 0  9997",
#         "2 90919  97.7414  22.3835 0001997 127.8106  95.3615 14.92130954    03"
#     )
# }

# # Confidence Gating Parameters
# MAX_TIME_STD_SECONDS = 15  # Maximum allowable 1-sigma uncertainty for the time shift

# # =============================================================================
# # Data Pipeline
# # =============================================================================

# def load_and_clean_data(root_dir: str, sat_id: str) -> pl.DataFrame:
#     """Loads and filters parquet files using Polars lazy execution."""
#     search_pattern = os.path.join(root_dir, f"forest{sat_id}*.parquet")
    
#     if not glob.glob(search_pattern):
#         raise FileNotFoundError(f"No parquet files found matching {search_pattern}")
        
#     doppler_col = 'lr1_receiver1_actualCarrierFrequencyOffset'
    
#     lf = pl.scan_parquet(search_pattern)
    
#     cleaned_df = lf.filter(
#         (pl.col('antenna1_position_elevation') > 1.0) &
#         (pl.col('antenna1_position_elevation') < 89.0) &
#         (pl.col(doppler_col).is_not_null()) &
#         (pl.col(doppler_col) != 0.0) &
#         (pl.col(doppler_col).abs() >= 0.1)
#     ).collect()
    
#     return cleaned_df


# def create_od_data(pass_df: pl.DataFrame) -> tuple[Data, float]:
#     """Maps a single pass DataFrame into the solver's Data dataclass."""
#     tle_line1 = pass_df['tle_line1'][0]
#     tle_line2 = pass_df['tle_line2'][0]
#     sat_tle = sk.TLE.from_lines([tle_line1, tle_line2])
    
#     lat = pass_df['station_lat'][0]
#     lon = pass_df['station_lon'][0]
#     alt = pass_df['station_alt'][0]
    
#     fc_hz = pass_df['expected_frequency'][0]
#     fc_GHz = fc_hz / 1e9
    
#     if pass_df['timestamp'].dtype == pl.Utf8:
#         pass_df = pass_df.with_columns(pl.col('timestamp').str.to_datetime())
        
#     py_datetimes = pass_df['timestamp'].to_list()
#     time_array = np.array([sk.time.from_datetime(dt) for dt in py_datetimes])
    
#     obs_doppler = pass_df['lr1_receiver1_actualCarrierFrequencyOffset'].to_numpy()
#     obs_pointing = np.zeros((len(time_array), 3))
    
#     od_data = Data(
#         time_array=time_array,
#         tle=sat_tle,
#         lat=lat,
#         lon=lon,
#         alt=alt,
#         obs_doppler=obs_doppler,
#         obs_pointing=obs_pointing
#     )
    
#     return od_data, fc_GHz

# # =============================================================================
# # Execution
# # =============================================================================

# if __name__ == "__main__":
    
#     # Global arrays to store point-wise Cartesian errors for the histogram
#     global_err_uncorr = []
#     global_err_corr = []
    
#     # Tracking list for all evaluated passes
#     pass_summaries = []
    
#     for sat_id in GPS_TLES.keys():
#         print(f"\n{'='*50}\nProcessing Satellite {sat_id}\n{'='*50}")
        
#         try:
#             full_df = load_and_clean_data(ROOT_DIR, sat_id)
#         except FileNotFoundError:
#             print(f"Skipping {sat_id} - no data found.")
#             continue
            
#         if full_df.is_empty():
#             print(f"Skipping {sat_id} - no valid data post-filtering.")
#             continue
            
#         # Group by contact_id and filter for passes with > 300 samples
#         pass_counts = full_df.group_by("contact_id").agg(pl.len().alias("count"))
#         valid_contacts = pass_counts.filter(pl.col("count") > 10)["contact_id"].to_list()
        
#         # Isolate the first 5 valid passes
#         target_contacts = valid_contacts[:50]
        
#         if not target_contacts:
#             print(f"Skipping {sat_id} - no passes found with > 300 samples.")
#             continue
            
#         for contact_idx, target_contact in enumerate(target_contacts, start=1):
#             pass_df = full_df.filter(pl.col('contact_id') == target_contact)
            
#             print(f"\n  -> Pass {contact_idx}/5 | Contact {target_contact} | {len(pass_df)} samples")
            
#             obs_data, fc_GHz = create_od_data(pass_df)
            
#             config = Config(
#                 x0=np.array([0.0, 0.0, fc_GHz]),
#                 qmc_bounds=(
#                     (-60.0, -1e5, fc_GHz - 0.5),  
#                     (0.0, 1e5, fc_GHz + 0.5)       
#                 ),
#                 reg_weights=np.array([2e-5, 1e-5, 1e-5]),
#                 penalty_weight=1,
#                 f_scale=500.0,
#                 N_degrees=5.0,  
#                 method="trf",
#                 loss="cauchy",
#                 model_type="2-param",
#                 criterion="AIC",
#                 use_qmc=True,
#                 qmc_samples=1024
#             )
            
#             result = solve_time_shift(obs_data, config, fc_GHz)
            
#             # Extract standard deviation of the time shift from the covariance matrix
#             time_variance = result.cov[0, 0]
#             time_std = np.sqrt(time_variance) if time_variance > 0 else np.inf
            
#             print(f"     Fit Success: {result.success} | SSR: {result.ssr:.2f} | dt: {result.x[0]:.3f}s ± {time_std:.3f}s")
            
#             # Confidence Gating
#             if time_std > MAX_TIME_STD_SECONDS:
#                 print(f"     [!] Rejected: Confidence metric ({time_std:.3f}s) exceeds maximum allowable uncertainty ({MAX_TIME_STD_SECONDS}s).")
#                 continue
                
#             # Error Comparison Logic
#             truth_line1, truth_line2 = GPS_TLES[sat_id]
#             truth_tle = sk.TLE.from_lines([truth_line1, truth_line2])
            
#             t_array = obs_data.time_array
#             dt_seconds = float(result.x[0])
#             t_shifted = t_array + sk.duration(seconds=dt_seconds) + sk.duration(seconds=0.35)
            
#             p_true, _ = sk.sgp4(truth_tle, t_array)
#             p_uncorr, _ = sk.sgp4(obs_data.tle, t_array)
#             p_corr, _ = sk.sgp4(obs_data.tle, t_shifted)
            
#             err_uncorr_km = np.linalg.norm(p_true - p_uncorr, axis=1) / 1000.0
#             err_corr_km = np.linalg.norm(p_true - p_corr, axis=1) / 1000.0
            
#             # Accumulate pointwise errors for the histogram
#             global_err_uncorr.extend(err_uncorr_km.tolist())
#             global_err_corr.extend(err_corr_km.tolist())

#             # Store summary data for reporting
#             pass_summaries.append({
#                 "sat_id": sat_id,
#                 "contact_id": target_contact,
#                 "samples": len(pass_df),
#                 "dt": dt_seconds,
#                 "dt_std": time_std,
#                 "mean_error": np.median(err_corr_km)
#             })

#     # =====================================================================
#     # Tiered Accuracy Reports
#     # =====================================================================
#     def print_accuracy_report(passes: list, threshold: float):
#         """Filters passes by threshold and prints a formatted summary table."""
#         filtered_passes = [p for p in passes if p["mean_error"] < threshold]
        
#         print("\n" + "="*85)
#         title = f" passes with recovered accuracy < {threshold:.1f} km "
#         print(title.upper().center(85, " "))
#         print("="*85)
        
#         if not filtered_passes:
#             print(f"No passes achieved a mean recovered accuracy below {threshold:.1f} km.")
#             print("="*85)
#             return

#         print(f"{'Sat ID':<8} | {'Contact ID':<20} | {'Samples':<8} | {'Time Shift':<12} | {'Uncertainty':<12} | {'Mean Error'}")
#         print("-" * 85)
        
#         # Sort by mean error (ascending) for cleaner presentation
#         for p in sorted(filtered_passes, key=lambda x: x["mean_error"]):
#             print(f"{p['sat_id']:<8} | {p['contact_id']:<20} | {p['samples']:<8} | {p['dt']:>8.3f} s  | {p['dt_std']:>8.3f} s  | {p['mean_error']:>7.3f} km")
#         print("="*85)

#     # Generate the requested reports
#     if pass_summaries:
#         print_accuracy_report(pass_summaries, 1.0)
#         print_accuracy_report(pass_summaries, 2.0)
#         print_accuracy_report(pass_summaries, 5.0)

#     # =====================================================================
#     # Aggregate Histogram Plotting
#     # =====================================================================
#     if global_err_uncorr and global_err_corr:
#         print("\nGenerating final histogram...")
        
#         plt.figure(figsize=(12, 7))
        
#         max_err = max(max(global_err_uncorr, default=0), max(global_err_corr, default=0))
#         bins = np.linspace(0, max_err, 75)
        
#         plt.hist(global_err_uncorr, bins=bins, alpha=0.6, color='orange', edgecolor='black', label='Prior Error (Uncorrected)')
#         plt.hist(global_err_corr, bins=bins, alpha=0.6, color='green', edgecolor='black', label='Prior Error (Corrected)')
        
#         plt.axvline(np.median(global_err_uncorr), color='darkorange', linestyle='dashed', linewidth=2, label='Median Uncorrected')
#         plt.axvline(np.median(global_err_corr), color='darkgreen', linestyle='dashed', linewidth=2, label='Median Corrected')
        
#         plt.title('Aggregate Cartesian Error Distribution Across Confirmed Passes')
#         plt.xlabel('Cartesian Prediction Error (km)')
#         plt.ylabel('Frequency (Number of Samples)')
#         plt.legend()
#         plt.grid(True, alpha=0.3, linestyle='--')
        
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("\nInsufficient data to generate histogram.")
        
# # if __name__ == "__main__":
    
# #     # Tracking list for all evaluated passes
# #     pass_summaries = []
    
# #     for sat_id in GPS_TLES.keys():
# #         print(f"\n{'='*50}\nProcessing Satellite {sat_id}\n{'='*50}")
        
# #         try:
# #             full_df = load_and_clean_data(ROOT_DIR, sat_id)
# #         except FileNotFoundError:
# #             print(f"Skipping {sat_id} - no data found.")
# #             continue
            
# #         if full_df.is_empty():
# #             print(f"Skipping {sat_id} - no valid data post-filtering.")
# #             continue
            
# #         # Group by contact_id and filter for passes with > 300 samples
# #         pass_counts = full_df.group_by("contact_id").agg(pl.len().alias("count"))
# #         valid_contacts = pass_counts.filter(pl.col("count") > 200)["contact_id"].to_list()
        
# #         # Isolate the first 5 valid passes
# #         target_contacts = valid_contacts[:5]
        
# #         if not target_contacts:
# #             print(f"Skipping {sat_id} - no passes found with > 300 samples.")
# #             continue
            
# #         for contact_idx, target_contact in enumerate(target_contacts, start=1):
# #             pass_df = full_df.filter(pl.col('contact_id') == target_contact)
            
# #             print(f"\n  -> Pass {contact_idx}/5 | Contact {target_contact} | {len(pass_df)} samples")
            
# #             obs_data, fc_GHz = create_od_data(pass_df)
            
# #             config = Config(
# #                 x0=np.array([0.0, 0.0, fc_GHz]),
# #                 qmc_bounds=(
# #                     (-60.0, -1e5, fc_GHz - 0.5),  # Lower bounds
# #                     (0.0, 1e5, fc_GHz + 0.5)       # Upper bounds
# #                 ),
# #                 reg_weights=np.array([7e-5, 1e-5, 2e-7]),
# #                 penalty_weight=1e-2,
# #                 f_scale=500.0,
# #                 N_degrees=5.0,  
# #                 method="trf",
# #                 loss="cauchy",
# #                 model_type="2-param",
# #                 criterion="AIC",
# #                 use_qmc=True,
# #                 qmc_samples=1024
# #             )
            
            
# #             result = solve_time_shift(obs_data, config, fc_GHz)
            
# #             print(f"     Fit Success: {result.success} | SSR: {result.ssr:.2f} | dt: {result.x[0]:.3f}s")
            
# #             # Error Comparison Logic
# #             truth_line1, truth_line2 = GPS_TLES[sat_id]
# #             truth_tle = sk.TLE.from_lines([truth_line1, truth_line2])
            
# #             t_array = obs_data.time_array
# #             dt_seconds = float(result.x[0])
# #             t_shifted = t_array + sk.duration(seconds=dt_seconds) + sk.duration(seconds=0.350)
            
# #             p_true, _ = sk.sgp4(truth_tle, t_array)
# #             p_uncorr, _ = sk.sgp4(obs_data.tle, t_array)
# #             p_corr, _ = sk.sgp4(obs_data.tle, t_shifted)
            
# #             err_uncorr_km = np.linalg.norm(p_true - p_uncorr, axis=1) / 1000.0
# #             err_corr_km = np.linalg.norm(p_true - p_corr, axis=1) / 1000.0

# #             # Store summary data for reporting and final histogram
# #             pass_summaries.append({
# #                 "sat_id": sat_id,
# #                 "contact_id": target_contact,
# #                 "samples": len(pass_df),
# #                 "dt": dt_seconds,
# #                 "mean_error_uncorr": np.mean(err_uncorr_km),
# #                 "mean_error_corr": np.mean(err_corr_km)
# #             })

# #     # =====================================================================
# #     # Tiered Accuracy Reports
# #     # =====================================================================
# #     def print_accuracy_report(passes: list, threshold: float):
# #         """Filters passes by threshold and prints a formatted summary table."""
# #         filtered_passes = [p for p in passes if p["mean_error_corr"] < threshold]
        
# #         print("\n" + "="*70)
# #         title = f" passes with recovered accuracy < {threshold:.1f} km "
# #         print(title.upper().center(70, " "))
# #         print("="*70)
        
# #         if not filtered_passes:
# #             print(f"No passes achieved a mean recovered accuracy below {threshold:.1f} km.")
# #             print("="*70)
# #             return

# #         print(f"{'Sat ID':<8} | {'Contact ID':<20} | {'Samples':<8} | {'Time Shift':<12} | {'Mean Error'}")
# #         print("-" * 70)
        
# #         for p in sorted(filtered_passes, key=lambda x: x["mean_error_corr"]):
# #             print(f"{p['sat_id']:<8} | {p['contact_id']:<20} | {p['samples']:<8} | {p['dt']:>8.3f} s  | {p['mean_error_corr']:>7.3f} km")
# #         print("="*70)

# #     if pass_summaries:
# #         print_accuracy_report(pass_summaries, 1.0)
# #         print_accuracy_report(pass_summaries, 2.5)
# #         print_accuracy_report(pass_summaries, 5.0)

# #     # =====================================================================
# #     # Categorical Histogram (Bar Chart)
# #     # =====================================================================
# #     if pass_summaries:
# #         print("\nGenerating categorical error histogram...")
        
# #         # Define bin edges and labels
# #         bins = [0, 1.0, 2.5, 5.0, 10.0, np.inf]
# #         labels = ['0-1 km', '1-2.5 km', '2.5-5 km', '5-10 km', '> 10 km']
        
# #         uncorr_means = [p['mean_error_uncorr'] for p in pass_summaries]
# #         corr_means = [p['mean_error_corr'] for p in pass_summaries]
        
# #         # Compute histogram counts for the defined bins
# #         uncorr_counts, _ = np.histogram(uncorr_means, bins=bins)
# #         corr_counts, _ = np.histogram(corr_means, bins=bins)
        
# #         x = np.arange(len(labels))
# #         width = 0.35
        
# #         fig, ax = plt.subplots(figsize=(10, 6))
        
# #         # Plot grouped bars
# #         rects1 = ax.bar(x - width/2, uncorr_counts, width, label='Prior Error (Uncorrected)', color='orange', edgecolor='black', alpha=0.8)
# #         rects2 = ax.bar(x + width/2, corr_counts, width, label='Prior Error (Corrected)', color='green', edgecolor='black', alpha=0.8)
        
# #         # Styling and labels
# #         ax.set_ylabel('Number of Passes')
# #         ax.set_title('Pass Accuracy Distribution by Cartesian Error Bins')
# #         ax.set_xticks(x)
# #         ax.set_xticklabels(labels)
# #         ax.legend()
# #         ax.grid(axis='y', alpha=0.3, linestyle='--')
        
# #         # Attach exact count numbers on top of the bars
# #         ax.bar_label(rects1, padding=3)
# #         ax.bar_label(rects2, padding=3)
        
# #         # Force y-axis to use integers only
# #         ax.yaxis.get_major_locator().set_params(integer=True)
        
# #         plt.tight_layout()
# #         plt.show()
# #     else:
# #         print("\nInsufficient data to generate histogram.")