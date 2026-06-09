import numpy as np
import satkit as sk
import matplotlib.pyplot as plt
import time
from sgp4.api import Satrec

# Import shared structures and the specific model solver
from models.common import Data, Config
from models.mean_element_model import (
    solve_mean_elements, 
    multipass_doppler_model, 
    get_pass_indicators
)

if __name__ == "__main__":
    tle_lines = [
        '0 STARLINK-30477',
        '1 57912U 23146X   24099.49439401  .00006757  00000+0  51475-3 0  9997',
        '2 57912  43.0018 157.5807 0001420 272.5369  87.5310 15.02537576 31746'
    ]
    line1 = tle_lines[1]
    line2 = tle_lines[2]

    iss = sk.TLE.from_lines(tle_lines)
    
    # ---------------------------------------------------------
    # 1. Simulate 3 Distinct Passes
    # ---------------------------------------------------------
    # Create three 15-minute observation windows separated by ~90 minute orbital periods
    points_per_pass = 500
    pass_durations = np.linspace(0, 90, points_per_pass)
    
    chunk_1 = np.array([iss.epoch + sk.duration(minutes=x) for x in pass_durations])
    chunk_2 = np.array([iss.epoch + sk.duration(minutes=x + 95*2) for x in pass_durations])
    chunk_3 = np.array([iss.epoch + sk.duration(minutes=x + 190*2) for x in pass_durations])
    
    time_array = np.concatenate([chunk_1, chunk_2, chunk_3])
    
    # Extract base Keplerian elements to apply the shift
    sat = Satrec.twoline2rv(line1, line2)
    base_kep = np.array([sat.inclo, sat.nodeo, sat.ecco, sat.argpo, sat.mo, sat.no_kozai])
    
    # ---------------------------------------------------------
    # 2. Define the Hidden State
    # ---------------------------------------------------------
    # Shift Mean Anomaly by +0.01 rad and Mean Motion by +0.0005 rad/min
    hidden_mo = sat.mo + 0.1
    hidden_n = sat.no_kozai - 0.01
    
    # Define independent biases for the 3 passes (Hz)
    hidden_biases = [8500.0, -4200.0, 1500.0]
    
    hidden_x = np.array([hidden_mo, hidden_n] + hidden_biases)
    fc_GHz = 2.0
    
    # Generate pristine data containers
    temp_data = Data(
        time_array=time_array, 
        tle=iss, 
        lat=42.0, 
        lon=-71.0, 
        alt=100.0, 
        obs_doppler=np.zeros_like(time_array), 
        obs_pointing=np.zeros((len(time_array), 3))
    )
    
    # Calculate true physics using the multipass forward model
    pass_indicators = get_pass_indicators(time_array)
    doppler_clean, pointing_clean = multipass_doppler_model(
        hidden_x, pass_indicators, base_kep, line1, line2, temp_data, fc_GHz
    )
    
    # Add Gaussian noise and simulate dropped packets
    obs_doppler = doppler_clean + np.random.normal(scale=100.0, size=len(time_array)) + np.random.laplace(scale=100.0, size=len(time_array))
    dropout_mask = np.random.choice([True, False], size=len(obs_doppler), p=[0.99, 0.01])
    
    obs_data = Data(
        time_array=time_array[dropout_mask],
        tle=iss,
        lat=42.0,
        lon=-71.0,
        alt=100.0,
        obs_doppler=obs_doppler[dropout_mask],
        obs_pointing=pointing_clean[dropout_mask]
    )
    # ---------------------------------------------------------
    # 3. Configure and Execute Solver with Unified Bounds
    # ---------------------------------------------------------
    # Extract the exact number of passes present in the final observation data
    pass_indicators_actual = get_pass_indicators(obs_data.time_array)
    n_passes_actual = pass_indicators_actual.shape[1]
    n_params_total = 2 + n_passes_actual

    # Formulate boundaries matching the mean element vector: [mo, no_kozai, bias_1, ..., bias_N]
    # Guard the Mean Anomaly bounds to ensure compliance with SGP4 constraints [0, 2*pi]
    lb_mo = max(0.0, sat.mo - 0.1)
    ub_mo = min(2 * np.pi, sat.mo + 0.1)

    lower_bounds = [lb_mo, sat.no_kozai - 0.002] + [-15000.0] * n_passes_actual
    upper_bounds = [ub_mo, sat.no_kozai + 0.002] + [15000.0] * n_passes_actual

    config = Config(
        penalty_weight=1.0,
        N_degrees=1.0,
        loss="huber",      # Enabled Huber loss for Laplacian outliers
        f_scale=100.0,     # Outliers beyond 100 Hz are downweighted
        method="dogbox",   # Bounded optimization algorithm
        reg_weights=np.zeros(n_params_total),
        scale_by_jacobian=True,
        qmc_bounds=(lower_bounds, upper_bounds)  # Injecting the unified constraints
    )

    t0 = time.time()
    print("Starting Least Squares Optimization (Mean Element Model)...")
    
    result = solve_mean_elements(obs_data, config, line1, line2, fc_GHz)
    
    t1 = time.time()
    print(f"Optimization Time: {t1 - t0:.4f} seconds\n")

    # ---------------------------------------------------------
    # 4. Console Reporting
    # ---------------------------------------------------------
    print("--- State Comparison ---")
    print(f"Passes Identified: {result.passes_found}")
    print(f"Base Mo:           {sat.mo:.6f}")
    print(f"True Mo:           {hidden_x[0]:.6f}")
    print(f"Fitted Mo:         {result.x[0]:.6f}\n")
    
    print(f"Base Mean Motion:  {sat.no_kozai:.6f}")
    print(f"True Mean Motion:  {hidden_x[1]:.6f}")
    print(f"Fitted Mean Motion:{result.x[1]:.6f}\n")
    
    for p in range(result.passes_found):
        print(f"Pass {p+1} True Bias:   {hidden_x[2+p]:.2f} Hz")
        print(f"Pass {p+1} Fit Bias:    {result.x[2+p]:.2f} Hz")

    print(f"\nOptimizer Success: {result.success}")
    print(f"Message: {result.message}")
    
    # ---------------------------------------------------------
    # 5. Visualization
    # ---------------------------------------------------------
    # Re-evaluate the model using the fitted state to get the full predicted curve
    fitted_indicators = get_pass_indicators(obs_data.time_array)
    fitted_doppler, _ = multipass_doppler_model(
        result.x, fitted_indicators, base_kep, line1, line2, obs_data, fc_GHz
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    times = [t.as_datetime() for t in obs_data.time_array]
    
    # Plot 1: Doppler Fit
    ax1.scatter(times, obs_data.obs_doppler, color='blue', label='Noisy Observations', s=5, alpha=0.3)
    
    # Use scatter for the clean and fitted models to cleanly handle the 90-minute temporal gaps
    times_clean = [t.as_datetime() for t in temp_data.time_array[dropout_mask]]
    ax1.scatter(times_clean, doppler_clean[dropout_mask], color='red', label='True Doppler', s=2, alpha=0.8)
    ax1.scatter(times, fitted_doppler, color='black', label='Fitted Model', s=2)
    
    ax1.set_ylabel('Doppler (Hz)')
    ax1.set_title('Multi-Pass Mean Element Model: Doppler Shift Fit')
    ax1.legend()
    
    # Plot 2: Residuals
    residuals = obs_data.obs_doppler - fitted_doppler
    ax2.scatter(times, residuals, color='purple', s=5, alpha=0.5, label='Residuals')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Residual Error (Hz)')
    ax2.set_xlabel('Time (UTC)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()