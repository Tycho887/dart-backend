import numpy as np
import satkit as sk
import matplotlib.pyplot as plt
import time

# Import shared structures and the specific model solver
from models.common import Data, Config
from models.time_model import solve_time_shift, simulate_doppler_curve

if __name__ == "__main__":
    tle_lines = [
        '0 STARLINK-30477',
        '1 57912U 23146X   24099.49439401  .00006757  00000+0  51475-3 0  9997',
        '2 57912  43.0018 157.5807 0001420 272.5369  87.5310 15.02537576 31746'
    ]

    iss = sk.TLE.from_lines(tle_lines)
    time_array = np.array([iss.epoch + sk.duration(minutes=x) for x in np.linspace(0, 300, 1500)])

    # True state: time_shift (s), bias (Hz), center frequency (GHz)
    hidden_x = np.array([-10.0, 1e4, 2.0])
    
    # Generate pristine data
    temp_data = Data(
        time_array=time_array, 
        tle=iss, 
        lat=42.0, 
        lon=-71.0, 
        alt=100.0, 
        obs_doppler=np.zeros_like(time_array), 
        obs_pointing=np.zeros((len(time_array), 3))
    )
    
    # --- FIXED: Using the new simulation wrapper ---
    doppler_clean, pointing_clean = simulate_doppler_curve(hidden_x, temp_data)
    # -----------------------------------------------
    
    # Add Gaussian + Laplacian noise
    obs_doppler = doppler_clean + np.random.normal(scale=500.0, size=len(time_array)) + np.random.laplace(scale=3000.0, size=len(time_array))
    obs_pointing = pointing_clean
    
    # Apply dropout mask to simulate missing data segments
    dropout_mask = np.random.choice([True, False], size=len(obs_doppler), p=[0.9, 0.1])
    
    # Final data object
    obs_data = Data(
        time_array=time_array[dropout_mask],
        tle=iss,
        lat=42.0,
        lon=-71.0,
        alt=100.0,
        obs_doppler=obs_doppler[dropout_mask],
        obs_pointing=obs_pointing[dropout_mask]
    )
    
    # Configure the solver parameters
    config = Config(
        x0=np.array([0.0, 0.0, 2.0]),
        reg_weights=np.array([1e-5, 1e-4, 1e-5]),  # Stabilizing regularizer anchors
        penalty_weight=1.0,
        N_degrees=1.0,
        method="lm",
        loss="linear",
        f_scale=500.0
    )

    t0 = time.time()
    print("Starting Least Squares Optimization (Time-Shift Model)...")
    
    # Execute the solver
    result = solve_time_shift(obs_data, config)
    
    t1 = time.time()
    print(f"Optimization Time: {t1 - t0:.4f} seconds\n")

    print(f"Hidden State:    {hidden_x}")
    print(f"Fitted State:    {result.x}")
    print(f"Success:         {result.success}")
    print(f"SSR:             {result.ssr:.2f}")
    print(f"Covariance:\n{result.cov}")
    
    # --- FIXED: Re-evaluating visual plots via the wrapper ---
    fitted_doppler, _ = simulate_doppler_curve(result.x, obs_data)
    # ---------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    times = [t.as_datetime() for t in obs_data.time_array]
    
    # Top Plot: True vs Observed vs Fitted Doppler
    ax1.scatter(times, obs_data.obs_doppler, color='blue', label='Noisy Observations', s=5, alpha=0.2)
    ax1.plot(times, doppler_clean[dropout_mask], color='red', label='True Clean Doppler', linewidth=3, alpha=0.7)
    ax1.plot(times, fitted_doppler, color='black', linestyle=':', label='Fitted Model', linewidth=2)
    ax1.set_ylabel('Doppler (Hz)')
    ax1.set_title('Time-Shift Model: Doppler Shift Fit')
    ax1.legend()
    
    # Bottom Plot: Residuals
    residuals = obs_data.obs_doppler - fitted_doppler
    ax2.scatter(times, residuals, color='purple', s=5, alpha=0.5, label='Residuals')
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Residual Error (Hz)')
    ax2.set_xlabel('Time (UTC)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()