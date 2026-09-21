from dart.filters import DopplerFilter

filt = DopplerFilter(
    tle_lines=tle_lines,                 # two TLE strings
    receiver=receiver,                   # satkit.itrfcoord
    center_frequency_hz=400e6,
    epoch_unix_s=first_sample_time,
    initial_state=[0.0, 0.0, 0.0],
    initial_covariance=[[4.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1e8]],
    process_noise_rates=[0.0, 0.0, 0.0],  # illustrative static prior
    kind="ukf",                         # "ukf", "srukf", or "ekf"
    innovation_gate=None,                # optional positive NIS threshold
)
accepted, nis = filt.update(sample_time, doppler_hz, variance_hz2)
snapshot = filt.get_state()
time_offset_s, bias_hz, cf_offset_hz = snapshot.state
covariance = snapshot.covariance
