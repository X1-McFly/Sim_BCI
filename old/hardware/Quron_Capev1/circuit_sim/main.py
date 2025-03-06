import numpy as np

# Parameters
start_freq = 00       # Starting frequency in Hz
end_freq = 200       # Ending frequency in Hz
duration = 5.0       # Total duration in seconds
time_steps = 10000    # Number of time steps (resolution)

# Generate time points
time = np.linspace(0, duration, time_steps)

# Calculate instantaneous frequencies
freq = np.linspace(start_freq, end_freq, time_steps)

# Generate the waveform
wave = np.sin(2 * np.pi * freq * time)

# Write to PWL file
with open("wave.pwl", "w") as file:
    for t, w in zip(time, wave):
        file.write(f"{t:.6f} {w:.6f}\n")
