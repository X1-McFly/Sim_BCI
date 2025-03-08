import numpy as np
import pandas as pd
import time
from numpy.lib.stride_tricks import sliding_window_view

file_path = 'data/Atit(Session5)/'

def compute_fft():
    # Parameters
    WINDOW_SIZE = 200  # Number of rows per FFT window (e.g., one second of data)
    FS = 200.0         # Sampling frequency in Hz

    # Frequency band definitions (in Hz; resolution = 1 Hz)
    alpha_band = (8, 12)   # Alpha: 8–12 Hz
    beta_band  = (12, 30)  # Beta: 12–30 Hz
    gamma_band = (30, 100) # Gamma: 30–100 Hz

    # Load the cleaned data (make sure cleaned_data.csv is in your working directory)
    df = pd.read_csv(f'{file_path}merged_data.csv')
    print("Data loaded. Total rows:", len(df))

    # Ensure that the FFT columns exist (initialize if missing)
    for col in ['Alpha_FFT_Left', 'Alpha_FFT_Right', 
                'Beta_FFT_Left', 'Beta_FFT_Right', 
                'Gamma_FFT_Left', 'Gamma_FFT_Right']:
        if col not in df.columns:
            df[col] = np.nan

    # Determine EEG channels for left and right.
    # Adjust the names if needed (here we assume left: channels 0 & 1, right: channels 2 & 3)
    left_cols = [col for col in df.columns if col.lower().startswith('exg channel 0') or 
                col.lower().startswith('exg channel 1')]
    right_cols = [col for col in df.columns if col.lower().startswith('exg channel 2') or 
                col.lower().startswith('exg channel 3')]

    if not left_cols or not right_cols:
        raise ValueError("Could not find the required EEG channel columns. Check the column names in cleaned_data.csv.")

    # Compute the average signal for left and right channels.
    df['Avg_Left'] = df[left_cols].mean(axis=1)
    df['Avg_Right'] = df[right_cols].mean(axis=1)

    # Check that we have enough data for one window.
    n_samples = len(df)
    if n_samples < WINDOW_SIZE:
        raise ValueError("Not enough rows in data to compute FFT with the specified window size.")

    print("Computing sliding window FFT for Alpha, Beta, and Gamma bands...")
    start_time = time.time()

    # Create sliding windows (each row of windows_left/right is one window of size WINDOW_SIZE)
    windows_left = sliding_window_view(df['Avg_Left'].values, WINDOW_SIZE)
    windows_right = sliding_window_view(df['Avg_Right'].values, WINDOW_SIZE)

    # Compute the FFT for each window using np.fft.rfft (real FFT)
    fft_left = np.fft.rfft(windows_left, axis=1)
    fft_right = np.fft.rfft(windows_right, axis=1)

    # Compute the magnitude (absolute value) of the FFT coefficients.
    mag_left = np.abs(fft_left)
    mag_right = np.abs(fft_right)

    # Helper function to compute the average magnitude in a frequency band.
    def compute_band_avg(mag, band):
        start_bin, end_bin = band
        max_bin = mag.shape[1] - 1
        if end_bin > max_bin:
            end_bin = max_bin
        return np.mean(mag[:, start_bin:end_bin+1], axis=1)

    # Compute band averages for left channel.
    alpha_left = compute_band_avg(mag_left, alpha_band)
    beta_left  = compute_band_avg(mag_left, beta_band)
    gamma_left = compute_band_avg(mag_left, gamma_band)

    # Compute band averages for right channel.
    alpha_right = compute_band_avg(mag_right, alpha_band)
    beta_right  = compute_band_avg(mag_right, beta_band)
    gamma_right = compute_band_avg(mag_right, gamma_band)

    elapsed = time.time() - start_time
    print("FFT computation complete in {:.2f} seconds.".format(elapsed))

    # Create full-length arrays (with NaN for rows before a full window is available)
    def create_full_array(computed_vals):
        full_array = np.full(n_samples, np.nan)
        full_array[WINDOW_SIZE-1:] = computed_vals
        return full_array

    alpha_left_full  = create_full_array(alpha_left)
    beta_left_full   = create_full_array(beta_left)
    gamma_left_full  = create_full_array(gamma_left)
    alpha_right_full = create_full_array(alpha_right)
    beta_right_full  = create_full_array(beta_right)
    gamma_right_full = create_full_array(gamma_right)

    # Update the DataFrame columns.
    df['Alpha_FFT_Left']  = alpha_left_full
    df['Beta_FFT_Left']   = beta_left_full
    df['Gamma_FFT_Left']  = gamma_left_full
    df['Alpha_FFT_Right'] = alpha_right_full
    df['Beta_FFT_Right']  = beta_right_full
    df['Gamma_FFT_Right'] = gamma_right_full

    # Optionally, drop the temporary average columns.
    df.drop(columns=['Avg_Left', 'Avg_Right'], inplace=True)

    # Remove the first WINDOW_SIZE rows (which don't have a full FFT window)
    df = df.iloc[WINDOW_SIZE:].reset_index(drop=True)

    # Save the updated DataFrame back to cleaned_data.csv.
    df.to_csv(f'{file_path}final_data.csv', index=False)
    print("Updated final_data.csv with computed Alpha, Beta, and Gamma FFT columns.")

if __name__ == "__main__":
          compute_fft()
