import os
import glob
import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.fft import fft
import time
import joblib
import tensorflow as tf

# Import compute_fft from your compute_fft.py file.
from compute_fft import compute_fft

#####################################
# PART 1: Data Processing and Merging
#####################################

# Set the folder where all files are located.
data_folder = "data/MARTY3/"  # Folder with new data

# Define file patterns for EEG and accessory (wheel) files.
eeg_pattern = os.path.join(data_folder, "OpenBCI-RAW-*.txt")
acc_pattern = os.path.join(data_folder, "wheel_data_*.csv")

merged_output_file = "merged_data.csv"
cleaned_output_file = "cleaned_data.csv"

# Get lists of files.
eeg_files = glob.glob(eeg_pattern)
acc_files = glob.glob(acc_pattern)
print("Found EEG files:", eeg_files)
print("Found Accessory files:", acc_files)

# Process EEG files.
eeg_dfs = []
for file in eeg_files:
    temp_df = pd.read_csv(file, comment='%')
    temp_df.columns = temp_df.columns.str.strip()
    temp_df['timestamp'] = pd.to_datetime(
        temp_df['Timestamp (Formatted)'].str.replace(' ', ''),
        format='%Y-%m-%d%H:%M:%S.%f'
    )
    temp_df.drop(columns=['Timestamp', 'Timestamp (Formatted)'], inplace=True)
    cols_to_drop = [col for col in temp_df.columns if col.startswith('Other')]
    if cols_to_drop:
        temp_df.drop(columns=cols_to_drop, inplace=True)
    cols_to_drop = [col for col in temp_df.columns if col.startswith('Sample Index')]
    if cols_to_drop:
        temp_df.drop(columns=cols_to_drop, inplace=True)
    accel_cols = [col for col in temp_df.columns if col.startswith('Accel')]
    if accel_cols:
        temp_df.drop(columns=accel_cols, inplace=True)
    eeg_dfs.append(temp_df)
eeg_df = pd.concat(eeg_dfs, ignore_index=True)

# Process accessory (wheel) files.
acc_dfs = []
for file in acc_files:
    temp_df = pd.read_csv(file)
    temp_df.columns = temp_df.columns.str.strip()
    temp_df['timestamp'] = pd.to_datetime(
        temp_df['Timestamp'].str.replace(' ', ''),
        format='%Y/%m/%d_%H:%M:%S:%f'
    )
    temp_df.drop(columns=['Timestamp'], inplace=True)
    acc_dfs.append(temp_df)
acc_df = pd.concat(acc_dfs, ignore_index=True)

# Sort both DataFrames by timestamp.
eeg_df.sort_values('timestamp', inplace=True)
acc_df.sort_values('timestamp', inplace=True)

# Merge using a nearest timestamp join with a tolerance (100ms).
merged_df = pd.merge_asof(
    eeg_df, acc_df, on='timestamp', direction='nearest', tolerance=pd.Timedelta('100ms')
)
merged_df = merged_df[['timestamp'] + [col for col in merged_df.columns if col != 'timestamp']]
merged_df.columns = [col.capitalize() for col in merged_df.columns]

# Drop rows if Steering, Throttle, and Brake are all 0 (or NaN).
mask_all_zero = (
    ((merged_df['Steering'] == 0) | merged_df['Steering'].isna()) &
    ((merged_df['Throttle'] == 0) | merged_df['Throttle'].isna()) &
    ((merged_df['Brake'] == 0) | merged_df['Brake'].isna())
)
merged_df = merged_df[~mask_all_zero]

# Define a function to classify steering into 5 classes:
# -2: Hard left (< -1500)
# -1: Slight left (between -1500 and -500)
#  0: Centered (between -500 and 500)
#  1: Slight right (between 500 and 1500)
#  2: Hard right (> 1500)
def classify_steering_value(x):
    if x < -1500:
        return -2
    elif x < -500:
        return -1
    elif x <= 500:
        return 0
    elif x <= 1500:
        return 1
    else:
        return 2

merged_df['Steering_Classification'] = merged_df['Steering'].apply(classify_steering_value)
merged_df['Throttle_Classification'] = 0
merged_df['Braking_Classification'] = 0

# Apply a cutoff timestamp if needed.
cutoff_timestamp = merged_df['Timestamp'].min() + pd.Timedelta(minutes=0)
merged_df = merged_df[merged_df['Timestamp'] >= cutoff_timestamp]

# Save merged data.
merged_df.to_csv(merged_output_file, index=False)
print(f"Merged data saved to '{merged_output_file}'")

# Compute FFT (this function updates the merged file and saves cleaned_data.csv).
compute_fft()

#####################################
# PART 2: Inference on New Data
#####################################

# Load the cleaned data.
df_cleaned = pd.read_csv(cleaned_output_file)

# For inference, we will use only EEG and FFT columns as input features.
eeg_cols = [col for col in df_cleaned.columns if col.lower().startswith('exg channel')]
fft_cols = ['Alpha_FFT_Left', 'Alpha_FFT_Right',
            'Beta_FFT_Left', 'Beta_FFT_Right',
            'Gamma_FFT_Left', 'Gamma_FFT_Right']
feature_cols = eeg_cols + fft_cols

# Prepare features.
features = df_cleaned[feature_cols].values

# Compute the "actual" classification from the raw steering.
# (For consistency, we use the same classification function.)
df_cleaned['Actual_Class'] = df_cleaned['Steering'].apply(classify_steering_value)
actuals = df_cleaned['Actual_Class'].values

# Define lookback (sequence length used during training).
lookback = 20
if len(features) < lookback:
    raise ValueError("Not enough data for one sequence.")
num_sequences = len(features) - lookback

# Load the scaler.
scaler = joblib.load('scaler.save')

# Labels for display.
labels = ['Hard Left', 'Slight Left', 'Centered', 'Slight Right', 'Hard Right']

predicted_all = []
actual_all = []

print("\nStarting inference on new data (processing every 1000th sequence)...\n")
step = 1000  # Change the step if needed.
for i in range(0, num_sequences, step):
    # Build a sequence from rows i to i+lookback.
    seq = features[i:i+lookback]
    actual_class = actuals[i+lookback]
    raw_steering = df_cleaned['Steering'].iloc[i+lookback]
    
    # Reshape and standardize.
    seq = np.expand_dims(seq, axis=0)
    seq_reshaped = seq.reshape(-1, seq.shape[2])
    seq_scaled = scaler.transform(seq_reshaped)
    seq_scaled = seq_scaled.reshape(1, lookback, seq.shape[2])
    
    # Run inference.
    pred = model.predict(seq_scaled)
    predicted_class = np.argmax(pred, axis=1)[0]
    
    predicted_all.append(predicted_class)
    actual_all.append(actual_class)
    
    print("Row {:>5}: Raw Steering: {:>6} | Actual: {:>12} | Predicted: {:>12}".format(
        i+lookback, raw_steering, labels[actual_class+2], labels[predicted_class+2]
    ))
    
overall_accuracy = np.mean(np.array(predicted_all) == np.array(actual_all))
error_rate = (1 - overall_accuracy) * 100
print("\nFinal Accuracy (sampled): {:.2f}%".format(overall_accuracy * 100))
print("Percentage of time wrong (sampled): {:.2f}%".format(error_rate))
