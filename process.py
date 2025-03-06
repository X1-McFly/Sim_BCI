import os
import glob
import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.fft import fft
from compute_fft import compute_fft

# Set the folder where all files are located.
data_folder = "data/Atit(all)"  # Change to your folder path

# Define file patterns for EEG and accessory (wheel) files.
eeg_pattern = os.path.join(data_folder, "OpenBCI-RAW-*.txt")
acc_pattern = os.path.join(data_folder, "wheel_data_*.csv")

output_file = "merged_data.csv"

# Get list of files for each type.
eeg_files = glob.glob(eeg_pattern)
acc_files = glob.glob(acc_pattern)

print("Found EEG files:", eeg_files)
print("Found Accessory files:", acc_files)

# Process all EEG files.
eeg_dfs = []
for file in eeg_files:
    temp_df = pd.read_csv(file, comment='%')
    # Remove extra whitespace in column names.
    temp_df.columns = temp_df.columns.str.strip()
    # Convert timestamps.
    temp_df['timestamp'] = pd.to_datetime(
        temp_df['Timestamp (Formatted)'].str.replace(' ', ''),
        format='%Y-%m-%d%H:%M:%S.%f'
    )
    # Remove unnecessary columns.
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

# Combine all EEG files into one DataFrame.
try:
    eeg_df = pd.concat(eeg_dfs, ignore_index=True)
except Exception as e:
    print(f"Error combining EEG files: {e}")
    raise

# Process all accessory (wheel) files.
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

# Combine all accessory files into one DataFrame.
acc_df = pd.concat(acc_dfs, ignore_index=True)

# Sort both DataFrames by timestamp.
eeg_df.sort_values('timestamp', inplace=True)
acc_df.sort_values('timestamp', inplace=True)

# Merge using a nearest timestamp join with a tolerance (100ms).
merged_df = pd.merge_asof(
    eeg_df, acc_df, on='timestamp', direction='nearest', tolerance=pd.Timedelta('100ms')
)

# Move 'timestamp' to the first column and capitalize the first letter of each column.
merged_df = merged_df[['timestamp'] + [col for col in merged_df.columns if col != 'timestamp']]
merged_df.columns = [col.capitalize() for col in merged_df.columns]

# Only drop the row if Steering, Throttle, and Brake are all 0.
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
def classify_steering(x):
    if x < -1500:
        return -2  # Hard left
    elif x < -500:
        return -1  # Slight left
    elif x <= 500:
        return 0   # Centered
    elif x <= 1500:
        return 1   # Slight right
    else:
        return 2   # Hard right

merged_df['Steering_Classification'] = merged_df['Steering'].apply(classify_steering)

# Add new columns for throttle and braking classifications and set them to zero.
merged_df['Throttle_Classification'] = 0
merged_df['Braking_Classification'] = 0

# Define the cutoff timestamp; adjust minutes if needed.
cutoff_timestamp = merged_df['Timestamp'].min() + pd.Timedelta(minutes=0)
merged_df = merged_df[merged_df['Timestamp'] >= cutoff_timestamp]

# Save the merged DataFrame to CSV.
output_path = os.path.join(data_folder, output_file)
merged_df.to_csv(output_path, index=False)
print(f"Merged data saved to '{output_path}'")

compute_fft()
