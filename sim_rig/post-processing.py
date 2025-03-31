import pandas as pd
from datetime import datetime

# Load steering data from CSV
steering_df = pd.read_csv('wheel_data_20250213_114522.csv')  # Adjust with the correct path

# Load EEG data from TXT (assuming it's comma-separated; adjust if it's different)
eeg_df = pd.read_csv('OpenBCI-RAW-2025-02-13_11-45-22.txt', delimiter=',', on_bad_lines='skip')

# Remove spaces from all column names
steering_df.columns = steering_df.columns.str.replace(' ', '', regex=False)

# Check the updated column names
print(steering_df.columns)

steering_df['Timestamp'] = steering_df['Timestamp'].str.replace(':', '.', regex=False)

# Convert the 'Timestamp' column to datetime
steering_df['Timestamp'] = pd.to_datetime(steering_df['Timestamp'], format='%Y/%m/%d_%H:%M:%S.%f')


# Convert EEG data timestamps from Unix time to datetime (assuming the 'Timestamp' is in milliseconds)
eeg_df['Timestamp'] = pd.to_datetime(eeg_df['Timestamp'], unit='ms')

# Sort both dataframes by timestamp for merge_asof to work correctly
steering_df = steering_df.sort_values('Timestamp')
eeg_df = eeg_df.sort_values('Timestamp')

# Merge datasets based on the closest timestamps using merge_asof
merged_df = pd.merge_asof(steering_df, eeg_df, on='Timestamp', direction='nearest')

# Save the result to a CSV file or print it
merged_df.to_csv('merged_data.csv', index=False)  # Save to CSV
print(merged_df)  # Print the merged DataFrame
