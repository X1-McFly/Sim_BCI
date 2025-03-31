import pandas as pd
from datetime import datetime

# Load steering data from CSV
steering_df = pd.read_csv('wheel_data_20250213_114522.csv')  # Adjust with the correct path

# Remove spaces from all column names
steering_df.columns = steering_df.columns.str.replace(' ', '', regex=False)

# Strip leading and trailing spaces in the 'Timestamp' column
steering_df['Timestamp'] = steering_df['Timestamp'].str.strip()

# Replace colon with dot in the timestamp to make it compatible with datetime format
steering_df['Timestamp'] = steering_df['Timestamp'].str.replace(':', '.', regex=False)

# Convert the 'Timestamp' column to datetime with the correct format
steering_df['Timestamp'] = pd.to_datetime(steering_df['Timestamp'], format='%Y/%m/%d_%H:%M:%S.%f')

# Now proceed with the rest of the code...
print(steering_df)