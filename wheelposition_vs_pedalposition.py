import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the file path
file_path = 'data/Atit(all)/final_data.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)
print("Columns in the dataset:", df.columns.tolist())

# Check if the expected columns are in the DataFrame
expected_cols = ["Steering", "Throttle", "Brake"]
for col in expected_cols:
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' not found in the CSV file.")
    
# Normalize Throttle and Brake columns to start at zero
df["Throttle"] = df["Throttle"] - df["Throttle"].min()
df["Brake"] = df["Brake"] - df["Brake"].min()

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot Throttle vs. Steering
ax1.scatter(df["Steering"], df["Throttle"], alpha=0.6, color='tab:blue')
ax1.set_title("Throttle Position vs. Steering Wheel Position")
ax1.set_xlabel("Steering Wheel Position")
ax1.set_ylabel("Throttle Position")
ax1.grid(True)

# Plot Brake vs. Steering
ax2.scatter(df["Steering"], df["Brake"], alpha=0.6, color='tab:red')
ax2.set_title("Brake Position vs. Steering Wheel Position")
ax2.set_xlabel("Steering Wheel Position")
ax2.set_ylabel("Brake Position")
ax2.grid(True)

plt.tight_layout()

# Ensure the img folder exists
img_folder = 'img'
os.makedirs(img_folder, exist_ok=True)

# Save the plot as a PNG file in the img folder
save_path = os.path.join(img_folder, "plot.png")
plt.savefig(save_path)
print(f"Plot saved to {save_path}")
