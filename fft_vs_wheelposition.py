import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Define file path and load CSV
file_path = 'data/Atit(all)/final_data.csv'
df = pd.read_csv(file_path)
print("Columns in the dataset:", df.columns.tolist())

# Filter out "center" steering samples (only keep cases where |Steering| > 1000)
df_filtered = df[(df['Steering'] < -1000) | (df['Steering'] > 1000)]
print(f"Filtered dataset shape (|Steering| > 1000): {df_filtered.shape}")

# Define the frequency bands and their corresponding FFT columns
bands = {
    'Alpha': ['Alpha_FFT_Left', 'Alpha_FFT_Right'],
    'Beta': ['Beta_FFT_Left', 'Beta_FFT_Right'],
    'Gamma': ['Gamma_FFT_Left', 'Gamma_FFT_Right']
}

# Ensure the output folder exists.
img_folder = 'img'
os.makedirs(img_folder, exist_ok=True)

# Create one figure with 3 subplots (one per frequency band)
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=False)

for i, (band, cols) in enumerate(bands.items()):
    ax = axs[i]
    
    # Plot regression for the left channel (blue)
    sns.regplot(x='Steering', y=cols[0], data=df_filtered, 
                ax=ax, scatter_kws={'alpha': 0.5}, color='blue', ci=None, label=cols[0])
    # Calculate Pearson correlation for left channel
    r_left, _ = pearsonr(df_filtered['Steering'], df_filtered[cols[0]])
    
    # Plot regression for the right channel (red)
    sns.regplot(x='Steering', y=cols[1], data=df_filtered, 
                ax=ax, scatter_kws={'alpha': 0.5}, color='red', ci=None, label=cols[1])
    # Calculate Pearson correlation for right channel
    r_right, _ = pearsonr(df_filtered['Steering'], df_filtered[cols[1]])
    
    ax.set_title(f"{band} FFT vs Steering\nLeft r = {r_left:.2f}, Right r = {r_right:.2f}")
    ax.set_xlabel("Steering Wheel Position")
    ax.set_ylabel(f"{band} FFT Value")
    ax.legend()
    ax.grid(True)

plt.tight_layout()

# Save the figure as a PNG in the img folder
save_path = os.path.join(img_folder, "eeg_fft_vs_steering_correlation.png")
plt.savefig(save_path)
print(f"Correlation plot saved to {save_path}")
plt.show()
