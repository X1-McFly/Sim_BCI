import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import time

# Load saved model and scaler.
model = tf.keras.models.load_model('rnn_model.h5')
scaler = joblib.load('scaler.save')

# Load cleaned data.
df = pd.read_csv('cleaned_data.csv')
print("Columns in the dataset:", df.columns.tolist())

# Function to classify steering into 5 classes.
def classify_steering(x):
    if x < -1500:
        return 0  # Hard Left
    elif x < -500:
        return 1  # Slight Left
    elif x <= 500:
        return 2  # Centered
    elif x <= 1500:
        return 3  # Slight Right
    else:
        return 4  # Hard Right

# Compute the "actual" classification for each row based on raw steering.
df['Actual_Class'] = df['Steering'].apply(classify_steering)

# --- Filter out centered samples and remap classes as in training ---
# Remove samples where class is Centered (2)
df = df[df['Actual_Class'] != 2].reset_index(drop=True)
# Remap remaining classes: Hard Left (0)->0, Slight Left (1)->1, Slight Right (3)->2, Hard Right (4)->3.
def remap_class(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    elif x == 3:
        return 2
    elif x == 4:
        return 3
df['Actual_Class'] = df['Actual_Class'].apply(remap_class)
print("Filtered & remapped class distribution:")
print(df['Actual_Class'].value_counts())

# For inference, use only EEG and FFT columns as features.
eeg_cols = [col for col in df.columns if col.lower().startswith('exg channel')]
fft_cols = ['Alpha_FFT_Left', 'Alpha_FFT_Right',
            'Beta_FFT_Left', 'Beta_FFT_Right',
            'Gamma_FFT_Left', 'Gamma_FFT_Right']
feature_cols = eeg_cols + fft_cols

# Prepare features and targets.
features = df[feature_cols].values
actuals = df['Actual_Class'].values

# Define lookback (same as during training).
lookback = 20
if len(features) < lookback:
    raise ValueError("Not enough data for one sequence.")

num_sequences = len(features) - lookback
print("Total valid sequences for inference:", num_sequences)

# Labels for display.
labels = ['Hard Left', 'Slight Left', 'Slight Right', 'Hard Right']

predicted_all = []
actual_all = []

print("\nStarting sequential real-time simulation (processing every 500th sequence)...\n")
step = 500  # Adjust the step size to simulate sampling.
for i in range(0, num_sequences, step):
    # Build a sequence from rows i to i+lookback.
    seq = features[i:i+lookback]
    # The prediction is for row i+lookback.
    actual_class = actuals[i + lookback]
    raw_steering = df['Steering'].iloc[i + lookback]
    
    # Reshape sequence to (1, lookback, num_features) and standardize.
    seq = np.expand_dims(seq, axis=0)
    num_features = seq.shape[2]
    seq_reshaped = seq.reshape(-1, num_features)
    seq_scaled = scaler.transform(seq_reshaped)
    seq_scaled = seq_scaled.reshape(1, lookback, num_features)
    
    # Run inference.
    pred = model.predict(seq_scaled)
    predicted_class = np.argmax(pred, axis=1)[0]
    
    predicted_all.append(predicted_class)
    actual_all.append(actual_class)
    
    # Log current row details.
    print("Row {:>5}: Raw Steering: {:>6} | Actual: {:>12} | Predicted: {:>12}".format(
        i + lookback, raw_steering, labels[actual_class], labels[predicted_class]
    ))
    
    # Optionally pause to simulate real time.
    # time.sleep(0.05)

predicted_all = np.array(predicted_all)
actual_all = np.array(actual_all)
overall_accuracy = np.mean(predicted_all == actual_all)
error_rate = (1 - overall_accuracy) * 100

print("\nFinal Accuracy (sampled): {:.2f}%".format(overall_accuracy * 100))
print("Percentage of time wrong (sampled): {:.2f}%".format(error_rate))
