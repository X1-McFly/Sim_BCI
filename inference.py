import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

file_path = 'data/Atit(all)/'
model_path = 'models/model_v0.4/'

# Load saved model and scaler.
model = tf.keras.models.load_model(f'{model_path}rnn_model.h5')
scaler = joblib.load(f'{model_path}scaler.save')

# Load cleaned data.
df = pd.read_csv(f'{file_path}final_data.csv')
print("Columns in the dataset:", df.columns.tolist())

# Function to classify steering into binary classes.
def classify_steering(x):
    return 0 if x < 0 else 1  # Left (0) or Right (1)

# Apply classification.
df['Actual_Class'] = df['Steering'].apply(classify_steering)
print("Binary class distribution:")
print(df['Actual_Class'].value_counts())

# Use EEG and FFT columns as features.
eeg_cols = [col for col in df.columns if col.lower().startswith('exg channel')]
fft_cols = ['Alpha_FFT_Left', 'Alpha_FFT_Right',
            'Beta_FFT_Left', 'Beta_FFT_Right',
            'Gamma_FFT_Left', 'Gamma_FFT_Right']
feature_cols = eeg_cols + fft_cols

# Prepare features and targets.
features = df[feature_cols].values
actuals = df['Actual_Class'].values

# Define lookback (same as training).
lookback = 100
if len(features) < lookback:
    raise ValueError("Not enough data for one sequence.")

num_sequences = len(features) - lookback
print("Total valid sequences for inference:", num_sequences)

# Labels for display.
labels = ['Left', 'Right']

predicted_all = []
actual_all = []

print("\nStarting sequential real-time simulation...\n")
step = 2015  # Adjust step size for simulation speed.
for i in range(0, num_sequences, step):
    # Build a sequence from rows i to i+lookback.
    seq = features[i:i+lookback]
    actual_class = actuals[i + lookback]
    raw_steering = df['Steering'].iloc[i + lookback]
    
    # Reshape sequence and standardize.
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
    
    # Log details.
    print("Row {:>5}: Raw Steering: {:>6} | Actual: {:>6} | Predicted: {:>6}".format(
        i + lookback, raw_steering, labels[actual_class], labels[predicted_class]
    ))

# Compute accuracy.
predicted_all = np.array(predicted_all)
actual_all = np.array(actual_all)
overall_accuracy = np.mean(predicted_all == actual_all)

print("\nFinal Accuracy: {:.2f}%".format(overall_accuracy * 100))
