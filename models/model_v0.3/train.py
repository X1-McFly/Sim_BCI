import pandas as pd
import numpy as np
import collections
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import joblib  # for saving the scaler
import seaborn as sns

# Load the cleaned data.
df = pd.read_csv('cleaned_data.csv')
print("Columns in the dataset:", df.columns.tolist())

# Reclassify steering into 5 classes using new thresholds.
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

df['Steering_Classification'] = df['Steering'].apply(classify_steering)
print("Original class distribution (5 classes):", collections.Counter(df['Steering_Classification']))

# Remove all samples where classification is Centered (i.e. value 2).
df = df[df['Steering_Classification'] != 2]
print("After removing Centered samples:", collections.Counter(df['Steering_Classification']))

# Remap remaining classes to a contiguous range:
# Hard Left (old 0) -> 0, Slight Left (old 1) -> 1, Slight Right (old 3) -> 2, Hard Right (old 4) -> 3.
def remap_class(x):
    if x == 0:
        return 0
    elif x == 1:
        return 1
    elif x == 3:
        return 2
    elif x == 4:
        return 3

df['Steering_Classification'] = df['Steering_Classification'].apply(remap_class)
print("Remapped class distribution (4 classes):", collections.Counter(df['Steering_Classification']))

# Select EEG and FFT columns.
eeg_cols = [col for col in df.columns if col.lower().startswith('exg channel')]
fft_cols = [
    'Alpha_FFT_Left', 'Alpha_FFT_Right',
    'Beta_FFT_Left', 'Beta_FFT_Right',
    'Gamma_FFT_Left', 'Gamma_FFT_Right'
]
for col in fft_cols:
    if col not in df.columns:
        raise ValueError(f"{col} not found in cleaned_data.csv. Please ensure the FFT computation was successful.")

# Use only EEG and FFT columns as features.
feature_cols = eeg_cols + fft_cols

# Prepare feature and target arrays.
features = df[feature_cols].values
targets = df['Steering_Classification'].values

# Define lookback (sequence length).
lookback = 20  # Number of past time steps to use

# Create sequences: each input sequence is (lookback, num_features) and target is the next steering classification.
X_seq, y_seq = [], []
for i in range(len(features) - lookback):
    X_seq.append(features[i:i+lookback])
    y_seq.append(targets[i+lookback])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
print("Sequence shape:", X_seq.shape)

# Split into train and test sets using stratification.
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

# Standardize the features.
num_train_samples, seq_len, num_features = X_train.shape
X_train_reshaped = X_train.reshape(-1, num_features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_train = X_train_scaled.reshape(num_train_samples, seq_len, num_features)

num_test_samples = X_test.shape[0]
X_test_reshaped = X_test.reshape(-1, num_features)
X_test_scaled = scaler.transform(X_test_reshaped)
X_test = X_test_scaled.reshape(num_test_samples, seq_len, num_features)

# Build the RNN model with an initial Conv1D and LayerNormalization.
model = Sequential([
    # Convolution layer to capture local temporal patterns.
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(seq_len, num_features)),
    LayerNormalization(),
    Dropout(0.1),
    # LSTM layers.
    LSTM(64, return_sequences=True),
    Dropout(0.1),
    LSTM(128),
    Dropout(0.1),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(4, activation='softmax')  # 4 classes now.
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Setup callbacks.
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

# Compute class weights.
unique_classes, counts = np.unique(y_train, return_counts=True)
total_samples = len(y_train)
class_weight = {cls: total_samples / count for cls, count in zip(unique_classes, counts)}
print("Class weights:", class_weight)

# Train the RNN.
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=40,  # Increase epochs as needed.
    batch_size=64,
    callbacks=callbacks,
    class_weight=class_weight
)

# Evaluate the model.
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# Optionally, plot the confusion matrix.
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

conf_matrix = confusion_matrix(y_test, predicted_classes)
display_labels = ['Hard Left', 'Slight Left', 'Slight Right', 'Hard Right']
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=display_labels, yticklabels=display_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save the trained model and scaler.
model.save('rnn_model.h5')
joblib.dump(scaler, 'scaler.save')
print("Model and scaler saved.")
