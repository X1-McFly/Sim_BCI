import os
# Force TensorFlow to use CPU only.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
import collections
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model_save_path = 'models/model_v0.5/'
os.makedirs(model_save_path, exist_ok=True)
file_path = 'data/Atit(all)/'

# ------------------------ Data Loading & Preprocessing ------------------------ #
df = pd.read_csv(f"{file_path}final_data.csv")
print("Columns in the dataset:", df.columns.tolist())

# Define classification: left if steering < -500, right if steering > 500, centered otherwise.
def classify_turn(x):
    if x < -500:
        return 0  # Left
    elif x > 500:
        return 1  # Right
    else:
        return -1  # Centered (to be removed)

df['Steering_Classification'] = df['Steering'].apply(classify_turn)
print("Original class distribution (including centered):", collections.Counter(df['Steering_Classification']))

# Remove centered samples (where classification == -1).
df = df[df['Steering_Classification'] != -1]
print("After removing centered samples:", collections.Counter(df['Steering_Classification']))

# ------------------------ Feature Selection ------------------------ #
# Select EEG and FFT columns.
eeg_cols = [col for col in df.columns if col.lower().startswith('exg channel')]
fft_cols = [
    'Alpha_FFT_Left', 'Alpha_FFT_Right',
    'Beta_FFT_Left', 'Beta_FFT_Right',
    'Gamma_FFT_Left', 'Gamma_FFT_Right'
]
for col in fft_cols:
    if col not in df.columns:
        raise ValueError(f"{col} not found in final_data.csv. Please ensure the FFT computation was successful.")

feature_cols = eeg_cols + fft_cols
features = df[feature_cols].values
targets = df['Steering_Classification'].values

# ------------------------ Sequence Creation ------------------------ #
lookback = 200  # Number of past time steps to use

X_seq, y_seq = [], []
for i in range(len(features) - lookback):
    X_seq.append(features[i:i+lookback])
    y_seq.append(targets[i+lookback])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
print("Sequence shape:", X_seq.shape)

# ------------------------ Train-Test Split ------------------------ #
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

# ------------------------ Feature Scaling ------------------------ #
num_train_samples, seq_len, num_features = X_train.shape
X_train_reshaped = X_train.reshape(-1, num_features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_train = X_train_scaled.reshape(num_train_samples, seq_len, num_features)

num_test_samples = X_test.shape[0]
X_test_reshaped = X_test.reshape(-1, num_features)
X_test_scaled = scaler.transform(X_test_reshaped)
X_test = X_test_scaled.reshape(num_test_samples, seq_len, num_features)

# ------------------------ Model Building ------------------------ #
model = Sequential([
    # Convolution block.
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(seq_len, num_features)),
    LayerNormalization(),
    Dropout(0.1),
    # Recurrent layers.
    LSTM(128, return_sequences=True),
    Dropout(0.1),
    LSTM(128),
    Dropout(0.1),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(2, activation='softmax')  # 2 classes: 0=Left, 1=Right.
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------ Callbacks & Class Weights ------------------------ #
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

unique_classes, counts = np.unique(y_train, return_counts=True)
total_samples = len(y_train)
class_weight = {cls: total_samples / count for cls, count in zip(unique_classes, counts)}
print("Class weights:", class_weight)

# ------------------------ Model Training ------------------------ #
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weight
)

# ------------------------ Model Evaluation ------------------------ #
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# ------------------------ Confusion Matrix Plot & Save ------------------------ #
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
conf_matrix = confusion_matrix(y_test, predicted_classes)
display_labels = ['Left', 'Right']

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=display_labels, yticklabels=display_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Binary Classification: Left vs Right)")

confusion_matrix_file = os.path.join(model_save_path, "confusion_matrix.png")
plt.savefig(confusion_matrix_file)
print(f"Confusion matrix saved to {confusion_matrix_file}")
plt.show()

# ------------------------ Save Model & Scaler ------------------------ #
model.save(os.path.join(model_save_path, 'rnn_model.h5'))
joblib.dump(scaler, os.path.join(model_save_path, 'scaler.save'))
print("Model and scaler saved.")
