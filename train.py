import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, Input, Bidirectional, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------ Set Global Policy to FP16 ------------------------ #
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float16')

# ------------------------ GPU Memory Configuration ------------------------ #
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,7"  # Use specified GPUs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'         # Suppress unnecessary TensorFlow logs

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth instead of setting a fixed memory limit
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.config.optimizer.set_jit(True)

# ------------------------ Model Save Path ------------------------ #
model_save_path = 'models/model_v0.5/'
os.makedirs(model_save_path, exist_ok=True)
file_path = 'data/Atit(all)/'

# ------------------------ Data Loading & Preprocessing ------------------------ #
df = pd.read_csv(f"{file_path}final_data.csv")
def classify_turn(x):
    return 0 if x < -500 else 1 if x > 500 else -1
df['Steering_Classification'] = df['Steering'].apply(classify_turn)
df = df[df['Steering_Classification'] != -1]

# Feature Selection
eeg_cols = [col for col in df.columns if col.lower().startswith('exg channel')]
fft_cols = ['Alpha_FFT_Left', 'Alpha_FFT_Right', 'Beta_FFT_Left', 'Beta_FFT_Right', 'Gamma_FFT_Left', 'Gamma_FFT_Right']
feature_cols = eeg_cols + fft_cols
features = df[feature_cols].astype(np.float32).values

lookback = 50

# Sequence Creation
X_seq, y_seq = [], []
for i in range(len(features) - lookback):
    X_seq.append(features[i:i+lookback])
    y_seq.append(df['Steering_Classification'].values[i+lookback])
X_seq = np.array([x for x in X_seq if x.shape[0] == lookback], dtype=np.float32)
y_seq = np.array(y_seq[:len(X_seq)])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)
scaler = StandardScaler()

# Scale the features (keep as float32 for scaling, then cast to float16)
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_train = X_train_scaled.reshape(X_train.shape).astype(np.float16)

X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
X_test_scaled = scaler.transform(X_test_reshaped)
X_test = X_test_scaled.reshape(X_test.shape).astype(np.float16)

# ------------------------ Create TensorFlow Dataset ------------------------ #
def create_tf_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.cache()  # Cache dataset for faster epochs
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

batch_size = 8  # Increased batch size for better GPU utilization
train_dataset = create_tf_dataset(X_train, y_train, batch_size)
test_dataset = create_tf_dataset(X_test, y_test, batch_size)

# ------------------------ Model Building ------------------------ #
model = Sequential([
    # Input layer
    Input(shape=(lookback, X_train.shape[-1]), dtype='float16'),

    # Convolutional feature extraction block
    Conv1D(32, kernel_size=3, activation='relu', padding='same', dtype='float16'),
    BatchNormalization(dtype='float16'),
    Conv1D(32, kernel_size=3, activation='relu', padding='same', dtype='float16'),
    BatchNormalization(dtype='float16'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # Temporal modeling with Bidirectional LSTMs
    Bidirectional(LSTM(128, return_sequences=True, dtype='float16')),
    Dropout(0.2),
    Bidirectional(LSTM(64, dtype='float16')),
    Dropout(0.2),

    # Fully connected classification block
    Dense(64, activation='relu', dtype='float16'),
    BatchNormalization(dtype='float16'),
    Dropout(0.2),
    Dense(32, activation='relu', dtype='float16'),
    BatchNormalization(dtype='float16'),
    Dropout(0.2),
    
    # Output layer
    Dense(2, activation='softmax', dtype='float16')
])
# Compile with jit_compile enabled
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              jit_compile=True)

# ------------------------ Training ------------------------ #
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]
model.fit(train_dataset, validation_data=test_dataset, epochs=10, callbacks=callbacks)

# ------------------------ Evaluation & Saving ------------------------ #
test_acc = model.evaluate(test_dataset)[1]
print("Test Accuracy:", test_acc)
model.save(os.path.join(model_save_path, 'optimized_rnn_model.h5'))
joblib.dump(scaler, os.path.join(model_save_path, 'scaler.save'))
