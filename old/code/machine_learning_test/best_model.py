import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization

# Load data
all_data = pd.read_csv('eeg_data_with_window_5_derivatives.csv')

# Select features and target
x = all_data[["Fpz", "F3", "F4", "C3", "C4", "P3", "Pz", "P4", "O1", "O2", "Fpz_derivative", "F3_derivative", "F4_derivative", "C3_derivative", "C4_derivative", "P3_derivative", "Pz_derivative", "P4_derivative", "O1_derivative", "O2_derivative"]].fillna(0)
y = all_data["Annotations"].fillna(0)

y = y.astype(np.int32)
y = to_categorical(y, num_classes=3)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Build model
model = Sequential()
model.add(Dense(x.shape[1], activation=LeakyReLU(), input_shape=(x_train.shape[1],)))
model.add(Dense(256, activation=LeakyReLU()))
model.add(BatchNormalization())
model.add(Dropout(0.15))
model.add(Dense(256, activation=LeakyReLU()))
model.add(BatchNormalization())
model.add(Dropout(0.15))
model.add(Dense(256, activation=LeakyReLU()))
model.add(BatchNormalization())
model.add(Dropout(0.15))
model.add(Dense(3, activation='softmax'))

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Compile model
model.compile(optimizer=AdamW(learning_rate=0.005, weight_decay=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    epochs=200,
                    batch_size=64,
                    callbacks=[reduce_lr, early_stopping])

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Validation Loss: {loss:.4f}')
print(f'Validation Accuracy: {accuracy:.4f}')