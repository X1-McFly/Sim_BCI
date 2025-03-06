import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
all_data = pd.read_csv('eeg_data_with_window_5_derivatives.csv')

# Select features and target
x = all_data[["Fpz", "F3", "F4", "C3", "C4", "P3", "Pz", "P4", 
              "O1", "O2", "Fpz_derivative", "F3_derivative", "F4_derivative", 
              "C3_derivative", "C4_derivative", "P3_derivative", 
              "Pz_derivative", "P4_derivative", "O1_derivative", "O2_derivative"]].fillna(0)

y = all_data["Annotations"].fillna(0).astype(np.int32)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

# Convert target to categorical
y_train_smote = to_categorical(y_train_smote, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Standardize features
scaler = StandardScaler()
x_train_smote = scaler.fit_transform(x_train_smote)
x_test = scaler.transform(x_test)

# Define custom focal loss
def focal_loss(gamma=2., alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1 - 1e-8)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return loss_fn

# Define tuned Cyclic Learning Rate (CLR)
class CyclicLR(tf.keras.callbacks.Callback):
    def __init__(self, base_lr=1e-5, max_lr=5e-4, step_size=3000, mode='triangular2'):
        super().__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.clr_iterations = 0

    def on_train_begin(self, logs=None):
        if self.clr_iterations == 0:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)

    def on_batch_end(self, batch, logs=None):
        self.clr_iterations += 1
        lr = self.compute_lr()
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

    def compute_lr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))

        if self.mode == 'triangular2':
            lr = lr / float(2 ** (cycle - 1))

        return lr

clr = CyclicLR(base_lr=1e-5, max_lr=5e-4, step_size=3000)

# Build a simple ensemble of 3 models
def build_model():
    model = Sequential()
    model.add(Dense(256, activation=LeakyReLU(), input_shape=(x_train_smote.shape[1],), kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))  # Lower dropout

    model.add(Dense(128, activation=LeakyReLU(), kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    model.add(Dense(64, activation=LeakyReLU(), kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.05))

    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5), 
                  loss=focal_loss(gamma=2., alpha=0.25), 
                  metrics=['accuracy'])
    return model

# Train multiple models (Ensemble)
models = [build_model() for _ in range(3)]
for i, model in enumerate(models):
    print(f'Training model {i + 1}')
    model.fit(x_train_smote, y_train_smote,
              validation_data=(x_test, y_test),
              epochs=100,  # Slightly reduced epochs for each model
              batch_size=16,  # Smaller batch size for finer updates
              callbacks=[clr])

# Average predictions from all models
y_preds = np.zeros((x_test.shape[0], 3))
for model in models:
    y_preds += model.predict(x_test) / len(models)

y_pred_classes = np.argmax(y_preds, axis=1)
y_true = np.argmax(y_test, axis=1)

# Evaluate the ensemble
print(classification_report(y_true, y_pred_classes, target_names=["T0", "T1", "T2"]))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', 
            xticklabels=["T0", "T1", "T2"], yticklabels=["T0", "T1", "T2"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
