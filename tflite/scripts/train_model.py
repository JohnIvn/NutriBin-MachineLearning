import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
import os
import sys

# Configurable paths
DEFAULT_DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/images'))
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs'))
os.makedirs(output_dir, exist_ok=True)

img_height, img_width = 180, 180
batch_size = 32

dataset_path = DEFAULT_DATASET_PATH
if len(sys.argv) > 1:
    dataset_path = sys.argv[1]

if not os.path.isdir(dataset_path):
    print(f"ERROR: Dataset directory not found: {dataset_path}\nPlease make sure your images are in this folder, organized by class.")
    sys.exit(1)

print(f"Using dataset path: {dataset_path}")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

num_classes = len(train_ds.class_names)
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.save(os.path.join(output_dir, 'my_model.keras'))
print(f"Model saved to {os.path.join(output_dir, 'my_model.keras')}")

# --- Training summary and metrics ---

import datetime
from sklearn.metrics import precision_score

model_name = 'custom_cnn'
model_version = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

# Get last epoch validation accuracy and precision
val_acc = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else None
val_preds = []
val_labels = []
for batch in val_ds:
    x, y = batch
    preds = model.predict(x)
    val_preds.extend(np.argmax(preds, axis=1))
    val_labels.extend(y.numpy())
if val_labels:
    val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
else:
    val_precision = None

print("\n--- Training Summary ---")
print(f"Model: {model_name}")
print(f"Model version: {model_version}")
if val_acc is not None:
    print(f"Last training performance (validation set) - Accuracy: {val_acc:.4f}")
if val_precision is not None:
    print(f"Last training performance (validation set) - Precision score: {val_precision:.4f}")
print("Inferencing time: N/A")
print("Flash usage: N/A")

# Save summary to file
summary = {
    'model': model_name,
    'model_version': model_version,
    'val_accuracy': float(val_acc) if val_acc is not None else None,
    'val_precision': float(val_precision) if val_precision is not None else None,
    'inferencing_time': None,
    'flash_usage': None
}
import json
with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
