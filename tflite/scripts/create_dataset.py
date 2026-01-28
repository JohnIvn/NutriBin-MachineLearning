import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

# Set up paths
data_dir = '../data/images'  # New dataset path relative to scripts
output_dir = '../outputs'
os.makedirs(output_dir, exist_ok=True)
img_height, img_width = 180, 180
batch_size = 32

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Save class indices for deployment
with open(os.path.join(output_dir, 'class_indices.json'), 'w') as f:
    json.dump(train_generator.class_indices, f)
