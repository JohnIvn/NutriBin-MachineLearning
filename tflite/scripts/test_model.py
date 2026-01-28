
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json

# Load model and class indices
import os
output_dir = '../outputs'
model = tf.keras.models.load_model(os.path.join(output_dir, 'my_model.keras'))
with open(os.path.join(output_dir, 'class_indices.json'), 'r') as f:
    class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}

def predict(img_path):
    img = image.load_img(img_path, target_size=(180, 180))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    pred_class = np.argmax(preds, axis=1)[0]
    return class_names[pred_class]

# Example usage
if __name__ == '__main__':
    import sys
    default_img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/test/test_image_2.jpg'))
    img_path = default_img_path
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(f'Image file not found: {img_path}\nPlease provide a valid image path as an argument or place a test image at ../data/test/test_image.jpg')
        sys.exit(1)
    print('Predicted class:', predict(img_path))
