
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# For object detection metrics (IoU, mAP), you need bounding box predictions and ground truth.
# Here, we provide a placeholder for object detection metrics.
def compute_iou(boxA, boxB):
    # boxA and boxB: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Placeholder for mAP calculation
def mean_average_precision(pred_boxes, gt_boxes, iou_threshold=0.5):
    # pred_boxes, gt_boxes: list of [image_id, class, confidence, x1, y1, x2, y2]
    # This is a placeholder. Use a library like pycocotools for real mAP.
    return 0.0

# Load model and class indices
import os
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs'))
best_path = os.path.join(OUTPUT_DIR, 'best_model.keras')
fallback_path = os.path.join(OUTPUT_DIR, 'my_model.keras')
model_path = best_path if os.path.exists(best_path) else fallback_path
if not os.path.exists(model_path):
    raise FileNotFoundError(f'No saved model found. Checked: {best_path} and {fallback_path}')
print(f'Loading model: {model_path}')
model = tf.keras.models.load_model(model_path)
class_indices_path = os.path.join(OUTPUT_DIR, 'class_indices.json')
if os.path.exists(class_indices_path):
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
    # invert mapping: index -> class_name
    class_names = {int(v): k for k, v in class_indices.items()}
else:
    class_names = None

def predict(img_path):
    img = image.load_img(img_path, target_size=(180, 180))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    pred_class = np.argmax(preds, axis=1)[0]
    if class_names and pred_class in class_names:
        return class_names[pred_class]
    return int(pred_class)

# Test on a folder of images and print metrics
def run_evaluation_for_model(test_dir, model_path, label):
    global model, class_names
    print(f"\nEvaluating model '{label}' from {model_path}")
    model = tf.keras.models.load_model(model_path)
    # load class names if available
    class_indices_path = os.path.join(OUTPUT_DIR, 'class_indices.json')
    if os.path.exists(class_indices_path):
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
            class_names = {int(v): k for k, v in class_indices.items()}
    else:
        class_names = None

    from glob import glob
    y_true, y_pred = [], []
    img_paths = []
    found_class_folders = False
    # First, check for class subfolders
    for class_name in os.listdir(test_dir):
        class_folder = os.path.join(test_dir, class_name)
        if os.path.isdir(class_folder):
            found_class_folders = True
            for img_file in glob(os.path.join(class_folder, '*')):
                img_paths.append(img_file)
                y_true.append(class_name)
                y_pred.append(predict(img_file))
    # If no class subfolders, treat all images in test_dir as test images (no ground truth)
    if not found_class_folders:
        for img_file in glob(os.path.join(test_dir, '*')):
            if os.path.isfile(img_file) and img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                img_paths.append(img_file)
                pred = predict(img_file)
                y_pred.append(pred)
                print(f"{os.path.basename(img_file)}: Predicted class: {pred}")
        if not img_paths:
            print("No test images found in the test directory.")
        else:
            print(f"Tested {len(img_paths)} images. (No ground truth labels available for accuracy/precision)")
    else:
        if y_true:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall: {rec:.4f}")
            print(f"F1 Score: {f1:.4f}")
        else:
            print("No test images found in the test directory.")


if __name__ == '__main__':
    import sys
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/test'))
    if len(sys.argv) >= 2 and os.path.isdir(sys.argv[1]):
        test_dir = sys.argv[1]
    print(f"Testing on directory: {test_dir}")
    if not os.path.isdir(test_dir):
        print('Test directory not found:', test_dir)
        sys.exit(1)
    # determine available models
    candidates = []
    if os.path.exists(os.path.join(OUTPUT_DIR, 'best_model.keras')):
        candidates.append(('best_model', os.path.join(OUTPUT_DIR, 'best_model.keras')))
    if os.path.exists(os.path.join(OUTPUT_DIR, 'my_model.keras')):
        candidates.append(('final_model', os.path.join(OUTPUT_DIR, 'my_model.keras')))
    if not candidates:
        raise FileNotFoundError('No saved models found to evaluate')
    for label, path in candidates:
        run_evaluation_for_model(test_dir, path, label)
