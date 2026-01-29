from flask import Flask, render_template, request, jsonify
from PIL import Image
import io, base64
import numpy as np
import time
from pathlib import Path

app = Flask(__name__, template_folder="templates", static_folder="static")

# Prefer the trained weights produced by your training script, fallback to yolov8n.pt
try:
    from ultralytics import YOLO
    # Resolve repo root relative to this file so cwd doesn't matter
    REPO_ROOT = Path(__file__).resolve().parents[1]
    # paths to try (trained best first) using absolute paths
    CANDIDATE_PATHS = [
        REPO_ROOT / 'yolo' / 'outputs' / 'yolo_training' / 'weights' / 'best.pt',
        REPO_ROOT / 'yolo' / 'outputs' / 'yolo_training' / 'weights' / 'last.pt',
        REPO_ROOT / 'yolo' / 'scripts' / 'yolov8n.pt',
    ]

    MODEL_PATH = None
    for p in CANDIDATE_PATHS:
        if p.exists():
            MODEL_PATH = str(p)
            break

    if MODEL_PATH is None:
        # leave model None and return error from endpoint
        model = None
    else:
        # load the selected model
        model = YOLO(MODEL_PATH)
except Exception:
    MODEL_PATH = None
    model = None


def _load_model_from_path(path: Path):
    """Helper to load YOLO model from a Path object and update globals."""
    global model, MODEL_PATH
    try:
        model = YOLO(str(path))
        MODEL_PATH = str(path)
        return True, MODEL_PATH
    except Exception as e:
        return False, str(e)


@app.route('/reload_model', methods=['POST', 'GET'])
def reload_model():
    """Reload model at runtime. Optional query/body param `path` to specify exact file.
    If omitted, attempts to load `yolo/outputs/yolo_training/weights/best.pt`.
    """
    global model, MODEL_PATH
    try:
        # allow path via query param or POST form
        req_path = request.args.get('path') or request.form.get('path')
        if req_path:
            p = Path(req_path)
            if not p.is_absolute():
                p = REPO_ROOT / req_path
        else:
            p = REPO_ROOT / 'yolo' / 'outputs' / 'yolo_training' / 'weights' / 'best.pt'

        if not p.exists():
            return jsonify({"error": "model file not found", "path": str(p)}), 404

        ok, info = _load_model_from_path(p)
        if not ok:
            return jsonify({"error": info}), 500
        return jsonify({"reloaded": True, "model_path": MODEL_PATH})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return render_template("index.html")


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/jpeg;base64,{encoded}"


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "YOLO model not available on server. Install 'ultralytics' and ensure the model file exists."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)

    # Run inference (measure time)
    try:
        t0 = time.time()
        # lower confidence threshold to capture weaker detections for this app
        results = model(img_np, conf=0.1)
        t1 = time.time()
        inference_ms = (t1 - t0) * 1000.0
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Extract predictions
    preds = []
    r = results[0]
    # Use the Boxes arrays if available (safer and faster)
    try:
        boxes = getattr(r, 'boxes', None)
        if boxes is not None and hasattr(boxes, 'xyxy'):
            xyxy_arr = boxes.xyxy.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else None
            cls_arr = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else None
            for i in range(len(xyxy_arr)):
                preds.append({
                    "xyxy": xyxy_arr[i].tolist(),
                    "confidence": float(conf_arr[i]) if conf_arr is not None else None,
                    "class": int(cls_arr[i]) if cls_arr is not None else None,
                })
    except Exception:
        preds = []

    # If no predictions, include raw box arrays for debugging
    raw_debug = None
    if len(preds) == 0:
        try:
            if hasattr(r, 'boxes') and r.boxes is not None and hasattr(r.boxes, 'xyxy'):
                raw_debug = {
                    'xyxy': r.boxes.xyxy.cpu().numpy().tolist(),
                    'conf': r.boxes.conf.cpu().numpy().tolist() if hasattr(r.boxes, 'conf') else None,
                    'cls': r.boxes.cls.cpu().numpy().tolist() if hasattr(r.boxes, 'cls') else None,
                }
            else:
                raw_debug = str(r)
        except Exception:
            raw_debug = str(r)

    # Create annotated image
    try:
        annotated = r.plot()
        annotated_pil = Image.fromarray(annotated)
        annotated_b64 = pil_to_base64(annotated_pil)
    except Exception:
        annotated_b64 = None

    # Build statistics
    try:
        num_detections = len(preds)
        confs = [p['confidence'] for p in preds if p.get('confidence') is not None]
        avg_conf = float(np.mean(confs)) if len(confs) > 0 else None
        max_conf = float(np.max(confs)) if len(confs) > 0 else None
        min_conf = float(np.min(confs)) if len(confs) > 0 else None
        classes = sorted(list({p['class'] for p in preds if p.get('class') is not None}))
        class_counts = {str(c): sum(1 for p in preds if p.get('class') == c) for c in classes}
    except Exception:
        num_detections = 0
        avg_conf = max_conf = min_conf = None
        classes = []
        class_counts = {}

    # Try to get class names from model if available
    class_names = None
    try:
        names_map = getattr(model, 'names', None) or getattr(getattr(model, 'model', None), 'names', None)
        if names_map and classes:
            class_names = {str(c): (names_map[c] if c in names_map else None) for c in classes}
    except Exception:
        class_names = None

    stats = {
        "num_detections": num_detections,
        "avg_confidence": avg_conf,
        "max_confidence": max_conf,
        "min_confidence": min_conf,
        "classes": classes,
        "class_counts": class_counts,
        "class_names": class_names,
        "inference_ms": float(inference_ms) if 'inference_ms' in locals() else None,
    }

    return jsonify({"predictions": preds, "annotated": annotated_b64, "model_path": MODEL_PATH, "stats": stats, "raw_debug": raw_debug})


@app.route('/status')
def status():
    return jsonify({
        "model_loaded": MODEL_PATH is not None and model is not None,
        "model_path": MODEL_PATH,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
