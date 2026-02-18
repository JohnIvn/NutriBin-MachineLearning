import time
import uuid
import base64
import argparse
from pathlib import Path

import cv2
import socketio
import numpy as np
from PIL import Image
from pathlib import Path as _Path


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument('--server', default='https://nutribin-feed.up.railway.app', help='Socket.IO server URL')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--model', default=None, help='Path to YOLO weights (optional)')
    p.add_argument('--classifier-model', default=None, help='Path to image classifier folder (optional)')
    p.add_argument('--class-interval', type=int, default=3, help='Run classifier every N frames')
    p.add_argument('--quality', type=int, default=80, help='JPEG quality 0-100')
    return p.parse_args()


def load_model(model_path: Path | None):
    try:
        from ultralytics import YOLO
    except Exception:
        print('ultralytics not installed; please pip install ultralytics')
        return None

    if model_path:
        if not Path(model_path).exists():
            print('Model path not found:', model_path)
            return None
        return YOLO(str(model_path))

    # try to find a model in repo outputs
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / 'yolo' / 'outputs' / 'yolo_training' / 'weights' / 'best.pt'
    if candidate.exists():
        return YOLO(str(candidate))
    print('No model found; continuing without model (will stream raw frames)')
    return None


def load_classifier(model_path: _Path | None):
    if model_path is None:
        return None
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except Exception:
        print('transformers/torch not installed; classifier disabled')
        return None

    if not model_path.exists():
        print('Classifier path not found:', model_path)
        return None

    # load processor and model
    try:
        try:
            processor = AutoImageProcessor.from_pretrained(str(model_path))
        except Exception:
            from transformers import AutoFeatureExtractor

            processor = AutoFeatureExtractor.from_pretrained(str(model_path))
        model = AutoModelForImageClassification.from_pretrained(str(model_path))
    except Exception as e:
        print('Failed to load classifier:', e)
        return None

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # try to obtain id2label mapping
    id2label = {}
    if getattr(model.config, 'id2label', None):
        id2label = {int(k): v for k, v in model.config.id2label.items()} if isinstance(model.config.id2label, dict) else model.config.id2label
    # fallback config.json
    if not id2label:
        cfg = model_path / 'config.json'
        try:
            import json

            if cfg.exists():
                j = json.loads(cfg.read_text())
                if isinstance(j, dict):
                    if 'id2label' in j:
                        id2label = {int(k): v for k, v in j['id2label'].items()} if isinstance(j['id2label'], dict) else j['id2label']
                elif isinstance(j, list):
                    id2label = {i: label for i, label in enumerate(j)}
        except Exception:
            pass

    return {'processor': processor, 'model': model, 'device': device, 'id2label': id2label}


def main():
    args = build_args()

    # Allow automatic reconnection and fall back to polling if websockets fail
    sio = socketio.Client(logger=False, reconnection=True, reconnection_attempts=10, reconnection_delay=1)

    @sio.event
    def connect():
        print('Connected to Feed server')

    @sio.event
    def disconnect():
        print('Disconnected from Feed server')

    @sio.event
    def connect_error(data):
        print('Connection failed / error:', data)

    # Connect with an auth flag so server can distinguish producers (optional)
    try:
        print('Connecting to', args.server)
        sio.connect(args.server, auth={'role': 'producer'})
    except Exception as e:
        print('Failed to connect to server:', e)
        return

    model = load_model(Path(args.model) if args.model else None)
    classifier = load_classifier(_Path(args.classifier_model) if args.classifier_model else None)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print('Failed to open camera', args.camera)
        return

    frame_count = 0
    try:
        while sio.connected:
            ret, frame = cap.read()
            if not ret:
                break

            # Optionally resize for faster inference / lower bandwidth
            h, w = frame.shape[:2]
            if max(h, w) > args.imgsz:
                scale = args.imgsz / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            annotated_b64 = None
            preds = None
            stats = None

            if model is not None:
                try:
                    results = model.predict(source=frame, imgsz=args.imgsz, conf=0.1, verbose=False)
                    r = results[0]
                    preds = []
                    if getattr(r, 'boxes', None) is not None and hasattr(r.boxes, 'xyxy'):
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else None
                        cls = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, 'cls') else None
                        for i in range(len(xyxy)):
                            preds.append({'xyxy': xyxy[i].tolist(), 'confidence': float(confs[i]) if confs is not None else None, 'class': int(cls[i]) if cls is not None else None})

                    # annotated frame
                    try:
                        annotated = r.plot()
                        success, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
                        if success:
                            annotated_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
                    except Exception:
                        annotated_b64 = None

                    stats = {'num_detections': len(preds) if preds is not None else 0}
                except Exception as e:
                    print('Inference error:', e)

            # If model not available or annotation failed, send raw frame
            if annotated_b64 is None:
                success, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
                if not success:
                    continue
                annotated_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

            # Optionally run image classifier every N frames
            classifications = None
            try:
                frame_count += 1
                if classifier and (frame_count % max(1, args.class_interval) == 0):
                    import torch
                    import torch.nn.functional as F

                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    proc = classifier['processor']
                    model_cls = classifier['model']
                    device = classifier['device']
                    inputs = proc(images=img, return_tensors='pt')
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model_cls(**inputs)
                        logits = outputs.logits
                        probs = F.softmax(logits, dim=-1)
                        # full distribution
                        full_probs = probs[0].cpu().tolist()
                        id2label = classifier.get('id2label') or {}
                        # build full label -> percentage mapping
                        classification_percentages = {}
                        for idx, p in enumerate(full_probs):
                            label = id2label.get(idx, str(idx))
                            classification_percentages[label] = round(float(p) * 100.0, 2)

                        # top-k summary (kept for compatibility)
                        topk = min(3, len(full_probs))
                        topk_probs, topk_inds = torch.topk(probs, k=topk, dim=-1)
                        topk_probs = topk_probs[0].cpu().tolist()
                        topk_inds = topk_inds[0].cpu().tolist()
                        classifications = []
                        for prob, idx in zip(topk_probs, topk_inds):
                            label = id2label.get(idx, str(idx))
                            classifications.append({'label': label, 'score': float(prob)})
                        # print percentages for user visibility
                        try:
                            print('Classification percentages:')
                            for lbl, pct in classification_percentages.items():
                                print(f"  {lbl}: {pct}%")
                        except Exception:
                            pass
            except Exception as e:
                print('Classifier error:', e)

            payload = {
                'id': str(uuid.uuid4()),
                'frame': annotated_b64,
                'predictions': preds,
                'stats': stats,
                'classifications': classifications,
                'classification_percentages': classification_percentages if 'classification_percentages' in locals() else None,
                'ts': time.time(),
            }

            try:
                sio.emit('video-frame', payload)
            except Exception as e:
                print('Emit failed:', e)
                break

            # ~15 FPS
            time.sleep(0.06)

    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        cap.release()
        if sio.connected:
            sio.disconnect()


if __name__ == '__main__':
    main()
