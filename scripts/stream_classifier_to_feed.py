import time
import uuid
import base64
import argparse
from pathlib import Path

import cv2
import socketio
import numpy as np


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument('--server', default='https://nutribin-feed.up.railway.app', help='Socket.IO server URL')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--model', default=None, help='Path to YOLO weights (optional)')
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

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print('Failed to open camera', args.camera)
        return

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

            payload = {
                'id': str(uuid.uuid4()),
                'frame': annotated_b64,
                'predictions': preds,
                'stats': stats,
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
