"""Live testing with camera feed using YOLO detection.

This script opens your webcam and runs real-time YOLO detection
on the live feed. Press 'q' to exit.

Usage:
  python yolo/scripts/live_test.py --model best.pt --imgsz 640 --conf 0.25
"""

from pathlib import Path
import argparse
from datetime import datetime
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / 'outputs'


def find_default_weights():
    """Find the most recent or default weights file"""
    weights_dir = OUTPUT_DIR / 'yolo_training' / 'weights'
    
    # First, look for timestamped best.pt files (newest first)
    timestamped_models = sorted(OUTPUT_DIR.glob('*_best.pt'), reverse=True)
    if timestamped_models:
        return timestamped_models[0]
    
    # Fallback to standard best.pt
    for name in ('best.pt', 'last.pt'):
        p = weights_dir / name
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Path to weights (.pt)')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='auto', help="Device: 'auto', 'cpu', or CUDA id")
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print('OpenCV is required. Install with:')
        print('\n    pip install opencv-python\n')
        return

    try:
        from ultralytics import YOLO
    except Exception:
        print('The `ultralytics` package is required. Install with:')
        print('\n    pip install ultralytics\n')
        return

    # Select device
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    device_arg = (args.device or '').lower()
    if device_arg == 'auto':
        device = '0' if cuda_available else 'cpu'
    else:
        if device_arg != 'cpu' and not cuda_available and any(ch.isdigit() for ch in device_arg):
            print("Warning: CUDA not available; overriding device to 'cpu'.")
            device = 'cpu'
        else:
            device = args.device

    # Select model
    model_path = Path(args.model) if args.model else find_default_weights()
    if model_path is None or not model_path.exists():
        print('Model weights not found. Provide --model or train a model first.')
        print('Tried:', OUTPUT_DIR / 'yolo_training' / 'weights')
        return

    print(f'Loading model from: {model_path}')
    print(f'Using device: {device}')
    print('Opening camera... Press "q" to quit or press "s" to save snapshot.')
    print('Note: If display fails, snapshots will auto-save every 30 frames.\n')
    
    # Load model
    model = YOLO(str(model_path))

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'Error: Could not open camera {args.camera}')
        return

    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f'Camera opened: {frame_width}x{frame_height} @ {fps}fps')
    
    # Create output directory for snapshots
    snapshots_dir = OUTPUT_DIR / 'live_snapshots'
    snapshots_dir.mkdir(exist_ok=True)
    
    frame_count = 0
    detection_count = 0
    display_supported = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Error: Failed to read frame')
                break

            frame_count += 1
            
            # Run inference
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=device,
                verbose=False
            )

            # Visualize results
            annotated_frame = results[0].plot()
            
            # Count detections
            if results[0].boxes is not None:
                detection_count += len(results[0].boxes)

            # Try to display frame (may not work on all systems)
            if display_supported:
                try:
                    cv2.imshow('NutriBin Live Detection', annotated_frame)
                except cv2.error as e:
                    print(f'\nWarning: Display not supported on this system.')
                    print('Frames will be saved as snapshots instead.')
                    print('Press Ctrl+C to exit.\n')
                    display_supported = False
            
            # Auto-save snapshots every 30 frames if display is not supported
            if not display_supported and frame_count % 30 == 0:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
                snapshot_path = snapshots_dir / f'snapshot_{timestamp}.jpg'
                cv2.imwrite(str(snapshot_path), annotated_frame)
                print(f'Frame {frame_count}: {len(results[0].boxes)} detections')

            # Key handling (only if display is working)
            if display_supported:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print('Exiting...')
                    break
                elif key == ord('s'):
                    # Save snapshot
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    snapshot_path = snapshots_dir / f'snapshot_{timestamp}.jpg'
                    cv2.imwrite(str(snapshot_path), annotated_frame)
                    print(f'Snapshot saved: {snapshot_path}')

    except KeyboardInterrupt:
        print('\nInterrupted by user')
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        print(f'\n--- Session Summary ---')
        print(f'Frames processed: {frame_count}')
        print(f'Total detections: {detection_count}')
        if frame_count > 0:
            print(f'Average detections per frame: {detection_count / frame_count:.2f}')
        print(f'Snapshots saved to: {snapshots_dir}')
        print('Done!')


if __name__ == '__main__':
    main()
