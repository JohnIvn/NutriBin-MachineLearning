"""Run YOLO inference with a trained model on a test image directory.

This script will attempt to load weights produced by
`yolo/scripts/train_model.py` (for example
`yolo/outputs/yolo_training/weights/best.pt`) and run detection on images
found under `yolo/data/images/test`. If that folder is missing the script
falls back to `yolo/data/images/val` and then to `tflite/data/test` in the
repository root.

Usage:
  python yolo/scripts/test_model.py --model <model.pt> --imgsz 640

If `ultralytics` is not installed the script prints install instructions.
"""

from pathlib import Path
import argparse
import sys
import shutil


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'outputs'


def find_default_weights():
    weights_dir = OUTPUT_DIR / 'yolo_training' / 'weights'
    for name in ('best.pt', 'last.pt'):
        p = weights_dir / name
        if p.exists():
            return p
    return None


def find_test_dir():
    candidates = [
        ROOT / 'data' / 'images' / 'test',
        ROOT / 'data' / 'images' / 'val',
        REPO_ROOT / 'tflite' / 'data' / 'test',
    ]
    for p in candidates:
        if p.exists() and (any(p.rglob('*.jpg')) or any(p.rglob('*.png'))):
            return p
    # last-ditch: if images in images/val exist, return val
    val = ROOT / 'data' / 'images' / 'val'
    if val.exists():
        return val
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Path to weights (.pt)')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', type=str, default='auto', help="Device: 'auto', 'cpu', or CUDA id like '0'")
    parser.add_argument('--save', action='store_true', help='Save annotated images to outputs/detections')
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception:
        print('The `ultralytics` package is required. Install with:')
        print('\n    pip install ultralytics\n')
        return

    # Select device: prefer CUDA if available, otherwise CPU. If the user
    # explicitly requests a CUDA device but no CUDA is present we'll fall back
    # to CPU with a warning to avoid a hard exception from Ultralytics.
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    device_arg = (args.device or '').lower()
    if device_arg == 'auto':
        device = '0' if cuda_available else 'cpu'
    else:
        # user provided something explicit
        if device_arg != 'cpu' and not cuda_available and any(ch.isdigit() for ch in device_arg):
            print("Warning: CUDA not available; overriding device to 'cpu'.")
            device = 'cpu'
        else:
            device = args.device

    model_path = Path(args.model) if args.model else find_default_weights()
    if model_path is None or not model_path.exists():
        print('Model weights not found. Provide --model or run training first.')
        print('Tried default weights under', OUTPUT_DIR / 'yolo_training' / 'weights')
        return

    test_dir = find_test_dir()
    if test_dir is None:
        print('No test images found. Ensure images are in one of:')
        print(' -', ROOT / 'data' / 'images' / 'test')
        print(' -', ROOT / 'data' / 'images' / 'val')
        print(' -', REPO_ROOT / 'tflite' / 'data' / 'test')
        return

    # When saving detections, create a numbered folder (detection_1, detection_2, ...)
    # to avoid overwriting previous runs. If not saving, use a temporary folder.
    base_detections = OUTPUT_DIR / 'detections'
    if args.save:
        base_detections.mkdir(parents=True, exist_ok=True)
        # find next available index
        existing = [p.name for p in base_detections.iterdir() if p.is_dir() and p.name.startswith('detection_')]
        max_index = 0
        for name in existing:
            try:
                idx = int(name.split('_', 1)[1])
                if idx > max_index:
                    max_index = idx
            except Exception:
                continue
        next_idx = max_index + 1
        save_dir = base_detections / f'detection_{next_idx}'
        save_dir.mkdir(parents=True, exist_ok=False)
    else:
        # create a temporary folder to store results then clean up
        save_dir = OUTPUT_DIR / 'detections_temp'
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    print('Loading model from', model_path)
    model = YOLO(str(model_path))

    source = str(test_dir)
    print('Running inference on images in', source)

    # Use ultralytics high-level predict API. model.predict accepts a directory.
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        save=True,
        save_dir=str(save_dir),
    )

    # Summarize results
    n_images = len(results)
    total_boxes = 0
    for r in results:
        try:
            boxes = getattr(r, 'boxes', None)
            if boxes is None:
                bcount = 0
            else:
                try:
                    bcount = len(boxes)
                except Exception:
                    # fallback for older/newer API shapes
                    bcount = 0
        except Exception:
            bcount = 0
        total_boxes += bcount

    print(f'Processed {n_images} images — detected {total_boxes} boxes total')
    print('Annotated images saved to', save_dir)


if __name__ == '__main__':
    main()
