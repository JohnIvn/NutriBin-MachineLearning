"""Continue training (upgrade) of an existing YOLO model using local dataset.

This script expects images arranged as class subfolders under
`yolo/data/images/<class_name>/*.jpg`. If `yolo/data/data.yaml` is missing
the script can automatically run `create_dataset.py` to build a YOLO-style
dataset (labels and `data.yaml`).

Usage examples:
  python yolo/scripts/upgrade_model.py --base-weights path/to/best.pt --epochs 10
  python yolo/scripts/upgrade_model.py --base-weights outputs/yolo_training/weights/best.pt --auto-create

The script uses the `ultralytics` package (YOLOv8). Install with:
  pip install ultralytics
"""

from pathlib import Path
import argparse
import time
import runpy


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / 'data' / 'data.yaml'
OUTPUT_DIR = ROOT / 'outputs'


def find_default_weights():
    weights_dir = OUTPUT_DIR / 'yolo_training' / 'weights'
    for name in ('best.pt', 'last.pt'):
        p = weights_dir / name
        if p.exists():
            return p
    return None


def ensure_dataset(auto_create=False):
    if DATA_YAML.exists():
        return True
    if not auto_create:
        return False
    # try to create dataset by running create_dataset.py
    script = ROOT / 'scripts' / 'create_dataset.py'
    if not script.exists():
        return False
    print('Creating YOLO dataset from class-subfolders using', script)
    try:
        runpy.run_path(str(script), run_name='__main__')
    except Exception as e:
        print('Failed to create dataset:', e)
        return False
    return DATA_YAML.exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-weights', type=str, default=None,
                        help='Path to base weights (.pt) to continue training from')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto',
                        help="Device: 'auto', 'cpu', or CUDA id like '0'")
    parser.add_argument('--auto-create', action='store_true',
                        help='If data.yaml missing, run create_dataset.py automatically')
    args = parser.parse_args()

    if not ensure_dataset(auto_create=args.auto_create):
        print('Data YAML not found at', DATA_YAML)
        print('Arrange images under yolo/data/images/<class_name> and run:')
        print('  python yolo/scripts/create_dataset.py')
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

    base = Path(args.base_weights) if args.base_weights else find_default_weights()
    if base is None or not base.exists():
        print('Base weights not found. Provide --base-weights or place a .pt under:')
        print(' -', OUTPUT_DIR / 'yolo_training' / 'weights')
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Build a unique run name
    base_name = base.stem
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    run_name = f'upgrade_{base_name}_{timestamp}'

    print('Loading model from', base)
    model = YOLO(str(base))

    print('Starting training to upgrade model...')
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(OUTPUT_DIR),
        name=run_name,
        exist_ok=True,
    )

    print('Training finished. Check outputs under', OUTPUT_DIR / run_name)


if __name__ == '__main__':
    main()
