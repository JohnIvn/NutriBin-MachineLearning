"""Train a YOLO model using the Ultralytics package.

This script expects `yolo/data/data.yaml` created by
`create_dataset.py`. It will attempt to import `ultralytics` and
train a small `yolov8n` model. If `ultralytics` is not installed,
the script prints installation instructions.

Usage:
  python yolo/scripts/train_model.py --epochs 30
"""

from pathlib import Path
import argparse


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / 'data' / 'data.yaml'
OUTPUT_DIR = ROOT / 'outputs'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()

    if not DATA_YAML.exists():
        print('Data YAML not found at', DATA_YAML)
        print('Run yolo/scripts/create_dataset.py first')
        return

    try:
        from ultralytics import YOLO
    except Exception:
        print('The `ultralytics` package is required. Install with:')
        print('\n    pip install ultralytics\n')
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    # Start from a small pretrained backbone; Ultralytics will download if needed
    model = YOLO('yolov8n.pt')
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(OUTPUT_DIR),
        name='yolo_training',
        exist_ok=True,
    )


if __name__ == '__main__':
    main()

