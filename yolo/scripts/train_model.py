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
from datetime import datetime
import shutil


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
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Start from a small pretrained backbone; Ultralytics will download if needed
    model = YOLO('yolov8n.pt')
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(OUTPUT_DIR),
        name=f'yolo_training_{timestamp}',
        exist_ok=False,
    )
    
    # Copy the best and last weights with timestamp
    if results:
        training_dir = OUTPUT_DIR / f'yolo_training_{timestamp}'
        weights_dir = training_dir / 'weights'
        
        if weights_dir.exists():
            best_src = weights_dir / 'best.pt'
            last_src = weights_dir / 'last.pt'
            
            best_dst = OUTPUT_DIR / f'{timestamp}_best.pt'
            last_dst = OUTPUT_DIR / f'{timestamp}_last.pt'
            
            if best_src.exists():
                shutil.copy2(best_src, best_dst)
                print(f'Saved best model: {best_dst}')
            
            if last_src.exists():
                shutil.copy2(last_src, last_dst)
                print(f'Saved last model: {last_dst}')


if __name__ == '__main__':
    main()

