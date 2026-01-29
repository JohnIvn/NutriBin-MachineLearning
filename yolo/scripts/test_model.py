"""Evaluate or run inference with a YOLO model.

By default the script looks for a trained model in `yolo/outputs/`
created by `train_model.py` (it will try common locations). You can
also pass `--model path/to/model.pt` to evaluate a specific file.

Requires `ultralytics` package to be installed.
"""

from pathlib import Path
import argparse


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / 'data' / 'data.yaml'
OUTPUT_DIR = ROOT / 'outputs'


def find_model():
    # Common ultralytics output locations
    candidates = [
        OUTPUT_DIR / 'weights' / 'best.pt',
        OUTPUT_DIR / 'best.pt',
        OUTPUT_DIR / 'yolo_training' / 'weights' / 'best.pt',
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception:
        print('Install ultralytics: pip install ultralytics')
        return

    model_path = args.model or find_model()
    if model_path is None:
        print('No model found in outputs. Provide --model path/to/model.pt')
        return

    model = YOLO(model_path)
    if DATA_YAML.exists():
        print('Running validation...')
        res = model.val(data=str(DATA_YAML))
        print(res)
    else:
        print('Running inference on sample images in data/images/test (if any)')
        imgs = list((ROOT / 'data' / 'images' / 'test').glob('*'))
        if not imgs:
            print('No test images found')
            return
        out = model.predict(source=[str(p) for p in imgs], save=True)
        print('Saved predictions; see `runs/predict` by default')


if __name__ == '__main__':
    main()

