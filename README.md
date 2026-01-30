# NutriBin — Machine Learning

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Project Status](https://img.shields.io/badge/status-Experimental-orange.svg)](#)

An opinionated repository for object detection and classification experiments used by the NutriBin project — includes YOLO training & detection pipelines, TensorFlow/TFLite model artifacts, helper scripts, and a minimal web demo.

--

## Features

- **YOLO object detection**: training scripts, configs, and weights in the `yolo/` folder.
- **Image classification (TFLite-ready)**: training scripts and exported models in `tflite/`.
- **Web demo**: a small Flask app to run inference and show detections in `web/`.
- **Convenience scripts**: training and test wrappers under `scripts/` and `yolo/scripts/`.

## Repo Layout

- **`tflite/`**: datasets, training outputs, and TFLite-ready assets. See [tflite/](tflite/)
- **`yolo/`**: YOLO datasets, labels, outputs, and training scripts. See [yolo/](yolo/)
- **`web/`**: Flask app (`app.py`), static JS and HTML demo. See [web/](web/)
- **`scripts/`**: convenience utilities for dataset creation and testing. See [scripts/](scripts/)

## Quick Start

Prerequisites:

- **Python 3.8+**
- Recommended: create a virtual environment and install dependencies with `pip`.

Install basics (example):

```bash
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1   # PowerShell on Windows
pip install -r requirements.txt  # create this file if you don't have it
```

If you don't have a `requirements.txt`, install the main libs commonly used here:

```bash
pip install flask torch torchvision ultralytics opencv-python matplotlib tensorflow
```

## YOLO — Train & Test

Train YOLO (example):

```bash
python yolo/scripts/train_model.py --epochs 30
```

Run detection/test with an exported model (example):

```bash
python yolo/scripts/test_model.py \
	--model yolo/outputs/yolo_training/weights/best.pt \
	--imgsz 2000 --conf 0.25 --device auto --save
```

Outputs (weights and csv results) are written to `yolo/outputs/`.

## TFLite / Classification

Use the training script in `tflite/scripts/train_model.py` (or `tflite/scripts/create_dataset.py`) to prepare and train classification models. Trained Keras models are stored in `tflite/outputs/` (for example `best_model.keras` and `my_model.keras`). Convert to TFLite as needed for edge deployment.

## Web Demo (Inference)

Start the minimal Flask demo:

```bash
cd web
python app.py
```

Open `http://127.0.0.1:5000` to use the demo UI in `web/templates/index.html` with client-side JS in `web/static/app.js`.

## Data & Labels

- Put images under the dataset folders used by the scripts (see `yolo/data/` and `tflite/data/`).
- Label formats for YOLO should follow the standard YOLO `.txt` label files located under `yolo/data/labels/`.

## Example Commands

- Create YOLO dataset (script):

```bash
python yolo/scripts/create_dataset.py
```

- Create TFLite dataset (script):

```bash
python tflite/scripts/create_dataset.py
```

## Outputs

- Trained models and class indices are in `tflite/outputs/` and `yolo/outputs/`.
- Training summaries and results CSVs are stored alongside model weights for reproducibility.

## Contributing

- **Fork** the repo, create a feature branch, and open a PR.
- Add clear descriptions for experiments and attach seed/command lines used for training.

## Next Steps / Suggestions

- Add `requirements.txt` capturing exact pinned dependencies.
- Add CI to run linting and basic unit tests or smoke tests for the demo.
- Add a small sample dataset or link to an external sample for quick demos.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

--

If you'd like, I can add a `requirements.txt`, badges, or a compact `examples/` folder with one-click demo steps next.
