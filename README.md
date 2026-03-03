# NutriBin — Machine Learning

<img width="936" height="328" alt="image" src="https://github.com/user-attachments/assets/6c962171-3add-41db-a3ba-0d2597b2c2d6" />

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Project Status](https://img.shields.io/badge/status-Experimental-orange.svg)](#)

An opinionated repository for object detection and classification experiments used by the NutriBin project — includes YOLO training
& detection pipelines, TensorFlow/TFLite model artifacts, helper scripts, and a minimal web demo.

**NEW**: Now includes **Drowsiness Detection System** with ESP32 integration for real-time alerting! 🚨

--

## Features

- **YOLO object detection**: training scripts, configs, and weights in the `yolo/` folder.
- **Image classification (TFLite-ready)**: training scripts and exported models in `tflite/`.
- **Web demo**: a small Flask app to run inference and show detections in `web/`.
- **Convenience scripts**: training and test wrappers under `scripts/` and `yolo/scripts/`.
- **🆕 Drowsiness Detection with ESP32**: Real-time drowsiness monitoring with physical alerts (buzzers + vibrators) via WiFi integration.

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

## 🚨 Drowsiness Detection System (ESP32 Integration)

This system uses a **Raspberry Pi** running YOLO-based drowsiness detection to monitor driver alertness in real-time. When drowsiness is detected, it sends WiFi commands to an **ESP32 microcontroller** that activates buzzers and vibrator motors to alert the driver.

### System Components

- **Raspberry Pi**: Runs ML model, processes camera feed
- **ESP32**: WiFi AP, controls buzzers (2x) & vibrators (6x)
- **Camera**: USB/CSI camera on Raspberry Pi
- **Alerts**: Progressive intensity based on drowsiness level (0-100%)

### Quick Start (Drowsiness Detection)

```bash
# 1. Connect Raspberry Pi to ESP32 WiFi
sudo ./scripts/connect_esp32_wifi.sh
# SSID: ESP32-Drowsiness-AP | Password: drowsy123

# 2. Quick setup (installs dependencies, tests connection)
./scripts/quick_start.sh

# 3. Run drowsiness detection
python scripts/drowsiness_detection_esp32.py

# Or with display
python scripts/drowsiness_detection_esp32.py --display
```

### Drowsiness Levels

| Level | Classification | Alert Intensity |
|-------|----------------|-----------------|
| 0 | ALERT  FULLY AWAKE | No alert (0%) |
| 1 | EARLY DROWSINESS | Vibrator (20%) |
| 2 | MODERATE DROWSINESS | Buzzer + Vibrator (50%) |
| 3 | MICROSLEEP | Buzzer + Vibrator (80%) |
| 4-5 | REM SLEEP / STAGE N1-N3 | Maximum (100%) |

### Full Documentation

See **[DROWSINESS_SETUP.md](DROWSINESS_SETUP.md)** for complete setup, configuration, troubleshooting, and API reference.

---

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

Upgrade (continue training) an existing YOLO `.pt` from another user:

```bash
# continue training from a shared weights file
python yolo/scripts/upgrade_model.py \
  --base-weights yolo/outputs/yolo_training/weights/best.pt \
  --epochs 10 --imgsz 640 --batch 8

# auto-create YOLO dataset from `yolo/data/images/<class_name>/` then upgrade
python yolo/scripts/upgrade_model.py --base-weights path/to/shared.pt --auto-create
```

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
