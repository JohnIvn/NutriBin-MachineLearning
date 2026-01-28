# Computer Vision with TensorFlow

This folder contains code for a simple image classification pipeline using TensorFlow.

## Structure

- `create_dataset.py`: Prepares the dataset using Keras ImageDataGenerator.
- `train_model.py`: Defines and trains a CNN model.
- `deploy_model.py`: Loads the trained model and predicts the class of a new image.

## Instructions

1. **Prepare your dataset**
   - Place your images in `data/images/`, with one subfolder per class (e.g., `data/images/cat/`, `data/images/dog/`).

2. **Create dataset**
   - Run `python create_dataset.py` to verify dataset loading and save class indices.

3. **Train the model**
   - Run `python train_model.py` to train and save the model.

4. **Deploy and predict**
   - Run `python deploy_model.py` and provide a path to an image to get a prediction.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- numpy

Install requirements with:

```
pip install tensorflow numpy
```
