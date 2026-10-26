# Malaria Cell Classification using ANN and HOG

A Streamlit web app that classifies an uploaded blood-cell microscope image as **Parasitized** or **Uninfected** using a trained Artificial Neural Network (ANN) and Histogram of Oriented Gradients (HOG) features.

> **Disclaimer:** This is an educational machine-learning project, not a medical diagnostic tool. Do not use its predictions for healthcare decisions.

## Features

- Upload a cell image in PNG, JPG, or JPEG format.
- Resize the image to `64 × 64`.
- Convert the image to RGB and normalize pixel values to `[0, 1]`.
- Extract HOG features using the same parameters as the training notebook.
- Run the feature vector through the saved Keras ANN model.
- Display the predicted class and model output scores.

## Project structure

```text
malaria-ann-hog/
├── app.py
├── requirements.txt
├── README.md
└── model1.keras        # Add your trained model file here
```

The trained model is not included automatically. Copy your saved `model1.keras` file into the project folder before running the app. If your model has a different filename, update `MODEL_PATH` in `app.py`.

## Model and preprocessing

The training notebook uses `image_dataset_from_directory` with:

- Image size: `64 × 64`
- Batch size: `32`
- Label mode: `categorical`
- Pixel normalization: divide by `255.0`
- HOG orientations: `9`
- Pixels per cell: `(8, 8)`
- Cells per block: `(2, 2)`
- HOG channel axis: `-1` (three-channel image input)
- ANN output: two units

For a `64 × 64` RGB image and these HOG settings, the feature vector is expected to contain **1,764 features**.

The app configures class names as:

```python
CLASS_NAMES = ["Parasitized", "Uninfected"]
```

This must match the exact order in `dataset.class_names` from the training notebook. Since `image_dataset_from_directory` infers labels from folders, verify the order in your training environment before using the app.

## Run locally

### 1. Create or open the project folder

Put `app.py`, `requirements.txt`, and `model1.keras` in the same directory.

### 2. Install dependencies

Python 3.10 or 3.11 is a reasonable starting point for compatibility with TensorFlow; use a TensorFlow-supported Python version for your deployment environment.

```bash
pip install -r requirements.txt
```

### 3. Start Streamlit

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`, which you can open in your browser.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `README.md`, and `model1.keras` to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and select your repository and `app.py` as the main file.
4. Deploy the app.

**Model-file note:** Keras model files can be large. Check GitHub's current file-size limits before committing the model. If it is too large, store it using an appropriate model/file hosting service and update the loading code to retrieve it securely.

## Important implementation notes

- The uploaded image is converted from OpenCV's BGR format to RGB to match the TensorFlow image dataset pipeline.
- HOG parameters and image dimensions at inference time must match those used during training.
- The output layer in the training notebook uses sigmoid activations with categorical cross-entropy. The app selects the class with the highest output score. These scores should not automatically be interpreted as calibrated probabilities.
- This repository contains an inference app; it does not include the original cell-image dataset or retrain the ANN.

## Future improvements

- Add a confusion matrix and per-class precision, recall, and F1-score to the model evaluation.
- Save and document the exact class order and training configuration.
- Test predictions on a held-out dataset and monitor false negatives.
- Consider improving calibration and validating the model on representative data before any real-world use.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- scikit-image (HOG)
- Streamlit
