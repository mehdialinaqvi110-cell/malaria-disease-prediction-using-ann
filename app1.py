
import cv2
import numpy as np
import streamlit as st

from tensorflow.keras.models import load_model
from skimage.feature import hog

st.set_page_config(
    page_title="Malaria Cell Detector",
    page_icon="🔬"
)


@st.cache_resource
def load_ann_model():
    return load_model("malaria_model.keras")


model = load_ann_model()

# Must match the class encoding used during training
classes = ["Parasitized", "Uninfected"]

st.title("🔬 Malaria Cell Detector")
st.write("Upload a microscope image of a single blood cell.")

uploaded = st.file_uploader(
    "Upload cell image",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:

    file_bytes = np.frombuffer(
        uploaded.read(),
        dtype=np.uint8
    )

    # OpenCV loads the image in BGR format
    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if img is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    # Convert BGR to RGB, matching a typical
    # TensorFlow image_dataset_from_directory pipeline
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(
        rgb,
        caption="Uploaded blood cell",
        width=250
    )

    # IMPORTANT:
    # Set these dimensions to the exact dimensions
    # used when creating your training dataset.
    resized = cv2.resize(
        rgb,
        (64, 64),
        interpolation=cv2.INTER_LINEAR
    )

    # Normalize exactly as in training
    normalized = resized.astype(
        np.float32
    ) / 255.0

    # Extract HOG features using your training settings
    features = hog(
        normalized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1,
        block_norm="L2-Hys",
        feature_vector=True
    )

    x_input = np.array(
        features,
        dtype=np.float32
    ).reshape(1, -1)

    st.write("HOG input shape:", x_input.shape)
    st.write("ANN expected input shape:", model.input_shape)

    # Validate input feature count
    expected_features = model.input_shape[-1]

    if x_input.shape[1] != expected_features:
        st.error(
            f"Feature mismatch: extracted {x_input.shape[1]} "
            f"features, but the model expects {expected_features}. "
            "Check the training image dimensions and HOG settings."
        )
        st.stop()

    # Predict
    scores = model.predict(x_input, verbose=0)[0]

    predicted_index = int(np.argmax(scores))
    label = classes[predicted_index]

    if label == "Parasitized":
        st.error(f"Prediction: {label}")
    else:
        st.success(f"Prediction: {label}")

    st.write("Model output scores:")

    for cls, score in zip(classes, scores):
        st.progress(
            float(np.clip(score, 0, 1)),
            text=f"{cls}: {score:.1%}"
        )

st.caption(
    "Educational demo only. Not a medical diagnostic tool."
)
