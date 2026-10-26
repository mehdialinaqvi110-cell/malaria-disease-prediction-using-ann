import pickle

import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Malaria Cell Detector", page_icon="🔬")


@st.cache_resource
def load_model():
    with open("malaria_model.pkl", "rb") as f:
        return pickle.load(f)


bundle = load_model()
clf = bundle["model"]
classes = bundle["classes"]  # e.g. ['Parasitized', 'Uninfected']

st.title("🔬 Malaria Cell Detector")
st.write("Upload a microscope image of a single blood cell to classify it.")

uploaded = st.file_uploader("Cell image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    # Decode as BGR, same as cv2.imread in the notebook
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not read that image. Try a different file.")
        st.stop()

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded image", width=250)

    # Same preprocessing as training: resize to 64x64, flatten
    resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
    features = resized.flatten().reshape(1, -1)

    pred = clf.predict(features)[0]
    proba = clf.predict_proba(features)[0]
    label = classes[pred]

    if label == "Parasitized":
        st.error(f"Prediction: **{label}**")
    else:
        st.success(f"Prediction: **{label}**")

    st.write("Confidence:")
    for cls, p in zip(classes, proba):
        st.progress(float(p), text=f"{cls}: {p:.1%}")

st.caption("Educational demo only. Not a medical diagnostic tool.")