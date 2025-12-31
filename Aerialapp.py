import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.set_page_config(page_title="Bird vs Drone Classifier", layout="centered")

# Load model
model = load_model("models/resnet_bird_drone_finetuned.keras")

st.title("🛩️ Aerial Image Classifier (Bird vs Drone)")
st.write("Upload an image and the model will detect whether it's a **Bird** or a **Drone**.")

# File uploader
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True)

    # Preprocess
    x = img.resize((224,224))
    x = np.expand_dims(np.array(x) / 255.0, axis=0)

    prob = float(model.predict(x)[0][0])
    label = "Drone" if prob > 0.5 else "Bird"
    confidence = prob if prob > 0.5 else 1 - prob

    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: **{confidence:.2f}**")
