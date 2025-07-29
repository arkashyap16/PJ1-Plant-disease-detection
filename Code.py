import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("plant_disease_model.h5")

# Define class labels
class_names = [
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Pepper Bell - Bacterial Spot"
]

# Streamlit UI
st.title("🌿 Plant Disease Detector")
st.markdown("Upload a leaf image to detect plant disease")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Leaf Image', use_column_width=True)

    if st.button("Predict"):
        img = img.resize((128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"Prediction: **{class_names[class_index]}**")
        st.info(f"Confidence: {confidence:.2f}%")
