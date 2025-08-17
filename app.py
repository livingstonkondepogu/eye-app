import streamlit as st
from PIL import Image
import numpy as np

st.title("👁️ EyeShield - Eye Abnormality Detection App (Demo)")

st.write("Upload an eye image to test the app.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Eye Image', use_column_width=True)

    st.write("✅ Image uploaded successfully! (Model will be added here)")
