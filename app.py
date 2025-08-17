import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("👁️ EyeShield: Early Eye Risk Detection")
st.write("Upload an eye image and the app will simulate abnormality detection.")

uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Convert to BGR for OpenCV processing
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # ----- Dummy Segmentation (semi-transparent red overlay on edges) -----
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)  # dummy edge detection

    mask = np.zeros_like(img_cv)
    mask[:, :, 2] = edges  # Red channel mask
    alpha = 0.4  # transparency
    overlay = cv2.addWeighted(img_cv, 1, mask, alpha, 0)

    # Convert back to RGB for display
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Show images
    st.image(image, caption="Original Eye", use_column_width=True)
    st.image(overlay_rgb, caption="Detected Abnormality (Demo Overlay)", use_column_width=True)

    # ----- Dummy Text Result -----
    st.subheader("📋 Report")
    st.success("⚠️ Possible abnormality detected. Please consult an ophthalmologist for confirmation.")

