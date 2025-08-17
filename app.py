import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time

st.set_page_config(page_title="👁️ EyeShield", layout="centered")
st.title("👁️ EyeShield: Early Eye Risk Detection")
st.write("Upload an eye image and the app will simulate abnormality detection.")

uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="Original Eye Image", use_column_width=True)

    # ---- Progress Bar ----
    st.info("Processing image...")
    progress_bar = st.progress(0)
    for i in range(1, 101):
        time.sleep(0.01)  # simulate processing
        progress_bar.progress(i)

    # Convert to BGR for OpenCV
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # ----- Dummy Segmentation (semi-transparent red overlay on edges) -----
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    mask = np.zeros_like(img_cv)
    mask[:, :, 2] = edges  # Red channel mask
    alpha = 0.4
    overlay = cv2.addWeighted(img_cv, 1, mask, alpha, 0)

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Show processed image
    st.image(overlay_rgb, caption="Detected Abnormality (Demo Overlay)", use_column_width=True)

    # ----- Download Button -----
    buffered = io.BytesIO()
    overlay_pil = Image.fromarray(overlay_rgb)
    overlay_pil.save(buffered, format="PNG")
    st.download_button(
        label="⬇️ Download Processed Image",
        data=buffered,
        file_name="processed_eye.png",
        mime="image/png"
    )

    # ----- Dummy Report -----
    st.subheader("📋 Report")
    st.success("⚠️ Possible abnormality detected. Please consult an ophthalmologist for confirmation.")
