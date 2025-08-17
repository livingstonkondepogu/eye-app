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

    # ----- Dummy Segmentation (overlay red mask on center) -----
    mask = np.zeros_like(img_cv)
    h, w, _ = img_cv.shape
    cv2.circle(mask, (w//2, h//2), min(h,w)//4, (0,0,255), -1)  # red circle as abnormality

    # Overlay mask on original image
    overlay = cv2.addWeighted(img_cv, 0.7, mask, 0.3, 0)

    # Convert back to RGB for display
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Show images
    st.image(image, caption="Original Eye", use_column_width=True)
    st.image(overlay_rgb, caption="Detected Abnormality (Dummy)", use_column_width=True)

    # ----- Dummy Text Result -----
    st.subheader("📋 Report")
    st.success("⚠️ Possible abnormality detected. Please consult an ophthalmologist for confirmation.")
