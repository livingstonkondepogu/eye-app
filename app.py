import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------- Load Trained Model --------------------
@st.cache_resource
def load_fundus_model():
    model = load_model("fundus_model.keras")  # make sure this file is in same folder
    return model

model = load_fundus_model()
class_labels = ["Normal", "Abnormal"]

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="👁️ EyeShield", layout="centered")
st.title("👁️ EyeShield: Early Eye Risk Detection")
st.write("Upload an eye image and the app will analyze for abnormalities.")

uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Eye Image", use_column_width=True)

    # ---- Progress Bar ----
    st.info("Processing image...")
    progress_bar = st.progress(0)
    for i in range(1, 101):
        time.sleep(0.01)  # simulate processing
        progress_bar.progress(i)

    # -------------------- Preprocessing --------------------
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim

    # -------------------- Prediction --------------------
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    pred_label = class_labels[pred_class]
    confidence = float(np.max(preds)) * 100

    # -------------------- Visualization (overlay edges just for demo) --------------------
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    mask = np.zeros_like(img_cv)
    mask[:, :, 2] = edges
    overlay = cv2.addWeighted(img_cv, 1, mask, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    st.image(overlay_rgb, caption=f"Model Prediction: {pred_label}", use_column_width=True)

    # -------------------- Download Option --------------------
    buffered = io.BytesIO()
    overlay_pil = Image.fromarray(overlay_rgb)
    overlay_pil.save(buffered, format="PNG")
    st.download_button(
        label="⬇️ Download Processed Image",
        data=buffered,
        file_name="processed_eye.png",
        mime="image/png"
    )

    # -------------------- Report --------------------
    st.subheader("📋 Report")
    if pred_label == "Abnormal":
        st.error(f"⚠️ Abnormality detected with {confidence:.2f}% confidence. Please consult an ophthalmologist.")
    else:
        st.success(f"✅ No abnormality detected. Confidence: {confidence:.2f}%")

