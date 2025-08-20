import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------
# Load model with safe_mode=False
# -----------------------------
@st.cache_resource
def load_fundus_model():
    try:
        model = load_model("fundus_model.keras", safe_mode=False)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
    return model

model = load_fundus_model()

# -----------------------------
# Define class labels (edit to match your training)
# -----------------------------
CLASS_NAMES = [
    "Normal",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Age-related Macular Degeneration",
    "Hypertensive Retinopathy",
    "Other Abnormality"
]

# -----------------------------
# Preprocess uploaded image
# -----------------------------
def preprocess_image(uploaded_file):
    # Read as OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Resize to model input (225x225) — adjust if needed
    img = cv2.resize(img, (225, 225))

    # Normalize [0,1]
    img = img.astype("float32") / 255.0

    # Ensure shape (225, 225, 3)
    if img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("👁️ Fundus Image Abnormality Detection")
st.write("Upload a retinal fundus image to check for abnormalities.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = preprocess_image(uploaded_file)

    # Predict
    preds = model.predict(img)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    st.subheader("🔍 Prediction Result")
    st.write(f"**Class:** {CLASS_NAMES[pred_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")


