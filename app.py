import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Cache the model loading
@st.cache_resource
def load_fundus_model():
    model = load_model("fundus_model.keras")  # Make sure this file is in the same repo
    return model

model = load_fundus_model()

# Update these with the actual classes from your training
CLASS_NAMES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "AMD"]

st.title("👁️ Fundus Image Abnormality Detection")
st.write("Upload a fundus image to detect abnormalities.")

uploaded_file = st.file_uploader("Choose a fundus image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ✅ Ensure RGB (3 channels)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # ✅ Resize to match model input size (224x224, adjust if needed)
    img = img.resize((224, 224))

    # Convert to array and normalize
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape (1,224,224,3)

    # Prediction
    preds = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)

    # Show result
    st.subheader("🔍 Prediction Result")
    st.write(f"**{predicted_class}** (Confidence: {confidence:.2f})")


