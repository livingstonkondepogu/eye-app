import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
@st.cache_resource
def load_fundus_model():
    model = load_model("fundus_model.keras")
    return model

model = load_fundus_model()

# Define class labels (update if you have more than 2)
CLASS_NAMES = ["Normal", "Abnormal"]

st.title("👁️ Eye Disease Detection")

uploaded_file = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess (same as training)
    img = img.convert("RGB")       # ensure 3 channels
    img = img.resize((225, 225))   # same size as training
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0]

    # If binary classifier (1 output neuron with sigmoid)
    if len(prediction) == 1:
        label = "Abnormal" if prediction[0] > 0.5 else "Normal"
        confidence = prediction[0] if label == "Abnormal" else 1 - prediction[0]
    else:
        # Multi-class softmax
        label_index = np.argmax(prediction)
        label = CLASS_NAMES[label_index]
        confidence = prediction[label_index]

    st.subheader("Prediction")
    st.success(f"🩺 {label} (Confidence: {confidence:.2f})")


