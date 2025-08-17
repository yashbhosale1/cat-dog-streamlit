import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown(
    """
    <style>
    /* Center everything and set max width */
    .block-container {max-width: 800px; margin: auto;}

    /* Title style */
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #4B4BFF;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 20px;
        color: #333;
    }

    /* Uploaded image */
    .stImage {
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* Footer */
    .footer {
        margin-top: 40px;
        text-align: center;
        font-size: 0.9rem;
        color: #777;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ HEADER ------------------
st.markdown("<div class='title'>🐶🐱 Cat vs Dog Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image and I will tell you if it’s a <b>Cat</b> or a <b>Dog</b>.</div>", unsafe_allow_html=True)

# ------------------ MODEL ------------------
model = MobileNetV2(weights="imagenet")

# ------------------ FILE UPLOADER ------------------
uploaded_file = st.file_uploader("📂 Upload an image...", type=["jpg", "jpeg", "png"])

# ------------------ PREDICTION LOGIC ------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]

    # Check if prediction belongs to cat or dog
    cat_labels = ["Egyptian_cat", "tabby", "tiger_cat", "Persian_cat", "Siamese_cat"]
    pred_class = decoded[0][1]
    pred_id = np.argmax(preds)

    # Result styling
    if pred_class in cat_labels:
        st.success("🐱 It's a **Cat**!")
    elif 151 <= pred_id <= 268:
        st.success("🐶 It's a **Dog**!")
    else:
        st.warning(f"🤔 Not sure... looks like a **{pred_class}**")

# ------------------ FOOTER ------------------
st.markdown("<div class='footer'>Made with ❤️ using Streamlit + TensorFlow</div>", unsafe_allow_html=True)
