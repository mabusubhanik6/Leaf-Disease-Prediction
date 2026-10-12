import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)


# =====================================================
# FILE PATHS
# =====================================================

MODEL_PATH = "models/potato_leaf_cnn.keras"
INFO_PATH = "src/disease_info.json"
TRANSLATION_PATH = "src/translations.json"

IMG_SIZE = (128, 128)

CLASS_NAMES = [
    "Potato_Early_Blight",
    "Potato_Healthy",
    "Potato_Late_Blight"
]


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# =====================================================
# LOAD DISEASE INFORMATION
# =====================================================

with open(INFO_PATH, "r", encoding="utf-8") as file:
    disease_info = json.load(file)


# =====================================================
# LOAD TRANSLATIONS
# =====================================================

with open(TRANSLATION_PATH, "r", encoding="utf-8") as file:
    translations = json.load(file)


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_disease(image):

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    image_array = np.array(image)

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = predictions[0][predicted_index] * 100

    return predicted_class, confidence


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🌐 Language")

language = st.sidebar.selectbox(
    "Select Language",
    [
        "English",
        "Telugu",
        "Hindi",
        "Kannada"
    ]
)

text = translations[language]


# =====================================================
# HEADER
# =====================================================

st.title(text["title"])

st.write(text["description"])

st.markdown(
    """
    **🌱 Supported Crop:** Potato  
    **🧠 Model:** Convolutional Neural Network (CNN)
    """
)

st.divider()


# =====================================================
# IMAGE INPUT
# =====================================================

col1, col2 = st.columns(2)


with col1:

    uploaded_image = st.file_uploader(
        text["upload"],
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


with col2:

    camera_image = st.camera_input(
        text["camera"]
    )


# =====================================================
# SELECT IMAGE
# =====================================================

image = None

if camera_image is not None:

    image = camera_image

elif uploaded_image is not None:

    image = uploaded_image


# =====================================================
# DISPLAY IMAGE
# =====================================================

if image is not None:

    img = Image.open(image)

    st.divider()

    st.subheader(
        text["selected_leaf"]
    )

    image_col1, image_col2 = st.columns(
        [1, 1]
    )

    with image_col1:

        st.image(
            img,
            caption=text["selected_leaf"],
            use_container_width=True
        )

    with image_col2:

        st.write("### 🌿 Ready for Analysis")

        st.write(
            "The image will be processed by the trained CNN model."
        )

        predict_button = st.button(
            text["predict"],
            use_container_width=True
        )


    # =================================================
    # PREDICTION
    # =================================================

    if predict_button:

        predicted_class, confidence = predict_disease(
            img
        )

        # Get information according to selected language
        info = disease_info[
            predicted_class
        ][
            language
        ]

        st.divider()

        st.subheader(
            text["prediction"]
        )


        # =============================================
        # RESULT
        # =============================================

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.success(
                f"🌿 {text['disease']}\n\n"
                f"### {info['name']}"
            )


        with result_col2:

            st.info(
                f"📊 {text['confidence']}\n\n"
                f"### {confidence:.2f}%"
            )


        # =============================================
        # CONFIDENCE BAR
        # =============================================

        confidence_value = float(confidence) / 100

        st.progress(
                    min(confidence_value, 1.0)
                        )


        # =============================================
        # EXPLANATION
        # =============================================

        st.subheader(
            text["explanation"]
        )

        st.write(
            info["explanation"]
        )


        # =============================================
        # SYMPTOMS
        # =============================================

        st.subheader(
            text["symptoms"]
        )

        for symptom in info["symptoms"]:

            st.write(
                "🔹",
                symptom
            )


        # =============================================
        # PRECAUTIONS
        # =============================================

        st.subheader(
            text["precautions"]
        )

        for precaution in info["precautions"]:

            st.write(
                "🛡️",
                precaution
            )


        # =============================================
        # HEALTH TIPS
        # =============================================

        st.subheader(
            text["health_tips"]
        )

        for tip in info["health_tips"]:

            st.write(
                "🌱",
                tip
            )


        # =============================================
        # DISCLAIMER
        # =============================================

        st.divider()

        st.caption(
            "⚠️ This application provides an AI-based prediction "
            "and general plant-health information. For important "
            "agricultural decisions, consult a qualified local "
            "agriculture expert."
        )


# =====================================================
# NO IMAGE MESSAGE
# =====================================================

else:

    st.info(
        "🌿 Please upload a leaf image or take a photo "
        "to begin disease detection."
    )