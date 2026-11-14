
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path


# ==========================================
# 1. CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CNN BiLSTM Attention - Speech Emotion",
    page_icon="🎙️",
    layout="wide"
)

MODEL_PATH = (
    Path(__file__).parent
    / "CREMA-CNN-BILSTM-ATTENTION-RESULTS"
    / "best_crema_bilstm_attention.keras"
)

EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad"
]


# ==========================================
# 2. CUSTOM LAYER
# ==========================================

@tf.keras.utils.register_keras_serializable(
    package="CREMA",
    name="FrequencyPooling"
)
class FrequencyPooling(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=2)

    def get_config(self):
        return super().get_config()


CUSTOM_OBJECTS = {
    "FrequencyPooling": FrequencyPooling,
    "CREMA>FrequencyPooling": FrequencyPooling
}


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_emotion_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
        safe_mode=True
    )

    return model


# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================

def predict_emotion(model, features):

    # Accept (500, 128, 1) or (1, 500, 128, 1)
    if features.ndim == 3:
        features = np.expand_dims(features, axis=0)

    if features.shape != (1, 500, 128, 1):
        raise ValueError(
            f"Expected shape (1, 500, 128, 1), "
            f"received {features.shape}"
        )

    features = features.astype(np.float32)

    predictions = model.predict(
        features,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = np.argmax(probabilities)

    predicted_emotion = EMOTIONS[predicted_index]

    return predicted_emotion, probabilities


# ==========================================
# 5. STREAMLIT INTERFACE
# ==========================================

st.title("🎙️ Speech Emotion Recognition")

st.subheader("CNN + BiLSTM + Attention Model")

st.write(
    "Upload a preprocessed CREMA-D feature array "
    "to predict the speech emotion."
)

st.info(
    "Upload a NumPy .npy feature file. "
    "Raw audio files are not supported in this version."
)


# ==========================================
# 6. MODEL LOADING
# ==========================================

if not MODEL_PATH.exists():

    st.error(
        f"Model file not found: {MODEL_PATH}"
    )

    st.stop()

try:

    model = load_emotion_model()

    st.success("CNN + BiLSTM + Attention model loaded successfully.")

except Exception as e:

    st.error("Unable to load the trained model.")
    st.exception(e)

    st.stop()


# ==========================================
# 7. FEATURE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload preprocessed features (.npy)",
    type=["npy"]
)


if uploaded_file is not None:

    try:

        features = np.load(
            uploaded_file,
            allow_pickle=False
        )

        st.write("Feature shape:", features.shape)

        if features.ndim == 3 and features.shape == (500, 128, 1):
            st.success("Valid feature dimensions.")

        elif features.ndim == 4 and features.shape == (1, 500, 128, 1):
            st.success("Valid feature dimensions.")

        else:
            st.error(
                "Invalid feature shape. "
                "Expected (500, 128, 1) or (1, 500, 128, 1)."
            )
            st.stop()

        if st.button("Predict Emotion", type="primary"):

            with st.spinner("Predicting emotion..."):

                emotion, probabilities = predict_emotion(
                    model,
                    features
                )

            # Prediction
            st.divider()

            st.subheader("Prediction Result")

            st.metric(
                label="Predicted Emotion",
                value=emotion
            )

            # Probability table
            results = pd.DataFrame({
                "Emotion": EMOTIONS,
                "Probability": probabilities
            })

            results = results.sort_values(
                by="Probability",
                ascending=False
            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Emotion Probabilities")

                st.bar_chart(
                    results.set_index("Emotion")
                )

            with col2:

                st.subheader("Prediction Details")

                display_results = results.copy()

                display_results["Probability"] = (
                    display_results["Probability"] * 100
                ).map(lambda x: f"{x:.2f}%")

                st.dataframe(
                    display_results,
                    use_container_width=True,
                    hide_index=True
                )

    except Exception as e:

        st.error("Error processing the uploaded feature file.")
        st.exception(e)


# ==========================================
# 8. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("Model Information")

    st.write("**Dataset:** CREMA-D")

    st.write("**Architecture:**")

    st.write("""
    - CNN feature extraction
    - Frequency pooling
    - Bidirectional LSTM
    - Multi-head attention
    - Feed-forward network
    - Global average + max pooling
    - Dense classification layer
    """)

    st.divider()

    st.write("**Emotion Classes:**")

    for emotion in EMOTIONS:
        st.write(f"- {emotion}")
