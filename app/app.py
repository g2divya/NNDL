
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import librosa


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

MODEL_FILENAME = "best_crema_bilstm_attention.keras"

# Search possible model locations
POSSIBLE_PATHS = [
    BASE_DIR / MODEL_FILENAME,
    PROJECT_ROOT / MODEL_FILENAME,
    PROJECT_ROOT / "models" / MODEL_FILENAME,
]

MODEL_PATH = next(
    (path for path in POSSIBLE_PATHS if path.is_file()),
    None
)

SAMPLE_RATE = 16000
DURATION = 5
NUM_SAMPLES = SAMPLE_RATE * DURATION

FRAME_LENGTH = 400
FRAME_STEP = 160
FFT_LENGTH = 512

NUM_MEL_BINS = 128
LOWER_HZ = 80.0
UPPER_HZ = 7600.0

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad"
]


# =====================================================
# 2. CUSTOM FREQUENCY POOLING LAYER
# =====================================================

@tf.keras.utils.register_keras_serializable(
    package="CREMA",
    name="FrequencyPooling"
)
class FrequencyPooling(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=2)

    def get_config(self):
        return super().get_config()


# =====================================================
# 3. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():

    if MODEL_PATH is None:

        checked_paths = "\n".join(
            str(path) for path in POSSIBLE_PATHS
        )

        raise FileNotFoundError(
            "Trained model not found.\n\n"
            "Checked locations:\n" + checked_paths
        )

    if MODEL_PATH.stat().st_size == 0:
        raise ValueError("The model file is empty.")

    # Validate Keras archive
    if not zipfile.is_zipfile(MODEL_PATH):
        raise ValueError(
            f"Invalid Keras model archive: {MODEL_PATH}\n"
            "Make sure this is a real .keras model, "
            "not a Python script renamed to .keras."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "FrequencyPooling": FrequencyPooling,
            "CREMA>FrequencyPooling": FrequencyPooling
        },
        compile=False,
        safe_mode=True
    )

    return model


# =====================================================
# 4. CREATE MEL FILTER BANK
# =====================================================

@st.cache_resource
def create_mel_matrix():

    return tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MEL_BINS,
        num_spectrogram_bins=FFT_LENGTH // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=LOWER_HZ,
        upper_edge_hertz=UPPER_HZ
    )


# =====================================================
# 5. AUDIO PREPROCESSING
# =====================================================

def preprocess_audio(audio_bytes):

    audio, _ = librosa.load(
        io.BytesIO(audio_bytes),
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = audio.astype(np.float32)

    # Crop or pad to 5 seconds
    if len(audio) > NUM_SAMPLES:

        audio = audio[:NUM_SAMPLES]

    elif len(audio) < NUM_SAMPLES:

        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio))
        )

    audio = tf.convert_to_tensor(
        audio,
        dtype=tf.float32
    )

    # STFT
    stft = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
        pad_end=True
    )

    # Power spectrogram
    power_spectrogram = tf.square(
        tf.abs(stft)
    )

    # Mel spectrogram
    mel_matrix = create_mel_matrix()

    mel_spectrogram = tf.matmul(
        power_spectrogram,
        mel_matrix
    )

    # Log Mel spectrogram
    log_mel = tf.math.log(
        mel_spectrogram + 1e-6
    )

    # Per-recording normalization
    mean = tf.reduce_mean(log_mel)
    std = tf.math.reduce_std(log_mel)

    log_mel = (
        log_mel - mean
    ) / (std + 1e-6)

    # Add channel dimension
    log_mel = tf.expand_dims(
        log_mel,
        axis=-1
    )

    # Add batch dimension
    log_mel = tf.expand_dims(
        log_mel,
        axis=0
    )

    return log_mel


# =====================================================
# 6. PREDICTION
# =====================================================

def predict_emotion(model, audio_bytes):

    features = preprocess_audio(audio_bytes)

    expected_shape = tuple(
        model.input_shape[1:]
    )

    actual_shape = tuple(
        features.shape[1:]
    )

    if actual_shape != expected_shape:

        raise ValueError(
            f"Input shape mismatch.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {actual_shape}"
        )

    probabilities = model(
        features,
        training=False
    ).numpy()[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Model has {len(probabilities)} outputs, "
            f"but {len(CLASS_NAMES)} classes are configured."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned invalid probabilities.")

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_emotion = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index] * 100
    )

    return (
        predicted_emotion,
        confidence,
        probabilities
    )


# =====================================================
# 7. STREAMLIT USER INTERFACE
# =====================================================

st.title("🎙️ Speech Emotion Recognition")

st.markdown(
    """
    Upload a speech recording and let the trained
    **CNN + BiLSTM + Attention** model predict the
    emotion expressed in the audio.

    **Supported emotions:** Angry, Disgust, Fear,
    Happy, Neutral and Sad.
    """
)

st.divider()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("About the Model")

st.sidebar.markdown(
    """
    **Architecture**

    - CNN feature extractor
    - Bidirectional LSTM
    - Multi-Head Attention
    - Feed-forward network
    - Softmax classification

    **Dataset:** CREMA-D

    **Input:** 16 kHz mono audio

    **Duration:** 5 seconds
    """
)

if MODEL_PATH is not None:

    st.sidebar.success(
        f"Model found: {MODEL_PATH.name}"
    )

else:

    st.sidebar.error(
        "Trained model file not found."
    )


# =====================================================
# AUDIO UPLOAD
# =====================================================

st.subheader("Upload your speech recording")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=[
        "wav",
        "mp3",
        "flac",
        "ogg",
        "m4a"
    ]
)


if uploaded_file is not None:

    audio_bytes = uploaded_file.getvalue()

    st.audio(
        audio_bytes,
        format=uploaded_file.type
    )

    st.caption(
        f"File: {uploaded_file.name}"
    )

    if st.button(
        "Analyze Emotion",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Loading model and analyzing speech..."
            ):

                model = load_model()

                (
                    predicted_emotion,
                    confidence,
                    probabilities
                ) = predict_emotion(
                    model,
                    audio_bytes
                )

            st.success(
                "Emotion analysis completed!"
            )

            st.divider()

            # ==========================================
            # MAIN PREDICTION
            # ==========================================

            st.subheader("Predicted Emotion")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Detected emotion",
                    predicted_emotion
                )

            with col2:

                st.metric(
                    "Model confidence",
                    f"{confidence:.2f}%"
                )

            st.divider()

            # ==========================================
            # PROBABILITY CHART
            # ==========================================

            st.subheader(
                "Emotion Probability Distribution"
            )

            probability_df = pd.DataFrame({
                "Emotion": CLASS_NAMES,
                "Probability": probabilities * 100
            }).sort_values(
                "Probability",
                ascending=False
            )

            st.bar_chart(
                probability_df.set_index("Emotion"),
                horizontal=True,
                x_label="Probability (%)"
            )

            st.divider()

            # ==========================================
            # TOP THREE PREDICTIONS
            # ==========================================

            st.subheader("Top 3 Predictions")

            top_indices = np.argsort(
                probabilities
            )[::-1][:3]

            for rank, index in enumerate(
                top_indices,
                start=1
            ):

                emotion = CLASS_NAMES[index]

                score = float(
                    probabilities[index] * 100
                )

                st.write(
                    f"**{rank}. {emotion}** — {score:.2f}%"
                )

                st.progress(
                    float(probabilities[index])
                )

            st.divider()

            # ==========================================
            # DETAILED RESULTS
            # ==========================================

            st.subheader("Detailed Results")

            detailed_df = pd.DataFrame({
                "Emotion": CLASS_NAMES,
                "Probability (%)": np.round(
                    probabilities * 100,
                    2
                )
            }).sort_values(
                "Probability (%)",
                ascending=False
            )

            st.dataframe(
                detailed_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Probabilities are model outputs, not "
                "calibrated measures of certainty. "
                "Predictions may be affected by recording "
                "quality, background noise and speaking style."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {str(error)}"
            )

else:

    st.info(
        "Upload a speech recording to begin emotion analysis."
    )


# =====================================================
# 8. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
        <p>Speech Emotion Recognition</p>
        <p>CNN + BiLSTM + Attention | Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
