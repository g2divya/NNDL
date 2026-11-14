
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import librosa


# =====================================================
# 1. STREAMLIT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# Model is inside app/, alongside app.py
MODEL_PATH = BASE_DIR / "best_crema_bilstm_attention.keras"


# =====================================================
# 2. AUDIO CONFIGURATION
# =====================================================

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
# 3. CUSTOM FREQUENCY POOLING LAYER
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
# 4. VALIDATE MODEL FILE
# =====================================================

def validate_model_file():

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            f"Trained model not found at:\n{MODEL_PATH}"
        )

    if MODEL_PATH.stat().st_size == 0:

        raise ValueError(
            "The model file is empty."
        )

    if not zipfile.is_zipfile(MODEL_PATH):

        with open(MODEL_PATH, "rb") as file:
            content = file.read(500)

        if b"git-lfs.github.com/spec/v1" in content:

            raise ValueError(
                "Git LFS pointer detected instead of the actual "
                "trained model. Upload the real model file."
            )

        if content.lstrip().startswith(
            (b"import ", b"from ", b"#", b"class ", b"def ")
        ):

            raise ValueError(
                "The .keras file contains Python source code, "
                "not a trained Keras model. Save the trained model "
                "using model.save('best_crema_bilstm_attention.keras')."
            )

        raise ValueError(
            "Invalid Keras archive. The model file may be corrupted."
        )

    try:

        with zipfile.ZipFile(MODEL_PATH, "r") as archive:

            if archive.testzip() is not None:

                raise ValueError(
                    "The model archive contains corrupted data."
                )

            files = archive.namelist()

            if "config.json" not in files:

                raise ValueError(
                    "Invalid model: config.json is missing."
                )

            if "metadata.json" not in files:

                raise ValueError(
                    "Invalid model: metadata.json is missing."
                )

    except zipfile.BadZipFile:

        raise ValueError(
            "The Keras model archive is corrupted."
        )


# =====================================================
# 5. LOAD TRAINED CNN + BILSTM + ATTENTION MODEL
# =====================================================

@st.cache_resource
def load_model():

    validate_model_file()

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "FrequencyPooling": FrequencyPooling,
                "CREMA>FrequencyPooling": FrequencyPooling
            },
            compile=False,
            safe_mode=True
        )

    except Exception as error:

        raise RuntimeError(
            "Unable to load the trained CNN + BiLSTM + "
            "Attention model.\n\n"
            f"Details: {str(error)}"
        ) from error

    if isinstance(model.input_shape, list):

        raise ValueError(
            "The application expects a single-input model."
        )

    if isinstance(model.output_shape, list):

        raise ValueError(
            "The application expects a single-output model."
        )

    if model.output_shape[-1] != len(CLASS_NAMES):

        raise ValueError(
            f"Expected 6 output classes, "
            f"but received {model.output_shape[-1]}."
        )

    return model


# =====================================================
# 6. CREATE MEL FILTER BANK
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
# 7. AUDIO PREPROCESSING
# =====================================================

def preprocess_audio(audio_bytes):

    try:

        audio, _ = librosa.load(
            io.BytesIO(audio_bytes),
            sr=SAMPLE_RATE,
            mono=True
        )

    except Exception as error:

        raise ValueError(
            f"Could not process audio file: {str(error)}"
        ) from error

    if len(audio) == 0:

        raise ValueError(
            "The uploaded audio contains no samples."
        )

    audio = audio.astype(np.float32)

    # Crop or pad audio to 5 seconds
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

    # Short-Time Fourier Transform
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
# 8. EMOTION PREDICTION
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
            "Input shape mismatch!\n\n"
            f"Model expects: {expected_shape}\n"
            f"Preprocessing produces: {actual_shape}\n\n"
            "Check that the preprocessing parameters match "
            "the original training code."
        )

    predictions = model(
        features,
        training=False
    ).numpy()[0]

    if not np.all(np.isfinite(predictions)):

        raise ValueError(
            "The model returned invalid predictions."
        )

    if len(predictions) != len(CLASS_NAMES):

        raise ValueError(
            "Model output does not match the six emotion classes."
        )

    # Handle logits if the model does not already output probabilities
    if (
        np.any(predictions < 0)
        or not np.isclose(
            np.sum(predictions),
            1.0,
            atol=1e-3
        )
    ):

        predictions = tf.nn.softmax(
            predictions
        ).numpy()

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_emotion = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index] * 100
    )

    return (
        predicted_emotion,
        confidence,
        predictions
    )


# =====================================================
# 9. APPLICATION HEADER
# =====================================================

st.title("🎙️ Speech Emotion Recognition")

st.markdown(
    """
    Upload a speech recording to identify the emotion
    expressed in the voice using a trained

    **CNN + BiLSTM + Attention Deep Learning Model.**

    The model recognizes six emotions:

    - Angry
    - Disgust
    - Fear
    - Happy
    - Neutral
    - Sad
    """
)

st.divider()


# =====================================================
# 10. SIDEBAR
# =====================================================

st.sidebar.title("About the Model")

st.sidebar.markdown(
    """
    **Model Architecture**

    - CNN feature extractor
    - Bidirectional LSTM
    - Attention mechanism
    - Dense classification layer
    - Softmax output

    **Dataset:** CREMA-D

    **Sample Rate:** 16 kHz

    **Audio Duration:** 5 seconds

    **Features:** 128 Mel frequency bins
    """
)

if MODEL_PATH.is_file():

    st.sidebar.success(
        "Model file detected."
    )

    st.sidebar.caption(
        MODEL_PATH.name
    )

else:

    st.sidebar.error(
        "Trained model not found."
    )


# =====================================================
# 11. AUDIO UPLOAD
# =====================================================

st.subheader("Upload Speech Recording")

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


# =====================================================
# 12. ANALYZE AUDIO
# =====================================================

if uploaded_file is not None:

    audio_bytes = uploaded_file.getvalue()

    st.audio(
        audio_bytes,
        format=uploaded_file.type
    )

    st.caption(
        f"Uploaded file: {uploaded_file.name}"
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
                "Emotion analysis completed successfully!"
            )

            st.divider()

            # =========================================
            # PREDICTED EMOTION
            # =========================================

            st.subheader("Predicted Emotion")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Detected Emotion",
                    predicted_emotion
                )

            with col2:

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

            st.divider()

            # =========================================
            # PROBABILITY DISTRIBUTION
            # =========================================

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

            # =========================================
            # TOP THREE PREDICTIONS
            # =========================================

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
                    f"**{rank}. {emotion} — {score:.2f}%**"
                )

                st.progress(
                    float(probabilities[index])
                )

            st.divider()

            # =========================================
            # DETAILED RESULTS
            # =========================================

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
                "Probabilities are model outputs, not calibrated "
                "measures of certainty. Predictions can be affected "
                "by recording quality, background noise and speaking style."
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
# 13. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
        <p><b>Speech Emotion Recognition</b></p>
        <p>CNN + BiLSTM + Attention | CREMA-D Dataset</p>
    </div>
    """,
    unsafe_allow_html=True
)
