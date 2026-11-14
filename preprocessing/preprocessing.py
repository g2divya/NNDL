import io
import numpy as np
import librosa
import tensorflow as tf

# ==========================================
# CONFIGURATION
# ==========================================

SAMPLE_RATE = 16000
DURATION = 5
NUM_SAMPLES = SAMPLE_RATE * DURATION

FRAME_LENGTH = 400
FRAME_STEP = 160
FFT_LENGTH = 512

NUM_MEL_BINS = 128
LOWER_HZ = 80.0
UPPER_HZ = 7600.0


# ==========================================
# AUDIO PREPROCESSING
# ==========================================

def preprocess_audio(audio_bytes):

    # 1. Load audio as mono at 16 kHz
    audio, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=SAMPLE_RATE,
        mono=True
    )

    # 2. Crop or pad to exactly 5 seconds
    if len(audio) > NUM_SAMPLES:
        audio = audio[:NUM_SAMPLES]

    elif len(audio) < NUM_SAMPLES:
        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio))
        )

    # 3. Convert to TensorFlow tensor
    waveform = tf.convert_to_tensor(
        audio,
        dtype=tf.float32
    )

    # 4. Calculate STFT
    stft = tf.signal.stft(
        waveform,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
        pad_end=True
    )

    # 5. Calculate POWER spectrogram
    power_spectrogram = tf.square(
        tf.abs(stft)
    )

    # 6. Create Mel filter bank
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MEL_BINS,
        num_spectrogram_bins=FFT_LENGTH // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=LOWER_HZ,
        upper_edge_hertz=UPPER_HZ,
        dtype=tf.float32
    )

    # 7. Convert power spectrogram to Mel spectrogram
    mel_spectrogram = tf.matmul(
        power_spectrogram,
        mel_matrix
    )

    # 8. Log Mel spectrogram
    log_mel = tf.math.log(
        mel_spectrogram + 1e-6
    )

    # 9. Per-recording normalization
    mean = tf.reduce_mean(log_mel)
    std = tf.math.reduce_std(log_mel)

    normalized = (
        log_mel - mean
    ) / (std + 1e-6)

    # 10. Convert to NumPy
    features = normalized.numpy().astype(np.float32)

    # 11. Add batch and channel dimensions
    features = np.expand_dims(
        features,
        axis=(0, -1)
    )

    return features