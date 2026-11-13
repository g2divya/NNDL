import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio

# Generate Mel Spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=512,
    hop_length=160,
    n_mels=128,
    power=2
)

# Convert power to decibels
mel_db = librosa.power_to_db(
    mel_spec,
    ref=np.max
)

# Plot
librosa.display.specshow(
    mel_db,
    sr=sr,
    hop_length=160,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram")
plt.show()