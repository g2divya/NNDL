import librosa
import numpy as np
import os
import glob

DATA_PATH = r"C:\Users\Prachi Singh\OneDrive\เอกสาร\Downloads\RAVDESS\Audio_Speech_Actors_01-24_16k"

SAMPLE_RATE = 16000
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

X = []
y = []
actors = []

for file in glob.glob(DATA_PATH + "/Actor_*/*.wav"):

    # Extract emotion and actor ID from filename
    filename = os.path.basename(file)
    parts = filename.split("-")

    emotion = int(parts[2])
    actor = int(parts[6].split(".")[0])

    # Load audio
    audio, sr = librosa.load(
        file,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    # Make all audio files equal length
    if len(audio) < SAMPLES:
        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )
    else:
        audio = audio[:SAMPLES]

    # Generate Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=128,
        fmax=8000,
        power=2
    )

    # Convert to decibels
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Normalize
    mel_db = np.clip(mel_db, -80, 0)
    mel_db = (mel_db + 80) / 80

    X.append(mel_db)
    y.append(emotion - 1)
    actors.append(actor)

X = np.array(X)
y = np.array(y)
actors = np.array(actors)

# Add channel dimension for CNN
X = X[..., np.newaxis]

print("X shape:", X.shape)
print("y shape:", y.shape)