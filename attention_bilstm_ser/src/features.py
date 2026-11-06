"""RAVDESS discovery and log-mel/MFCC/chroma feature extraction."""
from pathlib import Path
import numpy as np
import librosa
from .config import LABELS, SAMPLE_RATE, MAX_SECONDS, N_MFCC, N_MELS

def emotion_from_path(path: Path) -> int:
    # RAVDESS filename: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
    code = int(path.stem.split("-")[2])
    mapping = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
    if code not in mapping:
        raise ValueError(f"Unknown RAVDESS emotion code in {path.name}")
    return LABELS.index(mapping[code])

def actor_id(path: Path) -> int:
    return int(path.stem.split("-")[-1])

def extract_features(path: str | Path) -> np.ndarray:
    """Return [time, 116] float32 features; pad/trim audio to four seconds."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=MAX_SECONDS)
    target = int(SAMPLE_RATE * MAX_SECONDS)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, n_fft=2048, hop_length=512)
    # Per-utterance standardization keeps feature scales comparable without fitting on test data.
    feat = np.concatenate([mfcc, mel_db, chroma], axis=0).T.astype(np.float32)
    mean = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True)
    return ((feat - mean) / np.maximum(std, 1e-5)).astype(np.float32)

def build_cache(data_dir: str | Path, output: str | Path):
    files = sorted(Path(data_dir).rglob("*.wav"))
    # RAVDESS speech only: modality 03, audio-only channel 01; excludes song recordings.
    files = [p for p in files if p.stem.startswith("03-01-")]
    if not files:
        raise FileNotFoundError(
            "No RAVDESS speech WAV files were found. Download and extract "
            "Audio_Speech_Actors_01-24.zip under "
            f"{Path(data_dir).resolve()} (the files should look like "
            "03-01-01-01-01-01-01.wav), then run prepare_data.py again. "
            "The small files in demo_audio are synthetic UI samples and cannot train the SER model."
        )
    xs, ys, actors, paths = [], [], [], []
    for i, path in enumerate(files, 1):
        try:
            xs.append(extract_features(path)); ys.append(emotion_from_path(path))
            actors.append(actor_id(path)); paths.append(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed processing {path}: {exc}") from exc
        if i % 100 == 0:
            print(f"Processed {i}/{len(files)}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X=np.asarray(xs), y=np.asarray(ys, dtype=np.int64),
                        actors=np.asarray(actors, dtype=np.int64), paths=np.asarray(paths))
    print(f"Saved {len(xs)} utterances to {output}")
