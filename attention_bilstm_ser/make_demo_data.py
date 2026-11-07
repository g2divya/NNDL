"""Create tiny synthetic tones for checking the local UI without RAVDESS."""
from pathlib import Path
import wave
import numpy as np

try:
    import soundfile as sf  # type: ignore
    HAS_SOUNDFILE = True
except ImportError:
    sf = None
    HAS_SOUNDFILE = False

OUT = Path(__file__).resolve().parent / "demo_audio"
SAMPLE_RATE = 22050
# Synthetic tone labels are fixtures for the interface only; they are not speech or training data.
SAMPLES = {
    "neutral": 220, "calm": 260, "happy": 330, "sad": 390,
    "angry": 470, "fearful": 560, "disgust": 670, "surprised": 800,
}


def _write_wav(path: Path, samples: np.ndarray, sr: int = SAMPLE_RATE):
    if HAS_SOUNDFILE:
        sf.write(path, samples.astype(np.float32), sr)
    else:
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())


def main():
    OUT.mkdir(exist_ok=True)
    duration = 2.0
    t = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    for emotion, frequency in SAMPLES.items():
        # Harmonic tone with a gentle amplitude envelope, deliberately not speech.
        envelope = 0.5 + 0.2 * np.sin(2 * np.pi * 2.0 * t)
        sig = envelope * (
            0.65 * np.sin(2 * np.pi * frequency * t)
            + 0.2 * np.sin(2 * np.pi * 2 * frequency * t)
        )
        _write_wav(OUT / f"demo_{emotion}.wav", sig.astype(np.float32), SAMPLE_RATE)
    print(f"Created {len(SAMPLES)} synthetic UI fixtures in {OUT}")


if __name__ == "__main__":
    main()
