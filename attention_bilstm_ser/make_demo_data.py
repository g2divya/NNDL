"""Create synthetic prosodic tones for checking the local UI without RAVDESS."""
from pathlib import Path
import wave
import numpy as np

OUT = Path(__file__).resolve().parent / "demo_audio"
SAMPLE_RATE = 22050
# Synthetic tone labels are fixtures for the interface only; they are not speech or training data.
SAMPLES = {
    "neutral": 220, "calm": 260, "happy": 330, "sad": 390,
    "angry": 470, "fearful": 560, "disgust": 670, "surprised": 800,
}

# Expressive syllable modulation rates per sample so waveforms and attention maps look dynamic
MODULATION = {
    "neutral": (1.8, 0.20),
    "calm": (1.2, 0.25),
    "happy": (3.2, 0.35),
    "sad": (1.0, 0.30),
    "angry": (4.5, 0.42),
    "fearful": (5.2, 0.38),
    "disgust": (2.2, 0.32),
    "surprised": (3.8, 0.45),
}


def write_wav(path: Path, wave_f32: np.ndarray, sr: int = SAMPLE_RATE):
    clipped = np.clip(wave_f32, -0.98, 0.98)
    pcm16 = (clipped * 32767.0).astype(np.int16)
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
        syl_rate, depth = MODULATION[emotion]
        # Smooth fade-in/out window with multi-syllable envelope and dominant fundamental pitch
        window = np.sin(np.pi * t / duration) ** 0.35
        envelope = window * (0.52 + depth * np.sin(2 * np.pi * syl_rate * t))
        harmonic = (
            0.68 * np.sin(2 * np.pi * frequency * t)
            + 0.20 * np.sin(2 * np.pi * 2 * frequency * t)
            + 0.08 * np.sin(2 * np.pi * 3 * frequency * t)
        )
        sig = (envelope * harmonic).astype(np.float32)
        write_wav(OUT / f"demo_{emotion}.wav", sig, SAMPLE_RATE)
    print(f"Created {len(SAMPLES)} synthetic UI fixtures in {OUT}")


if __name__ == "__main__":
    main()
