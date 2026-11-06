"""Create tiny synthetic tones for checking the local UI without RAVDESS."""
from pathlib import Path
import numpy as np
import soundfile as sf

OUT = Path(__file__).resolve().parent / "demo_audio"
SAMPLE_RATE = 22050
# Synthetic tone labels are fixtures for the interface only; they are not speech or training data.
SAMPLES = {
    "neutral": 220, "calm": 260, "happy": 330, "sad": 390,
    "angry": 470, "fearful": 560, "disgust": 670, "surprised": 800,
}

def main():
    OUT.mkdir(exist_ok=True)
    duration = 2.0
    t = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32) / SAMPLE_RATE
    for emotion, frequency in SAMPLES.items():
        # Harmonic tone with a gentle amplitude envelope, deliberately not speech.
        envelope = 0.5 + 0.2 * np.sin(2 * np.pi * 2.0 * t)
        wave = envelope * (0.65 * np.sin(2 * np.pi * frequency * t) + 0.2 * np.sin(2 * np.pi * 2 * frequency * t))
        sf.write(OUT / f"demo_{emotion}.wav", wave.astype(np.float32), SAMPLE_RATE)
    print(f"Created {len(SAMPLES)} synthetic UI fixtures in {OUT}")

if __name__ == "__main__":
    main()
