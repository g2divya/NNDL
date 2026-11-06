from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ravdess"
ARTIFACT_DIR = ROOT / "artifacts"
FEATURE_CACHE = ARTIFACT_DIR / "features.npz"
LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
SAMPLE_RATE = 22050
MAX_SECONDS = 4.0
N_MFCC = 40
N_MELS = 64
N_CHROMA = 12
FEATURE_DIM = N_MFCC + N_MELS + N_CHROMA
SEED = 42
