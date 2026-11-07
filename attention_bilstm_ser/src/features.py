"""RAVDESS discovery, audio preprocessing, and log-mel/MFCC/chroma feature extraction."""
from __future__ import annotations

import io
import os
import tempfile
import wave
from math import gcd
from pathlib import Path
import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly

from .config import LABELS, SAMPLE_RATE, MAX_SECONDS, N_MFCC, N_MELS, N_CHROMA, FEATURE_DIM

try:
    import librosa  # type: ignore
    HAS_LIBROSA = True
except ImportError:
    librosa = None
    HAS_LIBROSA = False


def _parse_ravdess_parts(path: Path) -> list[str]:
    parts = Path(path).stem.split("-")
    if len(parts) != 7 or not all(p.isdigit() for p in parts):
        raise ValueError(
            f"Invalid RAVDESS filename {Path(path).name!r}; expected 7 numeric fields "
            "(modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav)."
        )
    return parts


def emotion_from_path(path: Path) -> int:
    # RAVDESS filename: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
    parts = _parse_ravdess_parts(Path(path))
    code = int(parts[2])
    mapping = {
        1: "neutral", 2: "calm", 3: "happy", 4: "sad",
        5: "angry", 6: "fearful", 7: "disgust", 8: "surprised",
    }
    if code not in mapping:
        raise ValueError(f"Unknown RAVDESS emotion code {code} in {Path(path).name}")
    return LABELS.index(mapping[code])


def actor_id(path: Path) -> int:
    parts = _parse_ravdess_parts(Path(path))
    actor = int(parts[-1])
    if actor < 1:
        raise ValueError(f"Invalid actor ID {actor} in {Path(path).name}")
    return actor


# ---------------------------------------------------------------------------
# Signal Preprocessing: Silence Trimming, DC-Offset Removal & Peak Norm
# ---------------------------------------------------------------------------
def trim_silence(y: np.ndarray, sr: int = SAMPLE_RATE, top_db: float = 28.0) -> np.ndarray:
    """Idempotently trim leading and trailing silence using sample threshold and frame RMS energy."""
    y = np.asarray(y, dtype=np.float32)
    non_zero = np.where(np.abs(y) > 1e-5)[0]
    if len(non_zero) == 0:
        return y
    y = y[int(non_zero[0]) : int(non_zero[-1]) + 1]
    if HAS_LIBROSA:
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db, frame_length=2048, hop_length=512)
        if len(y_trimmed) > 0:
            nz = np.where(np.abs(y_trimmed) > 1e-5)[0]
            return y_trimmed[int(nz[0]) : int(nz[-1]) + 1] if len(nz) > 0 else y_trimmed
        return y

    rms = compute_rms_energy(y, frame_length=2048, hop_length=512)
    ref = float(np.max(rms))
    if ref < 1e-6:
        return y
    db = 20.0 * np.log10(np.maximum(rms, 1e-10) / ref)
    active = np.where(db > -top_db)[0]
    if len(active) == 0:
        return y
    start = max(0, int(active[0]) * 512)
    end = min(len(y), int(active[-1] + 1) * 512)
    y_sub = y[start:end] if end > start else y
    nz = np.where(np.abs(y_sub) > 1e-5)[0]
    return y_sub[int(nz[0]) : int(nz[-1]) + 1] if len(nz) > 0 else y_sub


def preprocess_waveform(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    max_seconds: float = MAX_SECONDS,
    trim_db: float = 28.0,
) -> tuple[np.ndarray, int]:
    """
    Idempotent speech preprocessing pipeline:
      1. Convert to 1D float32 and trim leading/trailing silence (top_db=28).
      2. Remove DC offset on the active voiced segment and re-trim any residual silence.
      3. Peak-normalize active speech to 0.95 (-0.45 dBFS) to match studio recording levels.
      4. Pad or trim to exactly sr * max_seconds (88,200 samples) and return (y_padded, active_frames).
    """
    target = int(sr * max_seconds)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    if len(y) == 0:
        return np.zeros(target, dtype=np.float32), 1

    # 1. Trim leading and trailing silence before DC removal so zero-padded tails stay zero
    y_trimmed = trim_silence(y, sr=sr, top_db=trim_db)
    if len(y_trimmed) < int(0.15 * sr):
        y_trimmed = y

    # 2. Remove DC offset on the active speech segment and re-trim
    y_trimmed = y_trimmed - float(np.mean(y_trimmed))
    y_trimmed = trim_silence(y_trimmed, sr=sr, top_db=trim_db)

    # 3. Peak-normalize active waveform
    peak = float(np.max(np.abs(y_trimmed))) if len(y_trimmed) > 0 else 0.0
    if peak > 1e-5:
        y_trimmed = (y_trimmed / peak) * 0.95

    # 4. Compute active frame count (hop_length=512, total 173 frames for 4.0s)
    total_frames = 1 + int(target) // 512
    clipped_len = min(len(y_trimmed), target)
    active_frames = min(total_frames, max(1, 1 + clipped_len // 512))

    # 5. Pad or trim to 4.0 seconds
    if len(y_trimmed) < target:
        y_out = np.pad(y_trimmed, (0, target - len(y_trimmed)))
    else:
        y_out = y_trimmed[:target]

    return y_out.astype(np.float32), int(active_frames)


# ---------------------------------------------------------------------------
# Pure NumPy / SciPy acoustic feature primitives (Slaney Mel + Ortho DCT-II)
# ---------------------------------------------------------------------------
def _hz_to_mel_slaney(frequencies: np.ndarray | float) -> np.ndarray:
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    f_min = 0.0
    f_sp = 200.0 / 3.0
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = frequencies >= min_log_hz
    mels[log_t] = min_log_mel + np.log(np.maximum(frequencies[log_t], 1e-10) / min_log_hz) / logstep
    return mels


def _mel_to_hz_slaney(mels: np.ndarray | float) -> np.ndarray:
    mels = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    f_min = 0.0
    f_sp = 200.0 / 3.0
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def _mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = 2048, n_mels: int = N_MELS) -> np.ndarray:
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)
    fftfreqs = np.linspace(0.0, float(sr) / 2.0, int(1 + n_fft // 2))
    mel_f = _mel_to_hz_slaney(
        np.linspace(_hz_to_mel_slaney(0.0)[0], _hz_to_mel_slaney(float(sr) / 2.0)[0], n_mels + 2)
    )
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


def _chroma_filterbank(sr: int = SAMPLE_RATE, n_fft: int = 2048, n_chroma: int = N_CHROMA) -> np.ndarray:
    freqs = np.linspace(0.0, float(sr) / 2.0, int(1 + n_fft // 2))
    wts = np.zeros((n_chroma, len(freqs)), dtype=np.float32)
    nz = freqs > 0
    pitch = 12.0 * np.log2(freqs[nz] / 440.0) + 69.0
    bins = np.mod(np.round(pitch), n_chroma).astype(int)
    for idx, b in zip(np.where(nz)[0], bins):
        wts[b, idx] = 1.0
    return wts


def _stft_power(y: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    y_pad = np.pad(y, int(n_fft // 2), mode="constant")
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    n_frames = 1 + (len(y_pad) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        y_pad,
        shape=(n_frames, n_fft),
        strides=(y_pad.strides[0] * hop_length, y_pad.strides[0]),
    )
    spec = np.fft.rfft(frames * window[None, :], n=n_fft, axis=1).T
    return (np.abs(spec) ** 2).astype(np.float32)


def power_to_db(S: np.ndarray, ref_max: bool = True) -> np.ndarray:
    if HAS_LIBROSA:
        return librosa.power_to_db(S, ref=np.max if ref_max else 1.0).astype(np.float32)
    S_safe = np.maximum(S, 1e-10)
    ref = float(np.max(S_safe)) if ref_max else 1.0
    log_spec = 10.0 * np.log10(S_safe) - 10.0 * np.log10(max(ref, 1e-10))
    return np.maximum(log_spec, log_spec.max() - 80.0).astype(np.float32)


def compute_mel_spectrogram_db(
    y: np.ndarray, sr: int = SAMPLE_RATE, n_mels: int = N_MELS, n_fft: int = 2048, hop_length: int = 512
) -> np.ndarray:
    """Return [n_mels, time] Mel spectrogram in dB."""
    if HAS_LIBROSA:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    power = _stft_power(y, n_fft=n_fft, hop_length=hop_length)
    mel = _mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels) @ power
    return power_to_db(mel, ref_max=True)


def compute_chroma(
    y: np.ndarray, sr: int = SAMPLE_RATE, n_fft: int = 2048, hop_length: int = 512, n_chroma: int = N_CHROMA
) -> np.ndarray:
    """Return [n_chroma, time] normalized chromagram."""
    if HAS_LIBROSA:
        return librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length).astype(np.float32)
    power = _stft_power(y, n_fft=n_fft, hop_length=hop_length)
    chroma = _chroma_filterbank(sr=sr, n_fft=n_fft, n_chroma=n_chroma) @ power
    norm = np.maximum(np.max(chroma, axis=0, keepdims=True), 1e-10)
    return (chroma / norm).astype(np.float32)


def compute_rms_energy(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Return 1D RMS energy per frame matching STFT frame count."""
    if HAS_LIBROSA:
        return librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0].astype(np.float32)
    y_pad = np.pad(np.asarray(y, dtype=np.float32), int(frame_length // 2), mode="constant")
    n_frames = 1 + (len(y_pad) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        y_pad,
        shape=(n_frames, frame_length),
        strides=(y_pad.strides[0] * hop_length, y_pad.strides[0]),
    )
    return np.sqrt(np.mean(frames ** 2, axis=1)).astype(np.float32)


def frames_to_time(n_frames: int, sr: int = SAMPLE_RATE, hop_length: int = 512) -> np.ndarray:
    if HAS_LIBROSA:
        return librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    return (np.arange(n_frames, dtype=np.float32) * hop_length) / float(sr)


def _decode_wav_fallback(raw_bytes: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode WAV bytes using scipy.io.wavfile or wave and resample to target_sr."""
    if not raw_bytes:
        raise ValueError("Empty audio input provided.")
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(io.BytesIO(raw_bytes))
        if data.dtype == np.int16:
            y = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            y = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            y = (data.astype(np.float32) - 128.0) / 128.0
        else:
            y = data.astype(np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    except Exception:
        try:
            with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                raw_frames = wf.readframes(wf.getnframes())
                if sampwidth == 2:
                    y = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    y = np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    y = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                if n_channels > 1:
                    y = y.reshape(-1, n_channels).mean(axis=1)
        except Exception as exc:
            raise ValueError(
                "Could not decode audio bytes. Ensure the file is a valid audio recording."
            ) from exc
    if sr != target_sr and len(y) > 0:
        g = gcd(int(target_sr), int(sr))
        y = resample_poly(y, int(target_sr) // g, int(sr) // g).astype(np.float32)
    return y.astype(np.float32)


def load_audio_bytes(raw_bytes: bytes, sr: int = SAMPLE_RATE, max_seconds: float = MAX_SECONDS) -> np.ndarray:
    """Decode audio bytes, apply speech preprocessing (DC removal, silence trim, peak norm), and pad/trim to 4.0s."""
    if not raw_bytes:
        raise ValueError("Empty audio payload.")
    y = None
    if HAS_LIBROSA:
        try:
            y, _ = librosa.load(io.BytesIO(raw_bytes), sr=sr, mono=True)
        except Exception:
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".audio")
                os.close(fd)
                Path(tmp_path).write_bytes(raw_bytes)
                y, _ = librosa.load(tmp_path, sr=sr, mono=True)
            except Exception:
                y = None
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    if y is None:
        y = _decode_wav_fallback(raw_bytes, target_sr=sr)
    y_clean, _ = preprocess_waveform(y, sr=sr, max_seconds=max_seconds)
    return y_clean


def extract_features_from_waveform(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return [time, 116] float32 standardized features from a mono waveform after preprocessing."""
    y_proc, active_frames = preprocess_waveform(y, sr=sr, max_seconds=MAX_SECONDS)

    if HAS_LIBROSA:
        mfcc = librosa.feature.mfcc(y=y_proc, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
        mel = librosa.feature.melspectrogram(y=y_proc, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        chroma = librosa.feature.chroma_stft(y=y_proc, sr=sr, n_fft=2048, hop_length=512)
    else:
        power = _stft_power(y_proc, n_fft=2048, hop_length=512)
        mel = _mel_filterbank(sr=sr, n_fft=2048, n_mels=N_MELS) @ power
        mel_db = power_to_db(mel, ref_max=True)
        mel_db_mfcc = power_to_db(mel, ref_max=False)
        mfcc = dct(mel_db_mfcc, type=2, axis=0, norm="ortho")[:N_MFCC].astype(np.float32)
        chroma_raw = _chroma_filterbank(sr=sr, n_fft=2048, n_chroma=N_CHROMA) @ power
        chroma = (chroma_raw / np.maximum(np.max(chroma_raw, axis=0, keepdims=True), 1e-10)).astype(np.float32)

    feat = np.concatenate([mfcc, mel_db, chroma], axis=0).T.astype(np.float32)
    if feat.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"Expected feature dimension {FEATURE_DIM}, got {feat.shape[1]}")
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize using active (non-padded) speech frames so trailing zero-padding
    # does not skew the utterance mean/std, and zero out padded tail frames.
    ref_slice = feat[:active_frames] if active_frames >= 4 else feat
    mean = ref_slice.mean(axis=0, keepdims=True)
    std = ref_slice.std(axis=0, keepdims=True)
    norm_feat = ((feat - mean) / np.maximum(std, 1e-5)).astype(np.float32)
    if active_frames < norm_feat.shape[0]:
        norm_feat[active_frames:] = 0.0
    return norm_feat


def extract_features(path: str | Path) -> np.ndarray:
    """Return [time, 116] float32 features; preprocess and pad/trim audio to four seconds."""
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Audio file not found: {path_obj}")
    if HAS_LIBROSA:
        y, _ = librosa.load(path_obj, sr=SAMPLE_RATE, mono=True)
        return extract_features_from_waveform(y, sr=SAMPLE_RATE)
    y = load_audio_bytes(path_obj.read_bytes(), sr=SAMPLE_RATE, max_seconds=MAX_SECONDS)
    return extract_features_from_waveform(y, sr=SAMPLE_RATE)


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
    xs, ys, actors, lengths, paths = [], [], [], [], []
    for i, path in enumerate(files, 1):
        try:
            if HAS_LIBROSA:
                raw_y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            else:
                raw_y = _decode_wav_fallback(path.read_bytes(), target_sr=SAMPLE_RATE)
            _, act_len = preprocess_waveform(raw_y, sr=SAMPLE_RATE, max_seconds=MAX_SECONDS)
            xs.append(extract_features_from_waveform(raw_y, sr=SAMPLE_RATE))
            ys.append(emotion_from_path(path))
            actors.append(actor_id(path))
            lengths.append(act_len)
            paths.append(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed processing {path}: {exc}") from exc
        if i % 100 == 0:
            print(f"Processed {i}/{len(files)}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=np.asarray(xs),
        y=np.asarray(ys, dtype=np.int64),
        actors=np.asarray(actors, dtype=np.int64),
        lengths=np.asarray(lengths, dtype=np.int64),
        paths=np.asarray(paths),
    )
    print(f"Saved {len(xs)} preprocessed utterances to {output}")
