"""Focused unit and integration tests for the Speech Emotion Recognition pipeline."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch

from src.config import (
    ARTIFACT_DIR,
    FEATURE_DIM,
    LABELS,
    MAX_SECONDS,
    N_CHROMA,
    N_MELS,
    N_MFCC,
    SAMPLE_RATE,
    SEED,
)
from src.features import (
    actor_id,
    compute_chroma,
    compute_mel_spectrogram_db,
    compute_rms_energy,
    emotion_from_path,
    extract_features,
    extract_features_from_waveform,
    frames_to_time,
    load_audio_bytes,
)
from src.models import EmotionRNN, NumpyEmotionRNN, load_inference_model
from train import actor_disjoint_split, compute_balanced_weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "demo_audio"


class TestSERPipeline(unittest.TestCase):
    def test_config_specifications(self):
        self.assertEqual(len(LABELS), 8)
        self.assertEqual(
            LABELS,
            ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
        )
        self.assertEqual(SAMPLE_RATE, 22050)
        self.assertEqual(MAX_SECONDS, 4.0)
        self.assertEqual(N_MFCC, 40)
        self.assertEqual(N_MELS, 64)
        self.assertEqual(N_CHROMA, 12)
        self.assertEqual(FEATURE_DIM, 116)

    def test_ravdess_label_and_actor_parsing(self):
        for code, name in enumerate(LABELS, start=1):
            fname = Path(f"03-01-0{code}-01-01-01-14.wav")
            self.assertEqual(emotion_from_path(fname), code - 1)
            self.assertEqual(LABELS[emotion_from_path(fname)], name)
            self.assertEqual(actor_id(fname), 14)

        with self.assertRaises(ValueError):
            emotion_from_path(Path("03-01-09-01-01-01-01.wav"))
        with self.assertRaises(ValueError):
            emotion_from_path(Path("invalid_filename.wav"))
        with self.assertRaises(ValueError):
            actor_id(Path("bad_name.wav"))

    def test_audio_loading_and_resampling(self):
        sample_wav = DEMO_DIR / "demo_calm.wav"
        self.assertTrue(sample_wav.is_file(), "Expected demo_calm.wav fixture to exist")
        raw = sample_wav.read_bytes()
        y = load_audio_bytes(raw, sr=SAMPLE_RATE, max_seconds=MAX_SECONDS)
        self.assertEqual(y.dtype, np.float32)
        self.assertEqual(y.shape, (int(SAMPLE_RATE * MAX_SECONDS),))

        with self.assertRaises(ValueError):
            load_audio_bytes(b"")
        with self.assertRaises(ValueError):
            load_audio_bytes(b"not-a-valid-audio-header")

    def test_feature_extraction_and_normalization_consistency(self):
        sample_wav = DEMO_DIR / "demo_happy.wav"
        feat_from_path = extract_features(sample_wav)
        y = load_audio_bytes(sample_wav.read_bytes(), sr=SAMPLE_RATE, max_seconds=MAX_SECONDS)
        feat_from_wave = extract_features_from_waveform(y, sr=SAMPLE_RATE)

        self.assertEqual(feat_from_path.shape, (173, FEATURE_DIM))
        self.assertEqual(feat_from_wave.shape, (173, FEATURE_DIM))
        self.assertEqual(feat_from_path.dtype, np.float32)
        np.testing.assert_allclose(feat_from_path, feat_from_wave, rtol=1e-5, atol=1e-5)

        # Verify per-utterance zero-mean standardization
        col_means = np.abs(feat_from_wave.mean(axis=0))
        self.assertTrue(np.all(col_means < 1e-4))
        self.assertFalse(np.isnan(feat_from_wave).any())

        # Verify helper spectrograms
        mel_db = compute_mel_spectrogram_db(y)
        chroma = compute_chroma(y)
        rms = compute_rms_energy(y)
        times = frames_to_time(len(rms))
        self.assertEqual(mel_db.shape, (N_MELS, 173))
        self.assertEqual(chroma.shape, (N_CHROMA, 173))
        self.assertEqual(rms.shape, (173,))
        self.assertEqual(times.shape, (173,))

    def test_actor_disjoint_split_and_class_weights(self):
        rng = np.random.default_rng(SEED)
        actors = np.repeat(np.arange(1, 25), 10)
        X = rng.standard_normal((len(actors), 10, FEATURE_DIM)).astype(np.float32)
        y = np.tile(np.arange(len(LABELS)), len(actors) // len(LABELS))

        tr_idx, va_idx, te_idx = actor_disjoint_split(X, y, actors, seed=SEED)
        tr_actors = set(actors[tr_idx])
        va_actors = set(actors[va_idx])
        te_actors = set(actors[te_idx])

        self.assertEqual(len(tr_actors & va_actors), 0)
        self.assertEqual(len(tr_actors & te_actors), 0)
        self.assertEqual(len(va_actors & te_actors), 0)
        self.assertEqual(len(tr_idx) + len(va_idx) + len(te_idx), len(actors))

        partial_y = np.array([0, 0, 1, 2, 2, 2])
        weights = compute_balanced_weights(partial_y, num_classes=8)
        self.assertEqual(weights.shape, (8,))
        self.assertTrue(np.all(weights > 0))

    def test_checkpoint_loading_and_pytorch_equivalence(self):
        rng = np.random.default_rng(SEED)
        x_single = rng.standard_normal((173, FEATURE_DIM)).astype(np.float32)
        x_batch = rng.standard_normal((4, 173, FEATURE_DIM)).astype(np.float32)
        lengths = np.array([120, 140, 160, 173])

        for kind in ("lstm", "bilstm", "attention_bilstm"):
            pt_path = ARTIFACT_DIR / f"{kind}.pt"
            self.assertTrue(pt_path.is_file(), f"Missing checkpoint {pt_path}")

            # 1. Portable inference model
            model_np = load_inference_model(pt_path)
            self.assertIsNotNone(model_np)
            self.assertEqual(model_np.kind, kind)
            self.assertEqual(model_np.input_dim, FEATURE_DIM)
            self.assertEqual(model_np.labels, LABELS)

            logits_np, probs_np, weights_np = model_np.predict(x_single, return_attention=True)
            self.assertEqual(logits_np.shape, (8,))
            self.assertEqual(probs_np.shape, (8,))
            self.assertAlmostEqual(float(probs_np.sum()), 1.0, places=5)

            # 2. PyTorch EmotionRNN loaded from the same checkpoint must match bit-for-bit
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            model_pt = EmotionRNN(
                input_dim=ckpt["input_dim"],
                hidden=ckpt["hidden"],
                num_classes=len(ckpt["labels"]),
                kind=ckpt["kind"],
            )
            model_pt.load_state_dict(ckpt["state_dict"])
            model_pt.eval()
            with torch.no_grad():
                logits_pt, weights_pt = model_pt(torch.tensor(x_single), return_attention=True)
                probs_pt = torch.softmax(logits_pt, dim=1).squeeze(0).numpy()

            np.testing.assert_allclose(logits_np, logits_pt.squeeze(0).numpy(), rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(probs_np, probs_pt, rtol=1e-4, atol=1e-4)
            if kind == "attention_bilstm":
                self.assertIsNotNone(weights_np)
                np.testing.assert_allclose(weights_np, weights_pt.squeeze(0).numpy(), rtol=1e-4, atol=1e-4)
            else:
                self.assertIsNone(weights_np)
                self.assertIsNone(weights_pt)

            # 3. Batch + length masking test
            logits_b, probs_b, weights_b = model_np.predict(x_batch, return_attention=True, lengths=lengths)
            self.assertEqual(logits_b.shape, (4, 8))
            self.assertEqual(probs_b.shape, (4, 8))
            if kind == "attention_bilstm":
                self.assertEqual(weights_b.shape, (4, 173))
                self.assertAlmostEqual(float(weights_b[0, 120:].sum()), 0.0, places=5)

        with self.assertRaises(ValueError):
            NumpyEmotionRNN({"kind": "invalid_kind", "state_dict": {}})
        with self.assertRaises(ValueError):
            EmotionRNN(kind="invalid_kind")

    def test_training_step_and_checkpoint_roundtrip(self):
        rng = np.random.default_rng(SEED)
        X_dummy = torch.tensor(rng.standard_normal((8, 20, FEATURE_DIM)).astype(np.float32))
        y_dummy = torch.tensor(np.arange(8), dtype=torch.long)

        model = EmotionRNN(input_dim=FEATURE_DIM, hidden=32, num_classes=8, kind="attention_bilstm")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        optimizer.zero_grad()
        logits = model(X_dummy)
        loss = criterion(logits, y_dummy)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.isnan(loss).item())

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_file = Path(tmp_dir) / "temp_attention_bilstm.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "kind": "attention_bilstm",
                    "input_dim": FEATURE_DIM,
                    "hidden": 32,
                    "labels": LABELS,
                    "sample_rate": SAMPLE_RATE,
                },
                ckpt_file,
            )
            reloaded = load_inference_model(ckpt_file)
            self.assertIsNotNone(reloaded)
            _, probs, weights = reloaded.predict(X_dummy[0].numpy(), return_attention=True)
            self.assertEqual(probs.shape, (8,))
            self.assertEqual(weights.shape, (20,))

    def test_saved_evaluation_artifacts_integrity(self):
        comp_path = ARTIFACT_DIR / "comparison.json"
        self.assertTrue(comp_path.is_file())
        rows = json.loads(comp_path.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 3)
        for row in rows:
            kind = row["model"]
            met_path = ARTIFACT_DIR / f"{kind}_metrics.json"
            npz_path = ARTIFACT_DIR / f"{kind}_test.npz"
            self.assertTrue(met_path.is_file())
            self.assertTrue(npz_path.is_file())
            met = json.loads(met_path.read_text(encoding="utf-8"))
            npz = np.load(npz_path)
            self.assertAlmostEqual(row["accuracy"], met["accuracy"], places=6)
            self.assertAlmostEqual(row["macro_f1"], met["macro_f1"], places=6)
            self.assertEqual(npz["y"].shape, (240,))
            self.assertEqual(npz["pred"].shape, (240,))
            self.assertEqual(npz["probs"].shape, (240, 8))


if __name__ == "__main__":
    unittest.main()
