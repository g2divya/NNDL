# Attention-Based BiLSTM for Speech Emotion Recognition

An end-to-end, from-scratch RAVDESS speech emotion recognition project. The attention model learns a score for each BiLSTM output frame, normalizes those scores over time, and uses their weighted sum as the context vector passed to the emotion classifier. Its attention map therefore reflects the representation used to make the prediction.

## Dataset

Download the **RAVDESS Audio-Only Speech** dataset from its [official Zenodo record](https://zenodo.org/records/1188976) and extract it under `data/ravdess/`. The folder may contain `Actor_01`...`Actor_24` directories. This project uses the audio-only speech files (filename prefix `03-01-`) and all eight RAVDESS emotion codes: neutral, calm, happy, sad, angry, fearful, disgust, and surprised. Keep the downloaded dataset out of source control.

## Setup and run

Use Python 3.10 or newer.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python prepare_data.py
python train.py
streamlit run app.py
```

`prepare_data.py` writes compressed features to `artifacts/features.npz`. Each four-second utterance is resampled to 22,050 Hz and represented by concatenated 40 MFCC, 64 log-mel, and 12 chroma coefficients for each frame. Features are standardized per utterance. `train.py` trains three independently initialized models (unidirectional LSTM, BiLSTM, attention-BiLSTM) and saves checkpoints, per-model classification reports, confusion matrices, training curves, and a comparison JSON under `artifacts/`.

The local website accepts uploaded audio or a microphone recording. Start it with `streamlit run app.py`; allow microphone access in the browser when prompted. To start the website immediately, the trained checkpoint `artifacts/attention_bilstm.pt` must exist, so first run the dataset preparation and training commands above.

### Try the interface without RAVDESS

Run `python make_demo_data.py`, start the app, then choose **Try a sample**. This creates eight tiny synthetic tone clips so you can check playback, waveform, spectrogram, and the page flow. Without a trained checkpoint, uploaded or recorded audio still gets visualized but does not get an emotion prediction. The synthetic clips show a tone-template label for UI testing only; they are not speech, training data, or evidence of emotion recognition.

## Evaluation design

Splits are grouped by actor: 70% train, 15% validation, 15% test (approximately; indivisible actor groups make exact proportions impossible). Test speakers are not used for training or early stopping. Reported precision/recall/F1 are available per class in each `*_metrics.json`; `comparison.json` contains accuracy and macro F1. Results may vary slightly with software/hardware versions. The default seed is 42; override with `--seed`.

```bash
python train.py --epochs 50 --batch-size 32 --hidden 128 --seed 42
```

The Streamlit app accepts WAV, MP3, OGG, FLAC, and M4A audio. The upload is decoded and processed locally by the app. Attention is an inspection aid, not a causal explanation. The evaluated split is actor-independent and can be challenging; do not interpret a single accuracy value as general performance on every population or recording condition.

## Project layout

```text
app.py                 Streamlit inference and plots
prepare_data.py        Feature-cache generation
train.py               Training, held-out evaluation, and plots
src/config.py          Dataset and model settings
src/features.py        RAVDESS labels and acoustic features
src/models.py          LSTM, BiLSTM, and additive attention BiLSTM
artifacts/             Generated checkpoints and evaluation outputs
data/ravdess/          User-provided dataset (not included)
```

## Model

For attention-BiLSTM, the two-layer bidirectional LSTM produces a hidden vector at each frame. Additive attention computes `score_t = vᵀ tanh(W h_t)`, `alpha = softmax(score)`, and `context = Σ alpha_t h_t`. A dropout and linear classification head maps this context to eight emotion logits. The baselines use the last recurrent output without attention. All models are trained from scratch with AdamW and validation-loss early stopping.

## Reproducibility notes

- The actor split prevents speaker overlap; the selected actors are printed at the beginning of training as group counts.
- The cache stores source paths and actor IDs to make the split auditable.
- Audio is capped/padded to four seconds and resampled consistently for training and inference.
- For an academic report, include the exact dataset version, library versions, selected split seed, per-class scores, and the generated confusion matrices. Avoid claiming clinical or universal emotion recognition.
