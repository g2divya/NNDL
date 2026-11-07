# Project Report: Attention-Based BiLSTM for Speech Emotion Recognition

## Abstract

This project evaluates whether a learned additive temporal attention layer improves speech emotion recognition (SER) over unidirectional and bidirectional LSTM baselines. RAVDESS speech audio is represented using frame-level MFCC (40), log-mel spectrogram (64), and chroma (12) features (116 dimensions per frame across 173 frames). Three models are trained from scratch using an actor-disjoint split (`seed=42`), and held-out performance is measured on 240 utterances from unseen speakers excluded from training and validation. Adding temporal attention yields **41.67% Top-1 Accuracy** (`62.08%` Top-2, `72.50%` Top-3) and **40.34% Macro F1** (`42.92%` Accuracy / `43.06%` Macro F1 with prior calibration), outperforming the unidirectional LSTM (**29.58% Accuracy / 27.64% Macro F1**) and BiLSTM (**26.67% Accuracy / 23.18% Macro F1**) baselines by **+12.08%** and **+15.00%** absolute accuracy, respectively.

## 1. Introduction and Objective

Speech Emotion Recognition (SER) must aggregate emotional cues that are distributed unevenly over time—voiced vowels, prosodic pitch inflections, and energy bursts carry strong affective signal, whereas leading and trailing silence carry none. In fixed-length padded utterances (4.0 seconds), standard RNN baselines that rely solely on the final hidden state suffer from information decay over trailing zero-padded frames. This project tests whether an additive Bahdanau-style temporal attention mechanism over a 2-layer BiLSTM resolves this bottleneck on an actor-independent benchmark.

## 2. Dataset

- **Dataset**: [RAVDESS](https://zenodo.org/records/1188976) (Ryerson Audio-Visual Database of Emotional Speech and Song), audio-only speech (`03-01-*.wav`).
- **Number of utterances used**: 1,440 utterances across 24 actors (240 held-out test utterances from 4 unseen actors).
- **Classes (8)**: `neutral` (support 16 in test), `calm` (32), `happy` (32), `sad` (32), `angry` (32), `fearful` (32), `disgust` (32), `surprised` (32).
- **Split method**: Actor-disjoint `GroupShuffleSplit` (~70% train / 15% validation / 15% test, `seed=42`), ensuring zero speaker overlap across splits.

## 3. Preprocessing and Features

Each audio file is resampled to 22,050 Hz mono and padded or trimmed to 4.0 seconds (88,200 samples, yielding 173 STFT frames with `n_fft=2048` and `hop_length=512`). Per frame, 40 MFCCs, 64 log-mel filterbank energies (dB), and 12 pitch-class chroma features are concatenated into a `[173, 116]` tensor and standardized per utterance (`mean=0, std=1`) to prevent speaker-level amplitude leakage.

## 4. Models and Training

All three models use a 2-layer recurrent backbone (`hidden=128`, `dropout=0.3`):
- **LSTM**: Unidirectional 2-layer LSTM (`128-d` final hidden state $h_T$).
- **BiLSTM**: Bidirectional 2-layer LSTM (`256-d` concatenated final states $[h_T^{\rightarrow}; h_1^{\leftarrow}]$).
- **Attention-BiLSTM**: Computes frame-wise additive attention scores $e_t = v^\top \tanh(W h_t + b)$, normalizes temporal weights $\alpha_t = \text{softmax}(e_t)$, and forms the context vector $c = \sum_{t=1}^T \alpha_t h_t \in \mathbb{R}^{256}$ passed to the dropout and linear classification head.

## 5. Held-Out Evaluation Results (240 Test Utterances)

| Model | Top-1 Accuracy | Top-2 Accuracy | Top-3 Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LSTM** | 29.58% | 45.00% | 59.58% | 32.81% | 30.47% | 27.64% | 27.72% |
| **BiLSTM** | 26.67% | 45.00% | 57.08% | 28.03% | 25.00% | 23.18% | 24.72% |
| **Attention-BiLSTM** | **41.67%** | **62.08%** | **72.50%** | **47.79%** | **40.23%** | **40.34%** | **41.22%** |
| **Attention-BiLSTM (Prior-Calibrated)** | **42.92%** | **62.50%** | **72.92%** | **46.85%** | **42.97%** | **43.06%** | **43.41%** |

### Per-Class Performance (Attention-BiLSTM)

| Emotion Class | Precision | Recall | F1-Score | Test Support |
|---|---:|---:|---:|---:|
| **Angry** | 54.55% | 56.25% | **55.38%** | 32 |
| **Calm** | 51.52% | 53.12% | **52.31%** | 32 |
| **Disgust** | 55.56% | 46.88% | **50.85%** | 32 |
| **Sad** | 25.58% | 68.75% | **37.29%** | 32 |
| **Happy** | 66.67% | 25.00% | **36.36%** | 32 |
| **Surprised** | 43.48% | 31.25% | **36.36%** | 32 |
| **Neutral** | 50.00% | 18.75% | **27.27%** | 16 |
| **Fearful** | 35.00% | 21.88% | **26.92%** | 32 |

## 6. Key Findings & Diagnostic Analysis

1. **Why Attention-BiLSTM Outperforms Baselines by +12–15% Accuracy**: Because utterances are padded to 4.0 seconds, the final frames ($t \approx 4.0\text{s}$) are predominantly silent padding. Unidirectional LSTM and BiLSTM pool final time-step hidden states that are diluted by trailing silence, whereas Additive Attention learns to assign peak weights $\alpha_t$ to active voiced syllables in the middle of the clip.
2. **Spotlight Wins**: On **38 out of 240 held-out utterances**, Attention-BiLSTM predicts the exact ground-truth emotion while *both* LSTM and BiLSTM fail simultaneously.
3. **Class Imbalance & Low-Arousal Confusion**: In RAVDESS, `neutral` has half the training samples of other emotions (16 vs. 32 in test), and shares low-energy acoustic profiles with `sad` and `calm`. Applying class-balanced loss weights during training or prior-calibrated inference (`probs / prior^0.7`) lifts Macro F1 from **40.34%** to **43.06%**.
