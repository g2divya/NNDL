# Project Report: Attention-Based BiLSTM for Speech Emotion Recognition

> Fill the result fields from the generated artifacts after training. Do not report unmeasured performance values.

## Abstract

This project evaluates whether a learned temporal attention layer improves speech emotion recognition over unidirectional and bidirectional LSTM baselines. RAVDESS speech audio is represented using MFCC, log-mel, and chroma features. Three models are trained from scratch, and held-out performance is measured on speakers excluded from training and validation.

## 1. Introduction and objective

Explain the SER task, motivation for modeling temporal context, project objectives, and the intended scope. Clarify that this is a dataset benchmark and not a clinical system.

## 2. Dataset

- Dataset: [RAVDESS](https://zenodo.org/records/1188976), audio-only speech.
- Number of utterances used: **[fill from prepare_data output]**
- Classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised.
- Split method: actor-disjoint grouped split, seed **[fill]**.
- Training/validation/test actor IDs: **[record from training run]**.

Discuss the controlled recording conditions and limits to generalization.

## 3. Preprocessing and features

Describe mono resampling to 22,050 Hz, four-second pad/trim, hop length 512, and per-utterance normalization. The frame representation concatenates 40 MFCC, 64 log-mel, and 12 chroma coefficients. State any actual software versions and hardware used.

## 4. Models and training

Describe the two-layer LSTM and BiLSTM baselines. For attention-BiLSTM, report the additive attention formula `e_t = vᵀ tanh(W h_t)`, `α_t = softmax(e_t)`, and context `c = Σ α_t h_t`. The learned context vector is input to the classifier. Record hidden size, dropout, optimizer, learning rate, batch size, epoch limit, early stopping patience, and seed.

## 5. Evaluation protocol

The test set contains actor groups excluded from training and validation. Report accuracy and per-class precision, recall, and F1, plus macro F1 and confusion matrices. Explain that the small speaker count makes actor-disjoint estimates variable.

## 6. Results

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| LSTM | [fill] | [fill] | [fill] | [fill] |
| BiLSTM | [fill] | [fill] | [fill] | [fill] |
| Attention-BiLSTM | [fill] | [fill] | [fill] | [fill] |

Insert the generated training curves and confusion matrices from `artifacts/`. Discuss per-class errors and whether attention changed the results; do not assume it improves every class.

## 7. Attention inspection

Show representative attention maps alongside the corresponding waveform or spectrogram. Describe these as model weighting visualizations, not causal explanations or guaranteed human-interpretable evidence.

## 8. Limitations and future work

Discuss dataset size, acted emotion, language/accent and recording-domain limits, speaker variability, class ambiguity, and the difference between acted labels and real-world affect. Possible future work includes repeated actor-group splits, confidence calibration, noise robustness, and evaluation on another corpus.

## 9. Conclusion

Summarize measured findings and answer the objective using the actual held-out metrics.

## Reproducibility appendix

Include the source revision, Python/package versions, command line, seed, split actors, and the generated `comparison.json` and `*_metrics.json` artifacts.
