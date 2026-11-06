# RNN-Based Attention Project

This project implements a compact sequence-to-sequence model with recurrent neural networks and attention. It is designed to be easy to run locally and to demonstrate the core idea behind attention in an RNN encoder-decoder setup.

## What this project does

The model learns to reverse short digit sequences using an encoder-decoder architecture with attention. It is a lightweight, educational alternative to a full transformer implementation and helps explain how attention works when applied to recurrent models.

## Files

- `rnn_attention_seq2seq.py` — the full training script and model definition
- `requirements.txt` — Python dependencies

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   . .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the project:
   ```bash
   python rnn_attention_seq2seq.py
   ```

## Notes

- The code runs on CPU by default and also works on CUDA if available.
- The task is intentionally synthetic, so no external dataset is required.
- This is a learning-focused implementation for RNN-based attention rather than a full transformer.
