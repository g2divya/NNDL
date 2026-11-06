"""Streamlit demo for trained attention-BiLSTM SER model."""
import io, json
from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from src.config import ARTIFACT_DIR, LABELS, SAMPLE_RATE, MAX_SECONDS
from src.features import extract_features
from src.models import EmotionRNN

DEMO_DIR = Path(__file__).resolve().parent / "demo_audio"
DEMO_FREQUENCIES = np.array([220, 260, 330, 390, 470, 560, 670, 800], dtype=float)

st.set_page_config(page_title="Attention BiLSTM SER", page_icon="🎙️", layout="wide")
st.title("Attention-Based BiLSTM for Speech Emotion Recognition")
st.caption("Upload a recording or speak into your microphone. Audio is processed locally by this app.")

@st.cache_resource
def load_model(checkpoint_mtime):
    path=ARTIFACT_DIR / "attention_bilstm.pt"
    if not path.exists(): return None
    checkpoint=torch.load(path, map_location="cpu", weights_only=False)
    model=EmotionRNN(input_dim=checkpoint["input_dim"], hidden=checkpoint["hidden"], num_classes=len(checkpoint["labels"]), kind=checkpoint["kind"])
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    return model

def audio_bytes_to_features(raw):
    # Decode using librosa to a temporary in-memory stream, then apply project feature extraction.
    y, _=librosa.load(io.BytesIO(raw), sr=SAMPLE_RATE, mono=True, duration=MAX_SECONDS)
    target=int(SAMPLE_RATE*MAX_SECONDS)
    y=np.pad(y[:target], (0,max(0,target-len(y))))
    import tempfile
    import os
    import soundfile as sf
    # Close the temporary WAV before librosa opens it; Windows does not allow
    # a second reader to access a NamedTemporaryFile while its handle is open.
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(tmp_path, y, SAMPLE_RATE)
        feat = extract_features(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return y, feat

def demo_probabilities(y):
    """Tone-frequency lookup for synthetic fixtures only; never used as a SER model."""
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    frequencies = np.fft.rfftfreq(len(y), 1 / SAMPLE_RATE)
    band = (frequencies >= 180) & (frequencies <= 900)
    pitch = frequencies[band][np.argmax(spectrum[band])]
    # Samples were generated with one known pitch per display label. This only validates UI flow.
    scores = -((DEMO_FREQUENCIES - pitch) / 35.0) ** 2
    probs = np.exp(scores - scores.max()); probs /= probs.sum()
    return probs

source=st.radio("Audio source", ["Upload a recording", "Record with microphone", "Try a sample"], horizontal=True)
if source == "Upload a recording":
    uploaded=st.file_uploader("Choose an audio file", type=["wav","mp3","ogg","flac","m4a"])
elif source == "Record with microphone":
    uploaded=st.audio_input("Record a short speech sample")
else:
    sample_names = [p.name for p in sorted(DEMO_DIR.glob("demo_*.wav"))] if DEMO_DIR.exists() else []
    if sample_names:
        selected = st.selectbox("Synthetic interface sample", sample_names, format_func=lambda name: name.removeprefix("demo_").removesuffix(".wav").title())
        sample_path = DEMO_DIR / selected
        st.audio(str(sample_path), format="audio/wav")
        uploaded = sample_path
        st.caption("These are synthetic tones, not RAVDESS speech. They only demonstrate the interface flow.")
    else:
        uploaded = None
        st.warning("No bundled demo clips found. Run `python make_demo_data.py` from the project folder.")

if uploaded:
    checkpoint_path=ARTIFACT_DIR / "attention_bilstm.pt"
    model=load_model(checkpoint_path.stat().st_mtime if checkpoint_path.exists() else None)
    demo_sample = source == "Try a sample"
    try:
        raw = uploaded.read() if isinstance(uploaded, Path) else uploaded.getvalue()
        y, feat=audio_bytes_to_features(raw)
        if model is None:
            weights = None
            if demo_sample:
                probs = demo_probabilities(y)
                st.warning("Demo mode: this synthetic tone label checks the interface only. It is not an emotion prediction.")
            else:
                probs = None
                st.warning("No trained SER model is available yet. Audio visualizations are shown below; run the RAVDESS preparation and training steps for real model predictions.")
        else:
            with torch.no_grad():
                logits, weights=model(torch.tensor(feat).unsqueeze(0), return_attention=True)
            probs=torch.softmax(logits,dim=1).squeeze(0).numpy()
        left,right=st.columns([1,2])
        with left:
            if probs is None:
                st.subheader("Prediction unavailable")
            elif demo_sample:
                idx=int(probs.argmax())
                st.subheader(f"Demo tone label: {LABELS[idx].title()}")
                st.metric("Tone-template match", f"{probs[idx]*100:.1f}%")
                st.bar_chart({LABELS[i].title():float(probs[i]) for i in range(len(LABELS))})
                st.caption("The bars are template-match scores, not emotion probabilities.")
            else:
                idx=int(probs.argmax())
                st.subheader(f"Prediction: {LABELS[idx].title()}")
                st.metric("Model confidence", f"{probs[idx]*100:.1f}%")
                st.bar_chart({LABELS[i].title():float(probs[i]) for i in range(len(LABELS))})
        with right:
            fig,ax=plt.subplots(figsize=(10,2.4)); librosa.display.waveshow(y,sr=SAMPLE_RATE,ax=ax); ax.set_title("Waveform"); st.pyplot(fig); plt.close(fig)
            mel=librosa.feature.melspectrogram(y=y,sr=SAMPLE_RATE,n_mels=64,n_fft=2048,hop_length=512)
            fig,ax=plt.subplots(figsize=(10,3)); librosa.display.specshow(librosa.power_to_db(mel,ref=np.max),sr=SAMPLE_RATE,hop_length=512,x_axis="time",y_axis="mel",ax=ax); ax.set_title("Mel spectrogram"); fig.colorbar(ax.collections[0],ax=ax,format="%+2.0f dB"); st.pyplot(fig); plt.close(fig)
        if weights is not None:
            values=weights.squeeze(0).numpy(); title="Temporal attention used by the classifier"
        else:
            # Energy trace is shown for UI inspection; it is not model attention.
            values=librosa.feature.rms(y=y,frame_length=2048,hop_length=512)[0]; title="Frame energy (demo only; not attention)"
        times=librosa.frames_to_time(np.arange(len(values)),sr=SAMPLE_RATE,hop_length=512)
        fig,ax=plt.subplots(figsize=(10,2.8)); ax.plot(times,values,color="#b23a48"); ax.fill_between(times,values,alpha=.25,color="#b23a48"); ax.set(xlabel="Time (s)",ylabel="Weight / energy",title=title); st.pyplot(fig); plt.close(fig)
        if weights is not None:
            st.caption("Attention weights show how the trained model distributes its temporal pooling across frames. They are useful for inspection, but do not by themselves establish a causal explanation.")
    except Exception as exc:
        st.error(f"Could not process this audio: {exc}")

st.divider(); st.subheader("Held-out evaluation")
comparison=ARTIFACT_DIR / "comparison.json"
if comparison.exists():
    rows=json.loads(comparison.read_text()); st.dataframe(rows, use_container_width=True, hide_index=True)
    cols=st.columns(3)
    for col, row in zip(cols, rows):
        image=ARTIFACT_DIR / f"{row['model']}_confusion.png"
        if image.exists(): col.image(str(image),caption=f"{row['model']} confusion matrix",use_container_width=True)
else:
    st.info("Train all three models to populate held-out comparison metrics and confusion matrices.")
