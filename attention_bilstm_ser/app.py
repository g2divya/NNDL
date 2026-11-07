"""Single-Page Minimalist Dashboard (Apple / Edgebook Inspired) for Attention-BiLSTM SER."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.interpolate import make_interp_spline

from src.config import ARTIFACT_DIR, LABELS, SAMPLE_RATE, MAX_SECONDS
from src.features import (
    compute_rms_energy,
    extract_features_from_waveform,
    frames_to_time,
    load_audio_bytes,
)
from src.models import load_inference_model

ROOT_DIR = Path(__file__).resolve().parent
DEMO_DIR = ROOT_DIR / "demo_audio"
CSS_PATH = ROOT_DIR / "style.css"
DEMO_FREQUENCIES = np.array([220, 260, 330, 390, 470, 560, 670, 800], dtype=float)

# Crisp Apple SF-Symbol-Style Inline SVGs
SVG_MIC = (
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>'
    '<path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>'
)

SVG_UPLOAD = (
    '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
    '<polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>'
)

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="〰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if CSS_PATH.is_file():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Model & Artifact Loaders (Graceful When Artifacts Are Missing)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_models(mtimes: tuple):
    loaded = {}
    for k in ("attention_bilstm", "bilstm", "lstm"):
        pt = ARTIFACT_DIR / f"{k}.pt"
        if pt.is_file():
            try:
                model = load_inference_model(pt)
                if model is not None:
                    loaded[k] = model
            except Exception:
                pass
    return loaded


@st.cache_data
def get_measured_metrics(comp_mtime: float | None):
    """Load measured held-out metrics strictly from saved evaluation artifacts."""
    comp_path = ARTIFACT_DIR / "comparison.json"
    if not comp_path.is_file():
        return {}
    try:
        rows = json.loads(comp_path.read_text(encoding="utf-8"))
        return {row["model"]: row for row in rows if "model" in row}
    except Exception:
        return {}


def demo_probabilities(y: np.ndarray) -> np.ndarray:
    """Tone-frequency lookup for synthetic fixtures only; never used as a SER model."""
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    frequencies = np.fft.rfftfreq(len(y), 1.0 / SAMPLE_RATE)
    band = (frequencies >= 180) & (frequencies <= 900)
    pitch = frequencies[band][np.argmax(spectrum[band])]
    scores = -((DEMO_FREQUENCIES - pitch) / 35.0) ** 2
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    return probs.astype(np.float32)


def plot_edgebook_curve(x: np.ndarray, y_vals: np.ndarray, unit_label: str = "", y_fmt: str = "{:.2f}"):
    """Render a minimal warm-ivory spline chart matching the Edgebook reference UI."""
    fig, ax = plt.subplots(figsize=(10.5, 2.75), dpi=160)
    fig.patch.set_facecolor("#FAF8F4")
    ax.set_facecolor("#FAF8F4")

    if len(x) > 6:
        idx_sub = np.linspace(0, len(x) - 1, min(len(x), 70)).astype(int)
        x_s, y_s = x[idx_sub], y_vals[idx_sub]
        x_dense = np.linspace(x_s.min(), x_s.max(), 320)
        try:
            spline = make_interp_spline(x_s, y_s, k=3)
            y_dense = np.clip(spline(x_dense), y_vals.min(), y_vals.max() * 1.05)
        except Exception:
            x_dense, y_dense = x, y_vals
    else:
        x_dense, y_dense = x, y_vals

    ax.plot(x_dense, y_dense, color="#2A6F4E", linewidth=1.8, zorder=4)
    ax.fill_between(x_dense, y_dense, y_dense.min(), color="#2A6F4E", alpha=0.09, zorder=2)

    peak_i = int(np.argmax(y_dense))
    px, py = float(x_dense[peak_i]), float(y_dense[peak_i])
    ax.scatter([px], [py], s=26, facecolor="#FAF8F4", edgecolor="#2A6F4E", linewidth=1.5, zorder=5)
    ax.annotate(
        f"Peak {px:.2f}s",
        xy=(px, py),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=7.8,
        color="#857F74",
        fontweight="500",
    )
    ax.scatter([x_dense[-1]], [y_dense[-1]], s=46, color="#2A6F4E", zorder=6)

    y_min, y_max = float(np.min(y_dense)), float(np.max(y_dense))
    span = max(y_max - y_min, 1e-4)
    ax.set_ylim(y_min - 0.08 * span, y_max + 0.22 * span)
    ax.set_xlim(float(x_dense[0]), float(x_dense[-1]) * 1.02)

    for level in (y_min + 0.1 * span, y_max * 0.92):
        ax.axhline(level, color="#EBE6DC", linewidth=0.8, zorder=1)

    ax.yaxis.tick_right()
    ax.set_yticks([y_min + 0.1 * span, y_max * 0.92])
    ax.set_yticklabels(
        [y_fmt.format(y_min + 0.1 * span) + unit_label, y_fmt.format(y_max * 0.92) + unit_label],
        color="#857F74",
        fontsize=7.8,
    )

    xticks = np.linspace(float(x_dense[0]), float(x_dense[-1]), 4)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.1f}s" for t in xticks], color="#857F74", fontsize=7.8)
    ax.tick_params(axis="both", length=0, pad=6)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0.25)
    return fig


def split_num_parts(val: float) -> tuple[str, str]:
    whole = int(val)
    dec = int(round((val - whole) * 10))
    if dec == 10:
        whole += 1
        dec = 0
    return f"{whole}", f".{dec}%"


# ---------------------------------------------------------------------------
# Load Models & Measured Evaluation State
# ---------------------------------------------------------------------------
mtimes = tuple(
    (ARTIFACT_DIR / f"{k}.pt").stat().st_mtime if (ARTIFACT_DIR / f"{k}.pt").is_file() else 0.0
    for k in ("attention_bilstm", "bilstm", "lstm")
)
models = get_models(mtimes)
primary_model = models.get("attention_bilstm")

comp_file = ARTIFACT_DIR / "comparison.json"
measured = get_measured_metrics(comp_file.stat().st_mtime if comp_file.is_file() else None)

if "microphone_recording_version" not in st.session_state:
    st.session_state.microphone_recording_version = 0
if "last_active_source" not in st.session_state:
    st.session_state.last_active_source = "sample"
if "last_mic_hash" not in st.session_state:
    st.session_state.last_mic_hash = None
if "last_upload_hash" not in st.session_state:
    st.session_state.last_upload_hash = None

# ---------------------------------------------------------------------------
# 1. Minimal Header ("Hi.")
# ---------------------------------------------------------------------------
status_label = "Attention BiLSTM Ready" if primary_model is not None else "Demo Mode · Checkpoint Missing"
st.markdown(
    f"""
    <div class="eb-topbar">
        <h1 class="eb-greeting">Hi.</h1>
        <span class="eb-session-pill"><span class="eb-dot"></span> {status_label}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 2. Direct Apple-Icon Input Dock (Record Voice & Upload Audio Side-by-Side)
# ---------------------------------------------------------------------------
dock_left, dock_right = st.columns(2, gap="medium")

with dock_left:
    with st.container(border=True):
        top_c1, top_c2 = st.columns([1.8, 1.1])
        with top_c1:
            st.markdown(
                f"""
                <div class="apple-input-header">
                    <span class="apple-input-title">
                        <span class="apple-icon-badge active">{SVG_MIC}</span>
                        <span>Record Voice</span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with top_c2:
            if st.button("New recording", key="record_new_microphone_sample", use_container_width=True):
                st.session_state.microphone_recording_version += 1
                st.session_state.last_mic_hash = None
                st.session_state.last_active_source = "sample"
                st.rerun()

        mic_audio = st.audio_input(
            "Record voice",
            key=f"microphone_recording_{st.session_state.microphone_recording_version}",
            label_visibility="collapsed",
        )

with dock_right:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="apple-input-header">
                <span class="apple-input-title">
                    <span class="apple-icon-badge">{SVG_UPLOAD}</span>
                    <span>Upload Audio</span>
                </span>
                <span class="apple-input-tag">WAV · MP3 · M4A · FLAC · OGG</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload audio file",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            label_visibility="collapsed",
        )

# Determine active audio source
mic_bytes = mic_audio.getvalue() if mic_audio is not None else None
upload_bytes = uploaded_file.getvalue() if uploaded_file is not None else None

if mic_bytes is not None:
    mh = hashlib.sha256(mic_bytes).hexdigest()[:8]
    if mh != st.session_state.last_mic_hash:
        st.session_state.last_mic_hash = mh
        st.session_state.last_active_source = "mic"

if upload_bytes is not None:
    uh = hashlib.sha256(upload_bytes).hexdigest()[:8]
    if uh != st.session_state.last_upload_hash:
        st.session_state.last_upload_hash = uh
        st.session_state.last_active_source = "upload"

is_demo_sample = False
raw_audio = None
source_tag = ""

if st.session_state.last_active_source == "mic" and mic_bytes is not None:
    raw_audio = mic_bytes
    source_tag = f"LIVE MICROPHONE · #{st.session_state.last_mic_hash.upper()}"
elif st.session_state.last_active_source == "upload" and upload_bytes is not None:
    raw_audio = upload_bytes
    source_tag = f"UPLOADED AUDIO · {uploaded_file.name.upper()}"
elif mic_bytes is not None:
    raw_audio = mic_bytes
    source_tag = f"LIVE MICROPHONE · #{hashlib.sha256(mic_bytes).hexdigest()[:8].upper()}"
elif upload_bytes is not None:
    raw_audio = upload_bytes
    source_tag = f"UPLOADED AUDIO · {uploaded_file.name.upper()}"
else:
    is_demo_sample = True
    sample_files = sorted(DEMO_DIR.glob("demo_*.wav")) if DEMO_DIR.is_dir() else []
    if sample_files:
        default_sample = DEMO_DIR / "demo_calm.wav"
        chosen_sample = default_sample if default_sample.is_file() else sample_files[0]
        raw_audio = chosen_sample.read_bytes()
        source_tag = f"SYNTHETIC UI FIXTURE ({chosen_sample.name.upper()}) · RECORD OR UPLOAD SPEECH ABOVE"
    else:
        st.info("Record your voice or upload an audio file above to run emotion recognition.")

# ---------------------------------------------------------------------------
# 3. Single Hero Card + Compact 8-Emotion Strip
# ---------------------------------------------------------------------------
if raw_audio is not None:
    try:
        y_wave = load_audio_bytes(raw_audio, sr=SAMPLE_RATE, max_seconds=MAX_SECONDS)
        feat = extract_features_from_waveform(y_wave, sr=SAMPLE_RATE)

        if primary_model is None:
            weights = None
            if is_demo_sample:
                probs = demo_probabilities(y_wave)
                st.warning(
                    "No trained checkpoint found at `artifacts/attention_bilstm.pt`. "
                    "Showing synthetic tone-template scores for UI verification only (not an emotion prediction)."
                )
            else:
                probs = None
                st.warning(
                    "No trained SER model is available yet (`artifacts/attention_bilstm.pt` not found). "
                    "Audio visualizations are shown below; run `python prepare_data.py` and `python train.py` for model predictions."
                )
        else:
            if is_demo_sample:
                probs = demo_probabilities(y_wave)
                _, _, weights = primary_model.predict(feat, return_attention=True)
            else:
                _, probs, weights = primary_model.predict(feat, return_attention=True)

        p_bi = models["bilstm"].predict(feat, return_attention=False)[1] if "bilstm" in models else None
        p_ls = models["lstm"].predict(feat, return_attention=False)[1] if "lstm" in models else None

        rms_vals = compute_rms_energy(y_wave)
        att_curve = weights if weights is not None else rms_vals
        t_axis = frames_to_time(len(att_curve), sr=SAMPLE_RATE, hop_length=512)
        peak_time = float(t_axis[int(np.argmax(att_curve))])

        # Measured held-out metrics from comparison.json (never fabricated)
        att_m = measured.get("attention_bilstm")
        bi_m = measured.get("bilstm")
        ls_m = measured.get("lstm")
        if att_m is not None:
            att_acc_val = float(att_m["accuracy"]) * 100.0
            att_f1_val = float(att_m["macro_f1"]) * 100.0
            acc_str = f"{att_acc_val:.1f}%"
            acc_sub = f"Macro F1: {att_f1_val:.1f}% (RAVDESS test)"
            acc_bar_w = f"{min(max(att_acc_val, 0.0), 100.0):.0f}%"
        else:
            acc_str = "N/A"
            acc_sub = "Run train.py for test metrics"
            acc_bar_w = "0%"

        if att_m is not None and bi_m is not None:
            gain_val = (float(att_m["accuracy"]) - float(bi_m["accuracy"])) * 100.0
            gain_str = f"{gain_val:+.1f}%"
        else:
            gain_str = "N/A"

        bi_top = LABELS[int(np.argmax(p_bi))].title() if p_bi is not None else "N/A"
        ls_top = LABELS[int(np.argmax(p_ls))].title() if p_ls is not None else "N/A"

        with st.container(border=True):
            h_left, h_right = st.columns([2.2, 1.0])
            with h_left:
                if probs is not None:
                    top_idx = int(np.argmax(probs))
                    top_emotion = LABELS[top_idx].title()
                    top_pct = float(probs[top_idx]) * 100.0
                    whole_str, dec_str = split_num_parts(top_pct)
                    second_idx = int(np.argsort(probs)[-2])
                    margin = top_pct - float(probs[second_idx]) * 100.0
                    metric_kind = "Tone-template match" if is_demo_sample else "Model confidence"
                    st.markdown(
                        f"""
                        <div class="eb-micro-label">{source_tag}</div>
                        <div class="eb-hero-number">{top_emotion} &nbsp;{whole_str}<span class="eb-hero-decimal">{dec_str}</span></div>
                        <div class="eb-hero-delta">
                            {metric_kind} · +{margin:.1f}% margin <span class="muted">over {LABELS[second_idx].title()} · Peak focus at {peak_time:.2f}s</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    top_idx = -1
                    top_emotion = "Unavailable"
                    top_pct = 0.0
                    st.markdown(
                        f"""
                        <div class="eb-micro-label">{source_tag}</div>
                        <div class="eb-hero-number">Prediction Unavailable</div>
                        <div class="eb-hero-delta">
                            <span class="muted">Train the model checkpoint to enable emotion classification · Peak energy at {peak_time:.2f}s</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with h_right:
                chart_mode = st.radio(
                    "Signal View",
                    ["Attention", "Energy", "Waveform"],
                    horizontal=True,
                    label_visibility="collapsed",
                )

            if chart_mode == "Attention":
                fig = plot_edgebook_curve(t_axis, att_curve * 100.0, unit_label="%", y_fmt="{:.1f}")
            elif chart_mode == "Energy":
                t_r = frames_to_time(len(rms_vals), sr=SAMPLE_RATE, hop_length=512)
                fig = plot_edgebook_curve(t_r, rms_vals / max(float(rms_vals.max()), 1e-6) * 100.0, unit_label="%", y_fmt="{:.0f}")
            else:
                env = np.abs(y_wave)
                t_w = np.linspace(0.0, MAX_SECONDS, len(env))
                fig = plot_edgebook_curve(t_w, env, unit_label="", y_fmt="{:.2f}")

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            conf_str = f"{top_pct:.1f}%" if probs is not None else "N/A"
            conf_sub = ("Template score (UI fixture)" if is_demo_sample else f"Top-1 · {top_emotion}") if probs is not None else "No checkpoint"
            focus_sub = "Attention α_t frame" if weights is not None else "Frame RMS energy"

            st.markdown(
                f"""
                <div class="eb-stat-strip" style="margin: 6px -18px -16px -18px;">
                    <div class="eb-stat-cell">
                        <div class="eb-micro-label">CONFIDENCE</div>
                        <div class="eb-stat-val sage">{conf_str}</div>
                        <div class="eb-stat-sub">{conf_sub}</div>
                    </div>
                    <div class="eb-stat-cell">
                        <div class="eb-micro-label">PEAK FOCUS</div>
                        <div class="eb-stat-val sage">{peak_time:.2f}s</div>
                        <div class="eb-stat-sub">{focus_sub}</div>
                    </div>
                    <div class="eb-stat-cell">
                        <div class="eb-micro-label">HELD-OUT ACCURACY</div>
                        <div class="eb-stat-val">{acc_str}</div>
                        <div class="eb-stat-sub">{acc_sub}</div>
                        <div class="eb-progress-track"><div class="eb-progress-fill" style="width:{acc_bar_w};"></div></div>
                    </div>
                    <div class="eb-stat-cell">
                        <div class="eb-micro-label">VS. BILSTM BASELINE</div>
                        <div class="eb-stat-val">{gain_str}</div>
                        <div class="eb-stat-sub">BiLSTM: {bi_top} · LSTM: {ls_top}</div>
                        <div class="eb-progress-track"><div class="eb-progress-fill bronze" style="width:{acc_bar_w};"></div></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if probs is not None:
            rank_order = {int(idx): r + 1 for r, idx in enumerate(np.argsort(probs)[::-1])}
            cells_html = []
            for i, lbl in enumerate(LABELS):
                p_val = float(probs[i]) * 100.0
                cls_name = "eb-cell active" if i == top_idx else "eb-cell"
                cells_html.append(
                    f'<div class="{cls_name}">'
                    f'  <div class="eb-cell-top">'
                    f'    <span class="eb-cell-name">{lbl}</span>'
                    f'    <span class="eb-rank-badge">{rank_order[i]}</span>'
                    f'  </div>'
                    f'  <div class="eb-cell-val">{p_val:.1f}%</div>'
                    f'</div>'
                )

            matrix_label = (
                "SYNTHETIC TONE-TEMPLATE SCORES · 8 CLASSES (UI CHECK ONLY)"
                if is_demo_sample
                else "CLASS PROBABILITIES · 8 EMOTIONS"
            )
            st.markdown(
                f"""
                <div class="eb-card" style="padding: 16px 20px;">
                    <div class="eb-micro-label">{matrix_label}</div>
                    <div class="eb-grid-8">
                        {''.join(cells_html)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception as exc:
        st.error(f"Could not process this audio: {exc}")
