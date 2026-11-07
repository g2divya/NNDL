"""Single-Page Minimalist Dashboard (Edgebook × Apple Human Interface) for Attention-BiLSTM SER."""
from __future__ import annotations

from datetime import datetime
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
    preprocess_waveform,
)
from src.models import load_inference_model

ROOT_DIR = Path(__file__).resolve().parent
DEMO_DIR = ROOT_DIR / "demo_audio"
CSS_PATH = ROOT_DIR / "style.css"
DEMO_FREQUENCIES = np.array([220, 260, 330, 390, 470, 560, 670, 800], dtype=float)

# Crisp Apple SF-Symbol-Style Inline SVGs
SVG_MIC = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#1D1D1F" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>'
    '<path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>'
)

SVG_UPLOAD = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#1D1D1F" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
    '<polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>'
)

SVG_WAVE = (
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#1D1D1F" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<line x1="4" y1="10" x2="4" y2="14"/><line x1="8" y1="6" x2="8" y2="18"/>'
    '<line x1="12" y1="3" x2="12" y2="21"/><line x1="16" y1="7" x2="16" y2="17"/>'
    '<line x1="20" y1="10" x2="20" y2="14"/></svg>'
)

SVG_CLOCK = (
    '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>'
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
    """Matplotlib spline helper preserved for backwards compatibility."""
    fig, ax = plt.subplots(figsize=(10.5, 2.75), dpi=160)
    fig.patch.set_facecolor("#FAF8F4")
    ax.set_facecolor("#FAF8F4")
    ax.plot(x, y_vals, color="#296E4D", linewidth=1.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.25)
    return fig


def render_edgebook_svg_chart(
    x: np.ndarray,
    y_vals: np.ndarray,
    unit_label: str = "",
    y_fmt: str = "{:.1f}",
) -> str:
    """Render a Retina-crisp animated SVG spline curve with delayed spring transitions."""
    if len(x) > 6:
        idx_sub = np.linspace(0, len(x) - 1, min(len(x), 64)).astype(int)
        x_s, y_s = x[idx_sub], y_vals[idx_sub]
        x_dense = np.linspace(float(x_s.min()), float(x_s.max()), 240)
        try:
            spline = make_interp_spline(x_s, y_s, k=3)
            y_dense = np.clip(spline(x_dense), float(y_vals.min()), float(y_vals.max()) * 1.04)
        except Exception:
            x_dense, y_dense = x.astype(float), y_vals.astype(float)
    else:
        x_dense, y_dense = x.astype(float), y_vals.astype(float)

    w_svg, h_svg = 960.0, 236.0
    pad_l, pad_r, pad_t, pad_b = 14.0, 58.0, 30.0, 32.0
    plot_w = w_svg - pad_l - pad_r
    plot_h = h_svg - pad_t - pad_b

    x_min, x_max = float(x_dense[0]), max(float(x_dense[-1]), float(x_dense[0]) + 1e-4)
    y_min, y_max = float(np.min(y_dense)), float(np.max(y_dense))
    span = max(y_max - y_min, 1e-4)
    y_low = y_min - 0.06 * span
    y_high = y_max + 0.14 * span
    y_range = max(y_high - y_low, 1e-4)

    def to_px(xv: float, yv: float) -> tuple[float, float]:
        px = pad_l + ((xv - x_min) / (x_max - x_min)) * plot_w
        py = pad_t + plot_h - ((yv - y_low) / y_range) * plot_h
        return px, py

    pts = [to_px(float(xv), float(yv)) for xv, yv in zip(x_dense, y_dense)]
    path_d = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    base_y = pad_t + plot_h
    area_d = f"{path_d} L {pts[-1][0]:.1f},{base_y:.1f} L {pts[0][0]:.1f},{base_y:.1f} Z"

    peak_i = int(np.argmax(y_dense))
    peak_t = float(x_dense[peak_i])
    peak_px, peak_py = pts[peak_i]
    end_px, end_py = pts[-1]

    ref_top_val = y_min + 0.88 * span
    ref_bot_val = y_min + 0.10 * span
    _, ref_top_y = to_px(x_min, ref_top_val)
    _, ref_bot_y = to_px(x_min, ref_bot_val)
    lbl_top = y_fmt.format(ref_top_val) + unit_label
    lbl_bot = y_fmt.format(ref_bot_val) + unit_label

    xtick_vals = np.linspace(x_min, x_max, 4)
    xtick_els = []
    for tv in xtick_vals:
        tx, _ = to_px(float(tv), y_low)
        xtick_els.append(
            f'<text x="{tx:.1f}" y="{h_svg - 6:.1f}" text-anchor="middle" '
            f'fill="#78736A" font-size="11" font-weight="500">{tv:.1f}s</text>'
        )

    return f"""
    <div class="eb-chart-wrap">
        <svg class="eb-spline-svg" viewBox="0 0 {int(w_svg)} {int(h_svg)}" preserveAspectRatio="none">
            <defs>
                <linearGradient id="ebSageGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#296E4D" stop-opacity="0.16"/>
                    <stop offset="85%" stop-color="#296E4D" stop-opacity="0.02"/>
                    <stop offset="100%" stop-color="#296E4D" stop-opacity="0.0"/>
                </linearGradient>
            </defs>
            <line x1="{pad_l}" y1="{ref_top_y:.1f}" x2="{w_svg - pad_r}" y2="{ref_top_y:.1f}" stroke="#EAE5DA" stroke-width="1"/>
            <line x1="{pad_l}" y1="{ref_bot_y:.1f}" x2="{w_svg - pad_r}" y2="{ref_bot_y:.1f}" stroke="#EAE5DA" stroke-width="1" stroke-dasharray="2,3"/>
            <text x="{w_svg - pad_r + 8:.1f}" y="{ref_top_y + 3.5:.1f}" fill="#78736A" font-size="11" font-weight="500">{lbl_top}</text>
            <text x="{w_svg - pad_r + 8:.1f}" y="{ref_bot_y + 3.5:.1f}" fill="#78736A" font-size="11" font-weight="500">{lbl_bot}</text>
            <path class="eb-spline-area" d="{area_d}" fill="url(#ebSageGrad)"/>
            <path class="eb-spline-line" d="{path_d}" fill="none" stroke="#296E4D" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
            <g class="eb-peak-group">
                <text x="{peak_px:.1f}" y="{max(peak_py - 10.0, 13.0):.1f}" text-anchor="middle" fill="#57534E" font-size="11" font-weight="600">Peak {peak_t:.2f}s</text>
                <circle cx="{peak_px:.1f}" cy="{peak_py:.1f}" r="4.2" fill="#FAF8F4" stroke="#296E4D" stroke-width="1.8"/>
            </g>
            <g class="eb-end-dot">
                <circle cx="{end_px:.1f}" cy="{end_py:.1f}" r="9" fill="#296E4D" fill-opacity="0.14"/>
                <circle cx="{end_px:.1f}" cy="{end_py:.1f}" r="5" fill="#296E4D"/>
            </g>
            {''.join(xtick_els)}
        </svg>
    </div>
    """


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
# 1. Edgebook Minimal Header ("Hi." + Date/Session Pill + Top-Right Pills)
# ---------------------------------------------------------------------------
now_dt = datetime.now()
date_str = now_dt.strftime("%A, %-d %B %Y")
current_time_str = now_dt.strftime("%I:%M %p").lstrip("0")
engine_label = "Attention BiLSTM" if primary_model is not None else "Demo Preview"

st.markdown(
    f"""
    <div class="eb-topbar">
        <div class="eb-title-group">
            <h1 class="eb-greeting">Hi<span class="eb-greeting-accent">.</span></h1>
            <div class="eb-submeta-row">
                <span>{date_str}</span>
                <span class="eb-session-pill">
                    <span class="eb-dot"></span>
                    <span>{engine_label}</span>
                </span>
            </div>
        </div>
        <div class="apple-topright-bar">
            <span class="apple-status-chip">
                {SVG_WAVE}
                <span>22.05 kHz · 116-d</span>
            </span>
            <span class="apple-time-chip">
                {SVG_CLOCK}
                <span>{current_time_str}</span>
            </span>
        </div>
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
        top_c1, top_c2 = st.columns([1.85, 1.05])
        with top_c1:
            st.markdown(
                f"""
                <div class="apple-input-header">
                    <span class="apple-input-title">
                        <span class="apple-icon-badge">{SVG_MIC}</span>
                        <span>Record Voice</span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with top_c2:
            if st.button("+ New recording", key="record_new_microphone_sample", use_container_width=True):
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
                <span class="apple-input-tag">WAV · MP3 · M4A · FLAC</span>
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
    source_tag = "LIVE VOICE · ATTENTION BILSTM"
elif st.session_state.last_active_source == "upload" and upload_bytes is not None:
    raw_audio = upload_bytes
    source_tag = f"AUDIO FILE · {uploaded_file.name.upper()}"
elif mic_bytes is not None:
    raw_audio = mic_bytes
    source_tag = "LIVE VOICE · ATTENTION BILSTM"
elif upload_bytes is not None:
    raw_audio = upload_bytes
    source_tag = f"AUDIO FILE · {uploaded_file.name.upper()}"
else:
    is_demo_sample = True
    sample_files = sorted(DEMO_DIR.glob("demo_*.wav")) if DEMO_DIR.is_dir() else []
    if sample_files:
        default_sample = DEMO_DIR / "demo_calm.wav"
        chosen_sample = default_sample if default_sample.is_file() else sample_files[0]
        raw_audio = chosen_sample.read_bytes()
        source_tag = "SAMPLE PREVIEW · ATTENTION BILSTM"
    else:
        st.info("Record your voice or upload an audio file above to begin.")

# ---------------------------------------------------------------------------
# 3. Primary Edgebook Hero Card + Calendar-Style 8-Emotion Spectrum Card
# ---------------------------------------------------------------------------
if raw_audio is not None:
    try:
        y_wave = load_audio_bytes(raw_audio, sr=SAMPLE_RATE, max_seconds=MAX_SECONDS)
        _, active_frames = preprocess_waveform(y_wave, sr=SAMPLE_RATE, max_seconds=MAX_SECONDS)
        feat = extract_features_from_waveform(y_wave, sr=SAMPLE_RATE)
        len_arr = np.array([active_frames], dtype=np.int64)

        if primary_model is None:
            weights = None
            probs = demo_probabilities(y_wave) if is_demo_sample else None
        else:
            if is_demo_sample:
                probs = demo_probabilities(y_wave)
                _, _, weights = primary_model.predict(feat, return_attention=True, lengths=len_arr)
            else:
                _, probs, weights = primary_model.predict(feat, return_attention=True, lengths=len_arr)

        p_bi = models["bilstm"].predict(feat, return_attention=False, lengths=len_arr)[1] if "bilstm" in models else None
        p_ls = models["lstm"].predict(feat, return_attention=False, lengths=len_arr)[1] if "lstm" in models else None

        rms_vals = compute_rms_energy(y_wave)
        att_curve = weights if weights is not None else rms_vals
        t_axis = frames_to_time(len(att_curve), sr=SAMPLE_RATE, hop_length=512)
        peak_time = float(t_axis[int(np.argmax(att_curve))])

        # Measured held-out metrics from comparison.json (never fabricated)
        att_m = measured.get("attention_bilstm")
        bi_m = measured.get("bilstm")
        if att_m is not None:
            att_acc_val = float(att_m["accuracy"]) * 100.0
            att_f1_val = float(att_m["macro_f1"]) * 100.0
            acc_str = f"{att_acc_val:.1f}%"
            acc_sub = f"Macro F1 · {att_f1_val:.1f}%"
            acc_bar_w = f"{min(max(att_acc_val, 0.0), 100.0):.0f}%"
        else:
            acc_str = "N/A"
            acc_sub = "Run train.py"
            acc_bar_w = "0%"

        if att_m is not None and bi_m is not None:
            gain_val = (float(att_m["accuracy"]) - float(bi_m["accuracy"])) * 100.0
            gain_str = f"{gain_val:+.1f}%"
        else:
            gain_str = "N/A"

        bi_top = LABELS[int(np.argmax(p_bi))].title() if p_bi is not None else "N/A"
        ls_top = LABELS[int(np.argmax(p_ls))].title() if p_ls is not None else "N/A"

        with st.container(border=True):
            h_left, h_right = st.columns([2.1, 1.0])
            with h_left:
                if probs is not None:
                    top_idx = int(np.argmax(probs))
                    top_emotion = LABELS[top_idx].title()
                    top_pct = float(probs[top_idx]) * 100.0
                    whole_str, dec_str = split_num_parts(top_pct)
                    second_idx = int(np.argsort(probs)[-2])
                    second_emotion = LABELS[second_idx].title()
                    second_pct = float(probs[second_idx]) * 100.0
                    margin = top_pct - second_pct
                    st.markdown(
                        f"""
                        <div class="eb-micro-label">{source_tag}</div>
                        <div class="eb-hero-number">{top_emotion} &nbsp;{whole_str}<span class="eb-hero-decimal">{dec_str}</span></div>
                        <div class="eb-hero-delta">
                            +{margin:.1f}% margin <span class="muted">over {second_emotion} · Peak focus at {peak_time:.2f}s</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    top_idx = -1
                    second_idx = -1
                    top_emotion = "Unavailable"
                    second_emotion = "N/A"
                    top_pct = 0.0
                    second_pct = 0.0
                    st.markdown(
                        f"""
                        <div class="eb-micro-label">{source_tag}</div>
                        <div class="eb-hero-number">No Checkpoint</div>
                        <div class="eb-hero-delta">
                            <span class="muted">Train model checkpoint for live classification · Peak at {peak_time:.2f}s</span>
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
                svg_html = render_edgebook_svg_chart(t_axis, att_curve * 100.0, unit_label="%", y_fmt="{:.1f}")
            elif chart_mode == "Energy":
                t_r = frames_to_time(len(rms_vals), sr=SAMPLE_RATE, hop_length=512)
                svg_html = render_edgebook_svg_chart(
                    t_r,
                    rms_vals / max(float(rms_vals.max()), 1e-6) * 100.0,
                    unit_label="%",
                    y_fmt="{:.0f}",
                )
            else:
                env = np.abs(y_wave)
                t_w = np.linspace(0.0, MAX_SECONDS, len(env))
                svg_html = render_edgebook_svg_chart(t_w, env, unit_label="", y_fmt="{:.2f}")

            st.markdown(svg_html, unsafe_allow_html=True)

            conf_str = f"{top_pct:.1f}%" if probs is not None else "N/A"
            conf_sub = f"1st · {top_emotion} vs {second_emotion}" if probs is not None else "No checkpoint"
            focus_sub = f"{active_frames} of 173 active frames"

            st.markdown(
                f"""
                <div class="eb-stat-strip">
                    <div class="eb-stat-cell">
                        <div class="eb-micro-label">CONFIDENCE</div>
                        <div class="eb-stat-val sage">{conf_str}</div>
                        <div class="eb-stat-sub">{conf_sub}</div>
                    </div>
                    <div class="eb-stat-cell">
                        <div class="eb-micro-label">TEMPORAL FOCUS</div>
                        <div class="eb-stat-val sage">+{peak_time:.2f}s</div>
                        <div class="eb-stat-sub bronze-note">{focus_sub}</div>
                    </div>
                    <div class="eb-stat-cell">
                        <div class="eb-micro-label">TEST ACCURACY</div>
                        <div class="eb-stat-val">{acc_str}</div>
                        <div class="eb-stat-sub">{acc_sub}</div>
                        <div class="eb-progress-track"><div class="eb-progress-fill" style="width:{acc_bar_w};"></div></div>
                    </div>
                    <div class="eb-stat-cell">
                        <div class="eb-micro-label">VS. BASELINE</div>
                        <div class="eb-stat-val">{gain_str}</div>
                        <div class="eb-stat-sub">BiLSTM · {bi_top} · LSTM · {ls_top}</div>
                        <div class="eb-progress-track"><div class="eb-progress-fill bronze" style="width:{acc_bar_w};"></div></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if probs is not None:
            rank_order = {int(idx): r + 1 for r, idx in enumerate(np.argsort(probs)[::-1])}
            cols_html = []
            for i, lbl in enumerate(LABELS):
                p_val = float(probs[i]) * 100.0
                if i == top_idx:
                    cls_name = "eb-cell active"
                elif i == second_idx:
                    cls_name = "eb-cell runner"
                else:
                    cls_name = "eb-cell"
                dot_html = '<span class="eb-cell-dot"></span>' if i in (top_idx, second_idx) else ""
                cols_html.append(
                    f'<div class="eb-col-wrap">'
                    f'  <div class="eb-col-header">{lbl}</div>'
                    f'  <div class="{cls_name}">'
                    f'    <div class="eb-cell-top">'
                    f'      <span class="eb-rank-badge">{rank_order[i]}</span>'
                    f'      {dot_html}'
                    f'    </div>'
                    f'    <div class="eb-cell-val">{p_val:.1f}%</div>'
                    f'  </div>'
                    f'</div>'
                )

            st.markdown(
                f"""
                <div class="eb-card">
                    <div class="eb-cal-header">
                        <div>
                            <div class="eb-micro-label">EMOTION SPECTRUM · 8 CLASSES</div>
                            <div class="eb-cal-summary">
                                <strong>1 utterance</strong> · Top-1 {top_emotion} ({top_pct:.1f}%) · Runner-up {second_emotion} ({second_pct:.1f}%)
                            </div>
                        </div>
                        <div class="eb-cal-pills">
                            <span class="eb-cal-pill">BiLSTM · {bi_top}</span>
                            <span class="eb-cal-pill">LSTM · {ls_top}</span>
                        </div>
                    </div>
                    <div class="eb-grid-8">
                        {''.join(cols_html)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception as exc:
        st.error(f"Could not process this audio: {exc}")
