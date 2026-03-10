"""
ui/app.py
──────────
Streamlit web interface for the Stutter Detection System.

Connects to the FastAPI backend at http://localhost:8000.

Run
───
  streamlit run ui/app.py

Features
────────
• Upload a video file
• Animated processing spinner
• Display: predicted label, confidence gauge, stutter %
• Pie chart of class probabilities
• Interactive Plotly timeline of per-window predictions
"""

import sys
import io
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import STUTTER_CLASSES

# ─────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Stutter Detection System",
    page_icon  = "🎙️",
    layout     = "wide",
)

API_URL = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────────
# Colour map
# ─────────────────────────────────────────────────────────────────────────

LABEL_COLORS = {
    "Fluent":        "#2ecc71",
    "Repetition":    "#e74c3c",
    "Prolongation":  "#e67e22",
    "Block":         "#8e44ad",
    "Interjection":  "#3498db",
}

# ─────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────

st.title("🎙️ Stutter Detection System")
st.markdown(
    """
    Upload a video file to analyse speech fluency.  
    The system uses a **multimodal ML model** (Wav2Vec2 + facial landmarks)
    fine-tuned on the SEP-28k stuttering dataset.
    """
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("ℹ️ Stutter Types")
    for label, color in LABEL_COLORS.items():
        st.markdown(
            f"<span style='color:{color}; font-weight:bold;'>■</span> {label}",
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("Backend API: " + API_URL)

    # Health check
    try:
        r = requests.get(API_URL + "/health", timeout=3)
        if r.status_code == 200:
            st.success("✅ API online")
        else:
            st.error("⚠️ API returned error")
    except requests.exceptions.ConnectionError:
        st.error("❌ API offline  \n`uvicorn api.main:app`")

# ─────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    help="Max recommended size: 100 MB",
)

if uploaded is not None:
    st.video(uploaded)

    if st.button("🔍 Analyse for Stuttering", type="primary", use_container_width=True):

        with st.spinner("Processing video … extracting audio, running model …"):
            try:
                files    = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                response = requests.post(
                    API_URL + "/detect-stutter",
                    files=files,
                    timeout=300,
                )
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Start it with:  `uvicorn api.main:app`")
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.text}")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.stop()

        # ── Results layout ────────────────────────────────────────────────
        st.divider()
        st.subheader("📊 Results")

        col1, col2, col3, col4 = st.columns(4)

        label      = data["predicted_label"]
        confidence = data["confidence"]
        stutter_pct = data["stutter_pct"]
        duration   = data["duration_sec"]
        color      = LABEL_COLORS.get(label, "#95a5a6")

        with col1:
            st.metric("Predicted Type", label)
            st.markdown(
                f"<div style='background:{color};padding:6px 12px;"
                f"border-radius:8px;text-align:center;color:white;"
                f"font-weight:bold;font-size:18px'>{label}</div>",
                unsafe_allow_html=True,
            )

        with col2:
            st.metric("Confidence", f"{confidence*100:.1f} %")
            # Simple progress bar as confidence gauge
            st.progress(confidence)

        with col3:
            st.metric("Stutter %", f"{stutter_pct:.1f} %")

        with col4:
            st.metric("Duration", f"{duration:.1f} s")

        st.divider()

        # ── Class probability pie chart ───────────────────────────────────
        probs = data["class_probs"]
        fig_pie = px.pie(
            names  = list(probs.keys()),
            values = list(probs.values()),
            title  = "Class Probability Distribution",
            color  = list(probs.keys()),
            color_discrete_map = LABEL_COLORS,
            hole   = 0.35,
        )
        fig_pie.update_layout(legend_title="Stutter Type")

        # ── Timeline chart ────────────────────────────────────────────────
        timeline = data.get("timeline", [])

        if timeline:
            tl_df = pd.DataFrame(timeline)
            tl_df["color"] = tl_df["label"].map(LABEL_COLORS)
            tl_df["text"]  = tl_df.apply(
                lambda r: f"{r['label']} ({r['confidence']:.2f})", axis=1
            )

            fig_tl = go.Figure()

            for _, row in tl_df.iterrows():
                fig_tl.add_trace(go.Bar(
                    x        = [row["end_sec"] - row["start_sec"]],
                    y        = [row["label"]],
                    base     = [row["start_sec"]],
                    orientation = "h",
                    marker_color = LABEL_COLORS.get(row["label"], "#95a5a6"),
                    text     = [f"{row['confidence']:.0%}"],
                    textposition = "inside",
                    hovertemplate = (
                        f"<b>{row['label']}</b><br>"
                        f"Time: {row['start_sec']:.1f}s – {row['end_sec']:.1f}s<br>"
                        f"Confidence: {row['confidence']:.2%}<extra></extra>"
                    ),
                    showlegend = False,
                ))

            # Legend traces (one per label)
            for lbl, col in LABEL_COLORS.items():
                fig_tl.add_trace(go.Bar(
                    x=[None], y=[None],
                    marker_color=col,
                    name=lbl,
                    showlegend=True,
                ))

            fig_tl.update_layout(
                title      = "Stutter Event Timeline",
                xaxis_title = "Time (seconds)",
                yaxis_title = "",
                barmode    = "stack",
                height     = 300,
                legend     = dict(title="Type", orientation="h",
                                  yanchor="bottom", y=1.02),
            )

        left, right = st.columns(2)
        with left:
            st.plotly_chart(fig_pie, use_container_width=True)
        with right:
            if timeline:
                st.plotly_chart(fig_tl, use_container_width=True)
            else:
                st.info("No timeline data available.")

        # ── Raw data expander ─────────────────────────────────────────────
        with st.expander("🔬 Raw API Response"):
            st.json(data)

else:
    st.info("⬆️ Please upload a video file to begin analysis.")
