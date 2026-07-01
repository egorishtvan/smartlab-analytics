"""
SmartLab Analytics — Dashboard
Layer 4 (Visualization) of the SmartLab / EdgeSense AI pipeline.

pip install plotly
Run with:
    streamlit run dashboard/app.py
    
"""

import os
import glob
import json
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# Page config & global style
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartLab Analytics",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DATA_PATH = os.path.join("data", "Projects")

ACCENT = "#5B8CFF"
ACCENT_SOFT = "#8FB2FF"
BG = "#0E1117"
CARD_BG = "#161B22"
BORDER = "#2A2F3A"
TEXT_MUTED = "#9AA4B2"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {BG};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {CARD_BG};
        border-right: 1px solid {BORDER};
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 16px 18px;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
    }}

    /* Headings */
    h1, h2, h3 {{
        letter-spacing: -0.02em;
    }}

    h1 span.accent {{
        color: {ACCENT_SOFT};
    }}

    /* Card container */
    .smartlab-card {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 14px;
    }}

    .smartlab-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        background: rgba(91, 140, 255, 0.15);
        color: {ACCENT_SOFT};
        border: 1px solid rgba(91, 140, 255, 0.35);
    }}

    .smartlab-muted {{
        color: {TEXT_MUTED};
        font-size: 13px;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {ACCENT};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }}
    .stButton > button:hover {{
        background-color: {ACCENT_SOFT};
        color: black;
    }}

    footer {{visibility: hidden;}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data access helpers
# ----------------------------------------------------------------------------
def list_projects():
    if not os.path.exists(BASE_DATA_PATH):
        return []
    return sorted(
        d for d in os.listdir(BASE_DATA_PATH)
        if os.path.isdir(os.path.join(BASE_DATA_PATH, d))
    )


def load_project_metadata(project: str):
    path = os.path.join(BASE_DATA_PATH, project, "project_metadata.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def list_sensors(project: str):
    path = os.path.join(BASE_DATA_PATH, project, "Sensors")
    if not os.path.exists(path):
        return []
    return sorted(
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    )


def load_sensor_metadata(project: str, sensor: str):
    path = os.path.join(BASE_DATA_PATH, project, "Sensors", sensor, "metadata.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def list_data_files(project: str, sensor: str):
    path = os.path.join(BASE_DATA_PATH, project, "Sensors", sensor)
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    return [os.path.basename(f) for f in files]


@st.cache_data(show_spinner=False)
def load_csv(project: str, sensor: str, filename: str) -> pd.DataFrame:
    full_path = os.path.join(BASE_DATA_PATH, project, "Sensors", sensor, filename)
    df = pd.read_csv(full_path)
    return df


def resolve_xy(df: pd.DataFrame):
    """Pick the timestamp/value columns, or fall back to index / first column."""
    cols_lower = {c.lower(): c for c in df.columns}
    ts_col = cols_lower.get("timestamp")
    val_col = cols_lower.get("value")

    if ts_col and val_col:
        x = pd.to_datetime(df[ts_col], errors="coerce")
        if x.isna().all():
            x = df[ts_col]
        y = df[val_col]
        return x, y, ts_col, val_col

    x = df.index
    y = df.iloc[:, 0]
    return x, y, "index", df.columns[0]


# ----------------------------------------------------------------------------
# Sidebar — navigation / selection
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧪 SmartLab Analytics")
    st.caption("EdgeSense AI · local-first research platform")
    st.divider()

    projects = list_projects()

    if not projects:
        st.warning(
            f"No projects found under `{BASE_DATA_PATH}`.\n\n"
            "Generate mock data or check your working directory."
        )
        st.stop()

    project = st.selectbox("📁 Project", projects)

    sensors = list_sensors(project)
    if not sensors:
        st.warning("This project has no Sensors folder yet.")
        st.stop()

    sensor = st.selectbox("📡 Sensor", sensors)

    files = list_data_files(project, sensor)
    if not files:
        st.warning("No CSV log files found for this sensor.")
        st.stop()

    filename = st.selectbox("🗓️ Data file", files, index=len(files) - 1)

    st.divider()
    smoothing = st.toggle("Apply moving-average smoothing", value=False)
    window = st.slider("Window size", 2, 50, 5, disabled=not smoothing)

    st.divider()
    st.caption(f"Last refreshed · {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
proj_meta = load_project_metadata(project)
sensor_meta = load_sensor_metadata(project, sensor)

st.markdown(f"# {project.replace('_', ' ')} <span class='accent'>· {sensor.replace('_', ' ')}</span>", unsafe_allow_html=True)

goal = proj_meta.get("goal") or proj_meta.get("Goal")
if goal:
    st.markdown(f"<div class='smartlab-muted'>{goal}</div>", unsafe_allow_html=True)

badges = []
if "type" in sensor_meta or "Type" in sensor_meta:
    badges.append(sensor_meta.get("type", sensor_meta.get("Type")))
if proj_meta.get("author") or proj_meta.get("Author"):
    badges.append(f"Author: {proj_meta.get('author', proj_meta.get('Author'))}")

if badges:
    st.markdown(
        " ".join(f"<span class='smartlab-badge'>{b}</span>" for b in badges),
        unsafe_allow_html=True,
    )

st.write("")

# ----------------------------------------------------------------------------
# Load & process data
# ----------------------------------------------------------------------------
try:
    df = load_csv(project, sensor, filename)
    if df.empty:
        st.error("The selected CSV file is empty.")
        st.stop()
except Exception as e:
    st.error(f"Failed to read data: {e}")
    st.stop()

x, y, x_label, y_label = resolve_xy(df)
y_numeric = pd.to_numeric(y, errors="coerce")

y_plot = y_numeric
if smoothing:
    y_plot = y_numeric.rolling(window=window, min_periods=1, center=True).mean()

# ----------------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Data points", f"{len(df):,}")
k2.metric("Mean", f"{y_numeric.mean():.3f}" if y_numeric.notna().any() else "—")
k3.metric("Std. dev", f"{y_numeric.std():.3f}" if y_numeric.notna().any() else "—")
delta = None
if len(y_numeric.dropna()) >= 2:
    delta = f"{(y_numeric.dropna().iloc[-1] - y_numeric.dropna().iloc[0]):+.3f}"
k4.metric("Min → Max", f"{y_numeric.min():.2f} → {y_numeric.max():.2f}", delta)

st.write("")

# ----------------------------------------------------------------------------
# Main chart
# ----------------------------------------------------------------------------
st.markdown("<div class='smartlab-card'>", unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x,
        y=y_numeric,
        mode="lines",
        line=dict(color="rgba(91,140,255,0.35)", width=1),
        name="raw",
        showlegend=smoothing,
    )
)
fig.add_trace(
    go.Scatter(
        x=x,
        y=y_plot,
        mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=4),
        name="smoothed" if smoothing else y_label,
    )
)

fig.update_layout(
    template="plotly_dark",
    plot_bgcolor=CARD_BG,
    paper_bgcolor=CARD_BG,
    margin=dict(l=10, r=10, t=30, b=10),
    height=440,
    xaxis_title=x_label,
    yaxis_title=y_label,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.update_xaxes(showgrid=True, gridcolor=BORDER)
fig.update_yaxes(showgrid=True, gridcolor=BORDER)

st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Details: raw data + metadata side by side
# ----------------------------------------------------------------------------
tab_data, tab_meta, tab_ai = st.tabs(["📄 Raw data", "⚙️ Sensor metadata", "🤖 AI insights"])

with tab_data:
    st.dataframe(df, use_container_width=True, height=320)
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )

with tab_meta:
    if sensor_meta:
        st.json(sensor_meta)
    else:
        st.info("No metadata.json found for this sensor.")

with tab_ai:
    st.info(
        "This tab is a hook for Layer 3 (`core/llm_analyzer.py`). "
        "Wire it up to run `tsfresh` feature extraction + the local Ollama model "
        "on the currently selected file, then render the returned Markdown/JSON here."
    )
    if st.button("Run local AI analysis"):
        st.warning("Connect this button to `core.llm_analyzer` to generate real insights.")