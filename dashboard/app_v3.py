"""
SmartLab Analytics — Dashboard
Layer 4 (Visualization) of the SmartLab / EdgeSense AI pipeline.

Run with:
    streamlit run dashboard/app.py
"""

import os
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

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

PALETTE = px.colors.qualitative.Set2 + px.colors.qualitative.Set3

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {BG};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {CARD_BG};
        border-right: 1px solid {BORDER};
    }}

    div[data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 16px 18px;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important;
    }}

    h1, h2, h3 {{
        letter-spacing: -0.02em;
    }}

    h1 span.accent {{
        color: {ACCENT_SOFT};
    }}

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

    .smartlab-chip {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(255,255,255,0.03);
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 6px 10px;
        margin-bottom: 6px;
        font-size: 13px;
    }}

    .smartlab-muted {{
        color: {TEXT_MUTED};
        font-size: 13px;
    }}

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
    return pd.read_csv(full_path)


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


@st.cache_data(show_spinner=False)
def load_series(project: str, sensor: str, filenames: tuple) -> pd.DataFrame:
    """Load & concatenate one or more CSV files for a single sensor into one
    continuous, time-sorted series with columns [x, y]."""
    frames = []
    for fname in filenames:
        raw = load_csv(project, sensor, fname)
        if raw.empty:
            continue
        x, y, x_label, y_label = resolve_xy(raw)
        frames.append(pd.DataFrame({"x": x, "y": pd.to_numeric(y, errors="coerce")}))

    if not frames:
        return pd.DataFrame(columns=["x", "y"])

    combined = pd.concat(frames, ignore_index=True)

    # If x looks like real timestamps, sort chronologically; otherwise keep
    # file order and re-index sequentially so multi-day files line up.
    if pd.api.types.is_datetime64_any_dtype(combined["x"]):
        combined = combined.sort_values("x").reset_index(drop=True)
    else:
        combined = combined.reset_index(drop=True)
        combined["x"] = combined.index

    return combined


def short_label(project: str, sensor: str) -> str:
    return f"{project.replace('_', ' ')} · {sensor.replace('_', ' ')}"


# ----------------------------------------------------------------------------
# Data processing helpers (outliers, resampling, trend fitting, differencing)
# ----------------------------------------------------------------------------
def remove_outliers_iqr(y: pd.Series) -> pd.Series:
    """Mask values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] as NaN."""
    y = pd.to_numeric(y, errors="coerce")
    q1, q3 = y.quantile(0.25), y.quantile(0.75)
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return y
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return y.where((y >= lower) & (y <= upper))


def resample_series(x, y: pd.Series, rule: str, agg: str):
    """Resample a datetime x-axis into fixed buckets. Returns (x, y, applied)."""
    x = pd.Series(x)
    if not pd.api.types.is_datetime64_any_dtype(x):
        return x, y, False
    tmp = pd.DataFrame({"x": x.values, "y": pd.to_numeric(y, errors="coerce").values}).set_index("x")
    grouped = tmp["y"].resample(rule)
    resampled = getattr(grouped, agg)()
    return pd.Series(resampled.index), resampled.reset_index(drop=True), True


def _x_to_numeric(x) -> np.ndarray:
    """Convert a datetime or numeric x-axis into a plain float array for curve fitting."""
    x = pd.Series(x)
    if pd.api.types.is_datetime64_any_dtype(x):
        return x.astype("int64").to_numpy(dtype="float64")
    return pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64")


def fit_trend(x, y: pd.Series, kind: str):
    """Fit a trend curve (Linear/Quadratic/Logarithmic/Exponential).
    Returns (y_fit_array, equation_str, r2) or None if it can't be fit."""
    x_num = _x_to_numeric(x)
    y_num = pd.to_numeric(y, errors="coerce").to_numpy(dtype="float64")
    mask = ~(np.isnan(x_num) | np.isnan(y_num))
    if mask.sum() < 3:
        return None

    xv, yv = x_num[mask], y_num[mask]
    try:
        if kind == "Linear":
            c = np.polyfit(xv, yv, 1)
            y_fit = np.polyval(c, x_num)
            eq = f"y = {c[0]:.4g}·x + {c[1]:.4g}"

        elif kind == "Quadratic":
            c = np.polyfit(xv, yv, 2)
            y_fit = np.polyval(c, x_num)
            eq = f"y = {c[0]:.4g}·x² + {c[1]:.4g}·x + {c[2]:.4g}"

        elif kind == "Logarithmic":
            shift = (abs(xv.min()) + 1) if xv.min() <= 0 else 0
            c = np.polyfit(np.log(xv + shift), yv, 1)
            y_fit = c[0] * np.log(x_num + shift) + c[1]
            eq = f"y = {c[0]:.4g}·ln(x{'+' + str(round(shift, 2)) if shift else ''}) + {c[1]:.4g}"

        elif kind == "Exponential":
            if np.any(yv <= 0):
                return None
            c = np.polyfit(xv, np.log(yv), 1)
            y_fit = np.exp(c[1]) * np.exp(c[0] * x_num)
            eq = f"y = {np.exp(c[1]):.4g}·e^({c[0]:.4g}·x)"

        else:
            return None
    except Exception:
        return None

    y_fit_valid = y_fit[mask]
    ss_res = np.sum((yv - y_fit_valid) ** 2)
    ss_tot = np.sum((yv - yv.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    return y_fit, eq, r2


# ----------------------------------------------------------------------------
# Session state for the comparison "series builder"
# ----------------------------------------------------------------------------
if "series" not in st.session_state:
    st.session_state.series = []  # list of {project, sensor, files: [...], label}


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

    compare_mode = st.toggle("🔀 Compare several sources", value=False)

    st.divider()

    if not compare_mode:
        # --- Single-source mode -------------------------------------------
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

    else:
        # --- Comparison mode: build a list of series -----------------------
        st.markdown("**➕ Add series for comparison**")

        cmp_project = st.selectbox("Project", projects, key="cmp_project")
        cmp_sensors = list_sensors(cmp_project)

        if not cmp_sensors:
            st.info("There are no sensors in this project.")
        else:
            cmp_sensor = st.selectbox("Sensor", cmp_sensors, key="cmp_sensor")
            cmp_files = list_data_files(cmp_project, cmp_sensor)

            if not cmp_files:
                st.info("There are no CSV files for this sensor.")
            else:
                cmp_selected_files = st.multiselect(
                    "Files (will be combined into a single series)",
                    cmp_files,
                    default=cmp_files,
                    key="cmp_files",
                )

                if st.button("➕ Add to comparison", use_container_width=True):
                    if not cmp_selected_files:
                        st.warning("Please select at least one file.")
                    else:
                        key = (cmp_project, cmp_sensor)
                        existing_keys = {(s["project"], s["sensor"]) for s in st.session_state.series}
                        if key in existing_keys:
                            st.info("This sensor is already added to the comparison.")
                        else:
                            st.session_state.series.append({
                                "project": cmp_project,
                                "sensor": cmp_sensor,
                                "files": cmp_selected_files,
                                "label": short_label(cmp_project, cmp_sensor),
                            })
                            st.rerun()

        st.divider()
        st.markdown(f"**Selected series: {len(st.session_state.series)}**")

        for i, s in enumerate(st.session_state.series):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f"<div class='smartlab-chip'>🎨 {s['label']}"
                    f"<span class='smartlab-muted'> · {len(s['files'])} file(s)</span></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("✕", key=f"remove_{i}"):
                    st.session_state.series.pop(i)
                    st.rerun()

        if st.session_state.series and st.button("🗑️ Clear all", use_container_width=True):
            st.session_state.series = []
            st.rerun()

    st.divider()
    smoothing = st.toggle("Smoothing (moving average)", value=False)
    window = st.slider("Window size", 2, 50, 5, disabled=not smoothing)

    normalize = False
    if compare_mode:
        normalize = st.toggle(
            "Normalize (0–1)",
            value=True,
            help="Useful if the series have different units of measurement (°C, %, hPa, etc.).",
        )

    st.divider()
    st.markdown("**📊 Additional processing**")

    remove_outliers = st.toggle(
        "Remove outliers (IQR)",
        value=False,
        help="Drops points outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR] before smoothing/plotting.",
    )

    resample_rule = st.selectbox(
        "Resample (needs real timestamps)",
        ["None", "1min", "5min", "15min", "1H", "1D"],
        help="Aggregates the series into fixed time buckets. Ignored if the x-axis isn't a timestamp.",
    )
    resample_agg = st.selectbox(
        "Aggregation",
        ["mean", "median", "max", "min"],
        disabled=(resample_rule == "None"),
    )

    diff_mode = st.toggle(
        "Show rate of change (Δ)",
        value=False,
        help="Adds the first difference of the series (point-to-point change) as a second line, "
             "on its own right-hand axis.",
    )

    trend_choice = "None"
    if not compare_mode:
        trend_choice = st.selectbox(
            "Trend line",
            ["None", "Linear", "Quadratic", "Logarithmic", "Exponential"],
            help="Fits a curve over the raw data and overlays it on the chart together with R².",
        )

    st.divider()
    st.caption(f"Last refreshed · {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()


# ============================================================================
# COMPARISON MODE
# ============================================================================
if compare_mode:
    st.markdown("# 🔀 Comparison <span class='accent'>of multiple sources</span>", unsafe_allow_html=True)
    st.markdown(
        "<div class='smartlab-muted'>Add sensors from the sidebar — each will become a separate line on the chart.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    if not st.session_state.series:
        st.info("The comparison list is empty. Add a series from the sidebar.")
        st.stop()

    loaded = []
    for s in st.session_state.series:
        df_s = load_series(s["project"], s["sensor"], tuple(s["files"]))
        if df_s.empty:
            continue
        x_s, y = df_s["x"], df_s["y"]

        if remove_outliers:
            y = remove_outliers_iqr(y)

        if resample_rule != "None":
            x_s, y, _applied = resample_series(x_s, y, resample_rule, resample_agg)

        y_plot = y.rolling(window=window, min_periods=1, center=True).mean() if smoothing else y
        if normalize and y_plot.notna().any() and y_plot.max() != y_plot.min():
            y_plot = (y_plot - y_plot.min()) / (y_plot.max() - y_plot.min())
        loaded.append({**s, "x": x_s, "y_raw": y, "y_plot": y_plot})

    if not loaded:
        st.error("Failed to load any of the selected series.")
        st.stop()

    # --- Summary table -------------------------------------------------
    summary_rows = []
    for item in loaded:
        y = item["y_raw"]
        summary_rows.append({
            "Series": item["label"],
            "Points": len(y),
            "Mean": round(y.mean(), 3) if y.notna().any() else None,
            "Std": round(y.std(), 3) if y.notna().any() else None,
            "Min": round(y.min(), 3) if y.notna().any() else None,
            "Max": round(y.max(), 3) if y.notna().any() else None,
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.write("")

    # --- Overlay chart ---------------------------------------------------
    st.markdown("<div class='smartlab-card'>", unsafe_allow_html=True)

    fig = go.Figure()
    for i, item in enumerate(loaded):
        color = PALETTE[i % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=item["x"],
                y=item["y_plot"],
                mode="lines",
                line=dict(color=color, width=2),
                name=item["label"],
            )
        )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=30, b=10),
        height=480,
        yaxis_title="Normalized value (0–1)" if normalize else "Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor=BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=BORDER)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📄 Combined data for each series"):
        for item in loaded:
            st.markdown(f"**{item['label']}**")
            st.dataframe(
                pd.DataFrame({"x": item["x"], "value": item["y_raw"]}),
                use_container_width=True,
                height=200,
            )

    st.stop()


# ============================================================================
# SINGLE-SOURCE MODE
# ============================================================================
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

if remove_outliers:
    y_numeric = remove_outliers_iqr(y_numeric)

was_resampled = False
if resample_rule != "None":
    x, y_numeric, was_resampled = resample_series(x, y_numeric, resample_rule, resample_agg)
    if not was_resampled:
        st.caption("⚠️ Resampling skipped — this file's x-axis isn't a real timestamp.")

y_plot = y_numeric
if smoothing:
    y_plot = y_numeric.rolling(window=window, min_periods=1, center=True).mean()

trend_result = fit_trend(x, y_numeric, trend_choice) if trend_choice != "None" else None

y_diff = y_numeric.diff() if diff_mode else None

# --- KPI row ---------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Data points", f"{len(y_numeric):,}")
k2.metric("Mean", f"{y_numeric.mean():.3f}" if y_numeric.notna().any() else "—")
k3.metric("Std. dev", f"{y_numeric.std():.3f}" if y_numeric.notna().any() else "—")
delta = None
if len(y_numeric.dropna()) >= 2:
    delta = f"{(y_numeric.dropna().iloc[-1] - y_numeric.dropna().iloc[0]):+.3f}"
k4.metric("Min → Max", f"{y_numeric.min():.2f} → {y_numeric.max():.2f}", delta)

st.write("")

# --- Main chart --------------------------------------------------------------
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

if trend_result:
    y_fit, trend_eq, trend_r2 = trend_result
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_fit,
            mode="lines",
            line=dict(color="#FFB454", width=2, dash="dash"),
            name=f"{trend_choice} trend (R²={trend_r2:.3f})",
        )
    )

if diff_mode and y_diff is not None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_diff,
            mode="lines",
            line=dict(color="#F06767", width=1.5),
            name="Δ (rate of change)",
            yaxis="y2",
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
    yaxis2=dict(
        title="Δ value",
        overlaying="y",
        side="right",
        showgrid=False,
    ) if diff_mode else {},
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.update_xaxes(showgrid=True, gridcolor=BORDER)
fig.update_yaxes(showgrid=True, gridcolor=BORDER)

st.plotly_chart(fig, use_container_width=True)

if trend_result:
    _, trend_eq, trend_r2 = trend_result
    st.caption(f"📐 {trend_choice} trend: `{trend_eq}` · R² = {trend_r2:.3f}")
elif trend_choice != "None":
    st.caption(f"⚠️ Couldn't fit a {trend_choice.lower()} trend to this data (too few points, or "
               f"non-positive values where the fit requires them).")

st.markdown("</div>", unsafe_allow_html=True)

# --- Details tabs --------------------------------------------------------
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
    st.markdown("#### 🧬 tsfresh feature extraction")
    st.caption("Pick how the series should be processed before features are computed.")

    fc_choice = st.radio(
        "Feature set",
        ["Minimal", "Efficient", "Comprehensive"],
        horizontal=True,
        help=(
            "Minimal — a handful of fast descriptive stats (mean, std, min/max...).\n"
            "Efficient — a broad, balanced set of tsfresh calculators, skipping the slowest ones.\n"
            "Comprehensive — tsfresh's full feature catalogue (accurate but can be slow on long series)."
        ),
    )

    col_a, col_b = st.columns(2)
    with col_a:
        windowed = st.toggle(
            "Split into windows",
            value=False,
            help="Extract features per fixed-size window instead of one row for the whole "
                 "series — useful for seeing how behaviour changes over time.",
        )
    with col_b:
        impute_missing = st.toggle(
            "Impute NaN / inf",
            value=True,
            help="Some tsfresh calculators can produce NaN/inf on edge-case data; "
                 "this cleans the result automatically so nothing breaks downstream.",
        )

    window_size_tsf = st.slider(
        "Window size (points)", 10, 500, 100, step=10, disabled=not windowed
    )

    run_tsfresh = st.button("🚀 Run tsfresh feature extraction")

    if run_tsfresh:
        try:
            from tsfresh import extract_features
            from tsfresh.utilities.dataframe_functions import impute as tsfresh_impute
            from tsfresh.feature_extraction.settings import (
                MinimalFCParameters,
                EfficientFCParameters,
                ComprehensiveFCParameters,
            )
        except ImportError:
            st.error(
                "`tsfresh` is not installed in this environment. "
                "Run `pip install tsfresh` and restart the app."
            )
        else:
            fc_param_map = {
                "Minimal": MinimalFCParameters(),
                "Efficient": EfficientFCParameters(),
                "Comprehensive": ComprehensiveFCParameters(),
            }

            series_clean = y_numeric.dropna().reset_index(drop=True)
            if series_clean.empty:
                st.warning("No numeric data available to extract features from.")
            else:
                if windowed:
                    n_windows = max(1, len(series_clean) // window_size_tsf)
                    window_ids = (series_clean.index // window_size_tsf).clip(upper=n_windows - 1)
                else:
                    window_ids = pd.Series(0, index=series_clean.index)

                long_df = pd.DataFrame({
                    "id": window_ids,
                    "time": series_clean.index,
                    "value": series_clean.values,
                })

                features_df = None
                with st.spinner(f"Extracting {fc_choice.lower()} features via tsfresh..."):
                    try:
                        features_df = extract_features(
                            long_df,
                            column_id="id",
                            column_sort="time",
                            column_value="value",
                            default_fc_parameters=fc_param_map[fc_choice],
                            disable_progressbar=True,
                        )
                        if impute_missing:
                            tsfresh_impute(features_df)
                    except Exception as e:
                        st.error(f"tsfresh extraction failed: {e}")

                if features_df is not None:
                    st.session_state["tsfresh_features"] = features_df
                    st.session_state["tsfresh_meta"] = {
                        "fc_choice": fc_choice,
                        "windowed": windowed,
                        "window_size": window_size_tsf,
                    }

    if "tsfresh_features" in st.session_state:
        feats = st.session_state["tsfresh_features"]
        meta = st.session_state.get("tsfresh_meta", {})
        st.success(
            f"Extracted {feats.shape[1]} features across {feats.shape[0]} "
            f"window(s) · set: {meta.get('fc_choice', '—')}"
            + (f" · window size: {meta.get('window_size')}" if meta.get("windowed") else "")
        )
        st.dataframe(feats, use_container_width=True, height=320)
        st.download_button(
            "Download features CSV",
            data=feats.to_csv().encode("utf-8"),
            file_name=f"{sensor}_{filename.rsplit('.', 1)[0]}_tsfresh_{meta.get('fc_choice', 'features').lower()}.csv",
            mime="text/csv",
        )

    st.divider()
    st.info(
        "This section is also a hook for Layer 3 (`core/llm_analyzer.py`) — once features "
        "are extracted above, wire this button to feed them into the local Ollama model."
    )
    if st.button("Run local AI analysis"):
        st.warning("Connect this button to `core.llm_analyzer` to generate real insights.")