import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Ferry Predictive Ops", page_icon="⛴️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetric"] {
        background: rgba(120,140,180,0.10);
        border: 1px solid rgba(120,140,180,0.25);
        padding: 12px 16px; border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("⛴️ Toronto Island Ferry: Predictive Decision Support")
st.caption("Transforming historical ferry ticket data into forward-looking operational intelligence.")

# ==========================================
# 2. MODEL BENCHMARK RESULTS (from experiments)
# ==========================================
BENCHMARKS = pd.DataFrame(
    [
        ("Naïve Baseline (t-1)", 21.42, 86.27),
        ("Moving Average Baseline (1H)", 19.49, 69.75),
        ("Random Forest", 17.39, 63.25),
        ("XGBoost (baseline)", 16.90, 63.02),
        ("Facebook Prophet", 40.45, 81.32),
        ("SARIMA", 50.16, 112.54),
        ("Optimized XGBoost", 16.90, 62.49),
    ],
    columns=["Model", "MAE", "RMSE"],
)
BEST_PARAMS = {
    "subsample": 0.8, "n_estimators": 500, "min_child_weight": 3,
    "max_depth": 5, "learning_rate": 0.01, "colsample_bytree": 0.7,
}
# Models that can be run live in the app (the others are reported as benchmarks only)
LIVE_MODELS = {
    "Optimized XGBoost": 62.49,
    "Naïve Baseline (t-1)": 86.27,
    "Moving Average Baseline (1H)": 69.75,
}

# ==========================================
# 3. LOADING
# ==========================================
@st.cache_data
def load_data():
    path = BASE_DIR / "Source" / "data" / "processed" / "engineered_ferry.csv"
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d-%m-%Y %H.%M")
    df = df.set_index("Timestamp").sort_index()
    # 1-hour moving average of the previous 4 intervals (no leakage)
    df["MA_1H"] = df["Sales Count"].shift(1).rolling(4).mean()
    return df


@st.cache_resource
def load_saved_model():
    model = joblib.load(BASE_DIR / "Source" / "models" / "xgboost_ferry_model.pkl")
    features = joblib.load(BASE_DIR / "Source" / "models" / "model_features.pkl")
    return model, features


with st.spinner("Loading operational data and predictive engine..."):
    df = load_data()
    model, features = load_saved_model()


def predict(name: str, frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.array([])
    if name == "Optimized XGBoost":
        return np.maximum(np.asarray(model.predict(frame[features])), 0)
    if name == "Naïve Baseline (t-1)":
        return frame["Sales Count_lag_1"].to_numpy(dtype=float)
    return frame["MA_1H"].fillna(frame["Sales Count_lag_1"]).to_numpy(dtype=float)


def with_bands(frame, preds, rmse):
    out = frame[["Sales Count"]].copy()
    out["Forecast"] = preds
    out["Upper"] = preds + 1.96 * rmse
    out["Lower"] = np.maximum(preds - 1.96 * rmse, 0)
    return out


def mae_rmse(actual, pred):
    err = np.asarray(actual) - np.asarray(pred)
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


# ==========================================
# 4. SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Operational Controls")
default_date = df.index.max().date() - timedelta(days=150)
selected_date = st.sidebar.date_input(
    "Select Date", default_date, min_value=df.index.min().date(), max_value=df.index.max().date()
)
selected_time = st.sidebar.time_input("Current Time", pd.to_datetime("11:00").time(), step=900)
current_dt = pd.to_datetime(f"{selected_date} {selected_time}")

horizon_options = {
    "15 Minutes (1 Step)": 1, "30 Minutes (2 Steps)": 2,
    "1 Hour (4 Steps)": 4, "2 Hours (8 Steps)": 8,
}
horizon_label = st.sidebar.selectbox("Forecast Horizon", list(horizon_options), index=2)
steps = horizon_options[horizon_label]

st.sidebar.divider()
active_model = st.sidebar.radio("Active Predictive Engine", list(LIVE_MODELS))
compare_models = st.sidebar.multiselect(
    "Models to compare", list(LIVE_MODELS), default=list(LIVE_MODELS)
)
history_hours = st.sidebar.slider("History window (hours)", 1, 12, 4)

st.sidebar.divider()
st.sidebar.subheader("Risk Thresholds (tickets / 15m)")
elevated_thr = st.sidebar.number_input("Elevated above", 50, 2000, 200, step=25)
critical_thr = st.sidebar.number_input("Critical above", 100, 3000, 500, step=25)
show_bands = st.sidebar.checkbox("Show 95% confidence band", True)

# ==========================================
# 5. SLICING
# ==========================================
past_cut = current_dt - timedelta(hours=history_hours)
future_start = current_dt + timedelta(minutes=15)
future_end = current_dt + timedelta(minutes=15 * steps)

history = df.loc[past_cut:current_dt]
future = df.loc[future_start:future_end]

if history.empty or future.empty:
    st.error("Selected date/time is outside the dataset range (or has no data). Please pick another time.")
    st.stop()

rmse_active = LIVE_MODELS[active_model]
fc = with_bands(future, predict(active_model, future), rmse_active)

current_crowd = int(history["Sales Count"].iloc[-1])
pred_total = int(fc["Forecast"].sum())
peak_risk = int(fc["Upper"].max())
actual_total = int(future["Sales Count"].sum())
w_mae, w_rmse = mae_rmse(future["Sales Count"], fc["Forecast"])

if peak_risk > critical_thr:
    status = "🔴 CRITICAL"
    action = "Deploy secondary ferry"
elif peak_risk > elevated_thr:
    status = "🟡 ELEVATED"
    action = "Monitor gates"
else:
    status = "🟢 NORMAL"
    action = "Standard operations"

# ==========================================
# 6. KPI ROW
# ==========================================
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Current Crowd (last 15m)", f"{current_crowd}")
k2.metric(f"Predicted Demand ({horizon_label.split(' (')[0]})", f"{pred_total}",
          delta=f"{pred_total - actual_total:+d} vs actual", delta_color="off")
k3.metric("Peak Risk (upper band)", f"{peak_risk}",
          help="Worst-case per-interval prediction from the 95% interval.")
k4.metric("Window MAE", f"{w_mae:.1f}", help="Mean absolute error of the active model on this horizon.")
k5.metric("Staff Readiness", status, help=action)

if status.startswith("🔴"):
    st.error(f"**{status}** — {action}. Peak upper bound: {peak_risk} tickets / 15m.")
elif status.startswith("🟡"):
    st.warning(f"**{status}** — {action}. Peak upper bound: {peak_risk} tickets / 15m.")
else:
    st.success(f"**{status}** — {action}.")

# ==========================================
# 7. TABS
# ==========================================
tab_fc, tab_cmp, tab_pva, tab_models = st.tabs(
    ["📈 Demand Forecast", "🆚 Model Comparison", "🎯 Predicted vs Actual", "🧠 Model Details"]
)

# ---- Tab 1: forecast ----
with tab_fc:
    st.subheader(f"Demand forecast — {active_model}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history["Sales Count"], mode="lines+markers",
                             name="Actual (history)", line=dict(color="royalblue", width=2)))
    if show_bands:
        fig.add_trace(go.Scatter(
            x=np.concatenate([fc.index, fc.index[::-1]]),
            y=np.concatenate([fc["Upper"], fc["Lower"][::-1]]),
            fill="toself", fillcolor="rgba(255,165,0,0.2)", line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", name="95% confidence band"))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["Forecast"], mode="lines+markers",
                             name="Forecast", line=dict(color="darkorange", width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=future.index, y=future["Sales Count"], mode="lines+markers",
                             name="Actual (future, ground truth)",
                             line=dict(color="seagreen", width=2, dash="dot")))
    fig.add_hline(y=elevated_thr, line_dash="dot", line_color="gold",
                  annotation_text="Elevated", annotation_position="top left")
    fig.add_hline(y=critical_thr, line_dash="dot", line_color="crimson",
                  annotation_text="Critical", annotation_position="top left")
    fig.add_vline(x=current_dt, line_dash="solid", line_color="gray")
    fig.update_layout(xaxis_title="Time", yaxis_title="Tickets sold (per 15m)", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      margin=dict(l=0, r=0, t=30, b=0), height=460)
    st.plotly_chart(fig, width='stretch')

    with st.expander("Forecast table"):
        tbl = fc.rename(columns={"Sales Count": "Actual"}).round(1)
        st.dataframe(tbl, width='stretch')
        st.download_button("Download forecast CSV", tbl.to_csv().encode(), "ferry_forecast.csv", "text/csv")

# ---- Tab 2: model comparison ----
with tab_cmp:
    st.subheader("Live comparison on the selected horizon")
    if not compare_models:
        st.info("Select at least one model in the sidebar.")
    else:
        colors = px.colors.qualitative.Set2
        figc = go.Figure()
        figc.add_trace(go.Scatter(x=future.index, y=future["Sales Count"], mode="lines+markers",
                                  name="Actual", line=dict(color="black", width=3)))
        rows = []
        for i, m in enumerate(compare_models):
            p = predict(m, future)
            figc.add_trace(go.Scatter(x=future.index, y=p, mode="lines+markers", name=m,
                                      line=dict(color=colors[i % len(colors)], dash="dash")))
            mae, rmse = mae_rmse(future["Sales Count"], p)
            rows.append({"Model": m, "Predicted total": int(p.sum()),
                         "Actual total": actual_total, "MAE": round(mae, 2), "RMSE": round(rmse, 2)})
        figc.update_layout(hovermode="x unified", yaxis_title="Tickets (per 15m)",
                           margin=dict(l=0, r=0, t=20, b=0), height=380,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(figc, width='stretch')
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        st.caption("Metrics on a short horizon are noisy; use the benchmark below for overall performance.")

    st.subheader("Full test-set benchmark")
    bench = BENCHMARKS.sort_values("MAE")
    c1, c2 = st.columns(2)
    for col, metric in zip((c1, c2), ("MAE", "RMSE")):
        b = bench.sort_values(metric, ascending=False)
        f = px.bar(b, x=metric, y="Model", orientation="h", text=metric, title=f"{metric} (lower is better)",
                   color=(b["Model"] == "Optimized XGBoost").map({True: "Best", False: "Other"}),
                   color_discrete_map={"Best": "darkorange", "Other": "#8aa0c8"})
        f.update_layout(showlegend=False, margin=dict(l=0, r=0, t=40, b=0),     height=340, yaxis_title=None)
        col.plotly_chart(f, width='stretch')

# ---- Tab 3: predicted vs actual ----
with tab_pva:
    st.subheader("Predicted vs actual — full day backtest")
    day = df.loc[str(selected_date):str(selected_date)]
    if day.empty:
        st.info("No data for the selected date.")
    else:
        dp = predict(active_model, day)
        dfp = with_bands(day, dp, rmse_active)
        d_mae, d_rmse = mae_rmse(dfp["Sales Count"], dfp["Forecast"])
        coverage = float(((dfp["Sales Count"] >= dfp["Lower"]) & (dfp["Sales Count"] <= dfp["Upper"])).mean() * 100)
        m1, m2, m3 = st.columns(3)
        m1.metric("Day MAE", f"{d_mae:.2f}")
        m2.metric("Day RMSE", f"{d_rmse:.2f}")
        m3.metric("95% band coverage", f"{coverage:.0f}%")

        figd = go.Figure()
        if show_bands:
            figd.add_trace(go.Scatter(
                x=np.concatenate([dfp.index, dfp.index[::-1]]),
                y=np.concatenate([dfp["Upper"], dfp["Lower"][::-1]]),
                fill="toself", fillcolor="rgba(255,165,0,0.15)", line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", name="95% band"))
        figd.add_trace(go.Scatter(x=dfp.index, y=dfp["Sales Count"], name="Actual", line=dict(color="royalblue")))
        figd.add_trace(go.Scatter(x=dfp.index, y=dfp["Forecast"], name="Predicted",
                                  line=dict(color="darkorange", dash="dash")))
        figd.add_vline(x=current_dt, line_color="gray")
        figd.update_layout(hovermode="x unified", yaxis_title="Tickets (per 15m)",
                           margin=dict(l=0, r=0, t=20, b=0), height=380,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(figd, width='stretch')

        s1, s2 = st.columns(2)
        sc = px.scatter(dfp, x="Sales Count", y="Forecast", opacity=0.6,
                        labels={"Sales Count": "Actual", "Forecast": "Predicted"}, title="Predicted vs actual")
        mx = float(max(dfp["Sales Count"].max(), dfp["Forecast"].max()))
        sc.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(color="gray", dash="dash"))
        sc.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
        s1.plotly_chart(sc, width='stretch')

        resid = dfp["Sales Count"] - dfp["Forecast"]
        hist = px.histogram(resid, nbins=30, title="Residuals (actual − predicted)")
        hist.update_layout(showlegend=False, height=320, margin=dict(l=0, r=0, t=40, b=0), xaxis_title="Error")
        s2.plotly_chart(hist, width='stretch')

# ---- Tab 4: model details ----
with tab_models:
    st.subheader("Model leaderboard")
    st.dataframe(
        BENCHMARKS.sort_values("MAE").reset_index(drop=True)
        .style.highlight_min(subset=["MAE", "RMSE"], color="#2e7d3255"),
        width='stretch', hide_index=True,
    )
    naive_mae = BENCHMARKS.loc[BENCHMARKS.Model == "Naïve Baseline (t-1)", "MAE"].iloc[0]
    xgb_mae = BENCHMARKS.loc[BENCHMARKS.Model == "Optimized XGBoost", "MAE"].iloc[0]
    st.info(f"Optimized XGBoost cuts MAE by **{(naive_mae - xgb_mae) / naive_mae * 100:.1f}%** "
            f"versus the naïve t-1 baseline ({naive_mae} → {xgb_mae}).")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Hyperparameter tuning (RandomizedSearchCV)**")
        st.caption("20 parameter combinations × 3 time-series folds = 60 fits")
        st.json(BEST_PARAMS)
        st.markdown(
            "- Baseline XGBoost: MAE 16.90 | RMSE 63.02\n"
            "- Optimized XGBoost: MAE 16.90 | RMSE 62.49\n"
            "- MAE improvement over baseline: ~0.01"
        )
    with d2:
        st.markdown("**Model notes**")
        st.markdown(
            "- **Naïve (t-1):** repeats the previous 15-minute sales value.\n"
            "- **Moving average (1H):** mean of the previous 4 intervals.\n"
            "- **Random Forest / XGBoost:** gradient-boosted and bagged trees on engineered lag, calendar and weather-style features.\n"
            "- **Prophet & SARIMA:** classical time-series approaches; weaker here (MAE 40.45 / 50.16), so excluded from live inference.\n"
            "- Random Forest, Prophet and SARIMA appear as benchmarks only; they aren't loaded live."
        )

    if hasattr(model, "feature_importances_"):
        imp = (pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
               .sort_values("Importance").tail(15))
        fi = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Top 15 XGBoost features")
        fi.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fi, width='stretch')

    st.markdown("**Forecast uncertainty**")
    st.markdown(
        "Bands are a 95% interval: forecast ± 1.96 × model RMSE (symmetric, floored at 0). "
        "When the upper bound crosses your risk thresholds, stage secondary vessels and add gate staff proactively. "
        "Thresholds are adjustable in the sidebar."
    )
    st.caption(
        "Note: lag features come from recorded data, so forecasts for later steps in the horizon use actual "
        "lags rather than recursive predictions. Treat multi-step accuracy here as optimistic."
    )