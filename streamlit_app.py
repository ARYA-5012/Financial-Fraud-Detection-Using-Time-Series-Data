"""
🛡️ Financial Fraud Detection — Streamlit Dashboard
Real-time transaction scoring with ensemble ML models
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import sys
import logging
from pathlib import Path
from datetime import datetime

# ── Project root on sys.path so src.* imports work ───────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════
#  Page Config
# ═════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════
#  Custom CSS — premium dark theme
# ═════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── global ──────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
.stApp { font-family: 'Inter', sans-serif; }

/* ── metric cards ────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(99,102,241,.12), rgba(139,92,246,.08));
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 14px; padding: 18px 22px;
}
div[data-testid="stMetric"] label { color: #a5b4fc !important; font-weight: 500; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-weight: 700; }

/* ── gauge container ─────────────────────────────────── */
.gauge-container {
    display: flex; flex-direction: column; align-items: center;
    padding: 30px 0;
}
.gauge-svg { filter: drop-shadow(0 0 20px rgba(99,102,241,.3)); }
.gauge-score {
    font-size: 52px; font-weight: 800; margin-top: -10px;
    background: linear-gradient(135deg, #a5b4fc, #818cf8);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── risk badges ─────────────────────────────────────── */
.risk-badge {
    display: inline-block; padding: 8px 24px; border-radius: 30px;
    font-weight: 700; font-size: 16px; letter-spacing: 1.5px;
    text-transform: uppercase; text-align: center; margin: 8px 0;
}
.risk-minimal  { background: rgba(16,185,129,.15); color: #34d399; border: 1px solid rgba(16,185,129,.3); }
.risk-low      { background: rgba(59,130,246,.15); color: #60a5fa; border: 1px solid rgba(59,130,246,.3); }
.risk-medium   { background: rgba(245,158,11,.15); color: #fbbf24; border: 1px solid rgba(245,158,11,.3); }
.risk-high     { background: rgba(239,68,68,.15);  color: #f87171; border: 1px solid rgba(239,68,68,.3); }
.risk-critical { background: rgba(239,68,68,.25);  color: #fca5a5; border: 2px solid rgba(239,68,68,.5);
                 animation: pulse-red 1.5s infinite; }
@keyframes pulse-red { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.4)} 50%{box-shadow:0 0 0 12px rgba(239,68,68,0)} }

/* ── reason chips ────────────────────────────────────── */
.reason-chip {
    display: inline-block; padding: 6px 14px; border-radius: 20px;
    font-size: 13px; margin: 4px; font-weight: 500;
    background: rgba(251,191,36,.1); color: #fbbf24;
    border: 1px solid rgba(251,191,36,.3);
}

/* ── history table ───────────────────────────────────── */
.history-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 16px; border-radius: 10px; margin: 4px 0;
    background: rgba(30,30,50,.5); border: 1px solid rgba(99,102,241,.15);
    font-size: 13px;
}

/* ── sidebar ─────────────────────────────────────────── */
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
}

/* ── hide Streamlit branding ─────────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
#  Model Loading  (cached so it runs once)
# ═════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading ML models…")
def load_models():
    """Load trained models from the models/ directory."""
    loaded = {}
    model_dir = PROJECT_ROOT / "models"

    # LightGBM
    lgb_path = model_dir / "lightgbm_detector.pkl"
    if lgb_path.exists():
        try:
            from src.models.lightgbm_model import LightGBMFraudDetector
            loaded["lightgbm"] = LightGBMFraudDetector.load(str(lgb_path))
            logger.info("✅ LightGBM loaded")
        except Exception as e:
            logger.warning("LightGBM load failed: %s", e)

    # XGBoost
    xgb_pkl = model_dir / "xgboost_detector.pkl"
    xgb_bin = model_dir / "xgboost_detector.xgb"
    if xgb_pkl.exists() and xgb_bin.exists():
        try:
            from src.models.xgboost_model import XGBoostFraudDetector
            loaded["xgboost"] = XGBoostFraudDetector.load(str(xgb_pkl))
            logger.info("✅ XGBoost loaded")
        except Exception as e:
            logger.warning("XGBoost load failed: %s", e)

    # CatBoost
    cat_path = model_dir / "catboost_detector.cbm"
    if cat_path.exists():
        try:
            from src.models.catboost_model import CatBoostFraudDetector
            loaded["catboost"] = CatBoostFraudDetector.load(str(cat_path))
            logger.info("✅ CatBoost loaded")
        except Exception as e:
            logger.warning("CatBoost load failed: %s", e)

    return loaded


models = load_models()


# ═════════════════════════════════════════════════════════
#  Scoring Engine
# ═════════════════════════════════════════════════════════
def score_transaction(
    amount: float,
    product_code: str,
    card_type: str,
    device_info: str | None,
    hour: int,
    weekday: int,
) -> tuple[float, str, list[str]]:
    """Return (fraud_probability, model_name, reasons)."""
    reasons: list[str] = []

    # Heuristic base score
    base = 1 / (1 + np.exp(-0.003 * (amount - 500)))

    if amount > 5000:
        base = min(base + 0.15, 0.95)
        reasons.append(f"Very high amount (${amount:,.2f})")
    elif amount > 1000:
        base = min(base + 0.08, 0.85)
        reasons.append(f"High amount (${amount:,.2f})")

    if not device_info:
        base = min(base + 0.05, 0.95)
        reasons.append("Missing device information")

    if product_code not in ("W", "H", "C", "S", "R"):
        base = min(base + 0.05, 0.95)
        reasons.append(f"Unusual product code: {product_code}")

    if hour < 5 or hour > 23:
        base = min(base + 0.07, 0.95)
        reasons.append(f"Unusual transaction hour ({hour}:00)")

    model_name = "Heuristic"

    # Model inference
    chosen_key = next((k for k in ("lightgbm", "catboost", "xgboost") if k in models), None)

    if chosen_key is not None:
        try:
            detector = models[chosen_key]
            feat_names = None
            if hasattr(detector, "feature_importance") and detector.feature_importance is not None:
                feat_names = list(detector.feature_importance["feature"])
            elif hasattr(detector, "model") and hasattr(detector.model, "feature_name"):
                feat_names = detector.model.feature_name()

            if feat_names is not None:
                row = pd.DataFrame(np.zeros((1, len(feat_names))), columns=feat_names)
                if "TransactionAmt" in row.columns:
                    row["TransactionAmt"] = amount
                if "log_amount" in row.columns:
                    row["log_amount"] = np.log1p(amount)
                if "amount_bin" in row.columns:
                    row["amount_bin"] = pd.cut(
                        [amount],
                        bins=[-np.inf, 50, 100, 200, 500, 1000, 5000, np.inf],
                        labels=[0, 1, 2, 3, 4, 5, 6],
                    ).codes[0]
                if "transaction_hour" in row.columns:
                    row["transaction_hour"] = hour
                if "transaction_dayofweek" in row.columns:
                    row["transaction_dayofweek"] = weekday

                model_prob = float(detector.predict(row)[0])
                fraud_prob = 0.4 * model_prob + 0.6 * base
                model_name = chosen_key.upper()
                if model_prob > 0.5:
                    reasons.append(f"ML model flagged ({model_prob:.2%})")
            else:
                fraud_prob = base
        except Exception as exc:
            logger.warning("Model scoring failed: %s", exc)
            fraud_prob = base
    else:
        fraud_prob = base

    return float(np.clip(fraud_prob, 0.0, 1.0)), model_name, reasons


def get_risk_level(score: float) -> tuple[str, str, str]:
    """Return (level, css_class, emoji)."""
    if score >= 0.8:
        return "CRITICAL", "risk-critical", "🚨"
    elif score >= 0.6:
        return "HIGH", "risk-high", "🔶"
    elif score >= 0.4:
        return "MEDIUM", "risk-medium", "⚠️"
    elif score >= 0.2:
        return "LOW", "risk-low", "✅"
    else:
        return "MINIMAL", "risk-minimal", "✅"


# ═════════════════════════════════════════════════════════
#  SVG Gauge Builder
# ═════════════════════════════════════════════════════════
def build_gauge_svg(score: float) -> str:
    """Return an SVG arc gauge coloured by risk level."""
    pct = max(0, min(score, 1))
    if pct >= 0.8:
        color = "#ef4444"
    elif pct >= 0.6:
        color = "#f97316"
    elif pct >= 0.4:
        color = "#eab308"
    elif pct >= 0.2:
        color = "#3b82f6"
    else:
        color = "#10b981"

    radius = 90
    circumference = np.pi * radius  # half-circle
    offset = circumference * (1 - pct)

    return f"""
    <div class="gauge-container">
      <svg class="gauge-svg" width="240" height="140" viewBox="0 0 240 140">
        <path d="M 20 130 A 90 90 0 0 1 220 130" fill="none"
              stroke="rgba(255,255,255,.08)" stroke-width="16" stroke-linecap="round"/>
        <path d="M 20 130 A 90 90 0 0 1 220 130" fill="none"
              stroke="{color}" stroke-width="16" stroke-linecap="round"
              stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
              style="transition: stroke-dashoffset .8s ease;"/>
      </svg>
      <div class="gauge-score">{pct:.0%}</div>
    </div>
    """


# ═════════════════════════════════════════════════════════
#  Session State  (prediction history)
# ═════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []
if "pred_count" not in st.session_state:
    st.session_state.pred_count = 0
if "total_latency" not in st.session_state:
    st.session_state.total_latency = 0.0


# ═════════════════════════════════════════════════════════
#  Sidebar — Transaction Input Form
# ═════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ Fraud Detection")
    st.caption("Enter transaction details below")
    st.divider()

    # ── Quick Presets ─────────────────────────────────────
    st.markdown("**Quick Presets**")
    preset_cols = st.columns(4)

    presets = {
        "🟢 Normal":     {"amt": 29.99,   "prod": "W", "card": "visa",  "device": "Chrome"},
        "🟡 Medium":     {"amt": 1200.00, "prod": "H", "card": "mastercard", "device": "iOS"},
        "🟠 Suspicious": {"amt": 8500.00, "prod": "C", "card": "amex",  "device": ""},
        "🔴 High Risk":  {"amt": 15000.0, "prod": "R", "card": "discover", "device": ""},
    }

    for i, (label, vals) in enumerate(presets.items()):
        with preset_cols[i % 4]:
            if st.button(label, use_container_width=True, key=f"preset_{i}"):
                st.session_state["inp_amount"] = vals["amt"]
                st.session_state["inp_product"] = vals["prod"]
                st.session_state["inp_card"] = vals["card"]
                st.session_state["inp_device"] = vals["device"]

    st.divider()

    # ── Input Form ────────────────────────────────────────
    txn_id = st.text_input("Transaction ID", value="TXN-001")
    user_id = st.text_input("User ID", value="USER-001")
    amount = st.number_input(
        "Amount ($)", min_value=0.01, value=st.session_state.get("inp_amount", 299.99),
        step=10.0, format="%.2f", key="amount_input",
    )
    product_code = st.selectbox(
        "Product Code",
        options=["W", "H", "C", "S", "R"],
        index=["W", "H", "C", "S", "R"].index(st.session_state.get("inp_product", "W")),
    )
    card_type = st.selectbox(
        "Card Type",
        options=["visa", "mastercard", "amex", "discover", "other"],
        index=["visa", "mastercard", "amex", "discover", "other"].index(
            st.session_state.get("inp_card", "visa")
        ),
    )
    device_info = st.text_input(
        "Device Info (optional)",
        value=st.session_state.get("inp_device", ""),
    )
    email_domain = st.text_input("Email Domain (optional)", value="gmail.com")

    txn_time = st.time_input("Transaction Time", value=datetime.now().time())

    st.divider()
    analyze_btn = st.button("🔍  Analyze Transaction", type="primary", use_container_width=True)


# ═════════════════════════════════════════════════════════
#  Main Area — Header
# ═════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='text-align:center; margin-bottom:0;'>"
    "🛡️ Financial Fraud Detection Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#a5b4fc; margin-top:4px;'>"
    "Real-time transaction scoring powered by ensemble ML models</p>",
    unsafe_allow_html=True,
)

# ── Stats bar ─────────────────────────────────────────────
stat1, stat2, stat3, stat4 = st.columns(4)
stat1.metric("Models Loaded", len(models))
stat2.metric("Primary Model", next(
    (k.upper() for k in ("lightgbm", "catboost", "xgboost") if k in models), "Heuristic"
))
stat3.metric("Predictions", st.session_state.pred_count)
avg_lat = (
    st.session_state.total_latency / st.session_state.pred_count
    if st.session_state.pred_count else 0
)
stat4.metric("Avg Latency", f"{avg_lat:.1f} ms")

st.divider()


# ═════════════════════════════════════════════════════════
#  Scoring & Results
# ═════════════════════════════════════════════════════════
if analyze_btn:
    t0 = time.time()
    fraud_score, model_used, reasons = score_transaction(
        amount=amount,
        product_code=product_code,
        card_type=card_type,
        device_info=device_info or None,
        hour=txn_time.hour,
        weekday=datetime.now().weekday(),
    )
    latency_ms = (time.time() - t0) * 1000
    st.session_state.pred_count += 1
    st.session_state.total_latency += latency_ms

    risk_level, risk_css, risk_emoji = get_risk_level(fraud_score)
    is_fraud = fraud_score >= 0.6
    confidence = abs(fraud_score - 0.5) * 2

    # Store in history
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "txn_id": txn_id,
        "amount": amount,
        "score": fraud_score,
        "risk": risk_level,
        "model": model_used,
        "latency": latency_ms,
    })

    # ── Display Results ───────────────────────────────────
    col_gauge, col_details = st.columns([1, 1])

    with col_gauge:
        st.markdown(build_gauge_svg(fraud_score), unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:center">'
            f'<span class="risk-badge {risk_css}">{risk_emoji} {risk_level}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_details:
        st.markdown("#### 📊 Scoring Details")
        d1, d2 = st.columns(2)
        d1.metric("Fraud Score", f"{fraud_score:.4f}")
        d2.metric("Confidence", f"{confidence:.2%}")
        d3, d4 = st.columns(2)
        d3.metric("Model Used", model_used)
        d4.metric("Latency", f"{latency_ms:.1f} ms")
        d5, d6 = st.columns(2)
        d5.metric("Flagged", "🚨 YES" if is_fraud else "✅ NO")
        d6.metric("Transaction", txn_id)

    # ── Risk Reasons ──────────────────────────────────────
    if reasons:
        st.markdown("#### ⚠️ Risk Factors")
        chips_html = " ".join(f'<span class="reason-chip">⚡ {r}</span>' for r in reasons)
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.success("No risk factors detected — transaction appears normal.")

    st.divider()

else:
    # ── Empty state ───────────────────────────────────────
    st.markdown(
        "<div style='text-align:center; padding:60px 0; color:#6366f1;'>"
        "<p style='font-size:48px; margin-bottom:8px;'>🔍</p>"
        "<h3>Enter transaction details in the sidebar</h3>"
        "<p style='color:#64748b;'>Fill in the form and click <b>Analyze Transaction</b> "
        "to get a real-time fraud score</p></div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════
#  Prediction History
# ═════════════════════════════════════════════════════════
if st.session_state.history:
    st.markdown("#### 📜 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    hist_df.columns = ["Time", "Transaction", "Amount ($)", "Score", "Risk", "Model", "Latency (ms)"]
    hist_df["Amount ($)"] = hist_df["Amount ($)"].map("${:,.2f}".format)
    hist_df["Score"] = hist_df["Score"].map("{:.4f}".format)
    hist_df["Latency (ms)"] = hist_df["Latency (ms)"].map("{:.1f}".format)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════
#  Footer
# ═════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#64748b; font-size:13px; padding:10px;'>"
    "🛡️ Financial Fraud Detection System &nbsp;·&nbsp; "
    "Models: LightGBM · XGBoost · CatBoost &nbsp;·&nbsp; "
    "Trained on IEEE-CIS Fraud Detection Dataset"
    "</div>",
    unsafe_allow_html=True,
)
