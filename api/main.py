"""
Fraud Detection API
FastAPI application for real-time fraud scoring
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import sys
import time
import logging
from contextlib import asynccontextmanager

# ── Add project root to path so we can import src.* ──────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────
models: Dict[str, Any] = {}
prediction_count: int = 0
total_latency: float = 0.0

MODEL_DIR = PROJECT_ROOT / "models"


# ═══════════════════════════════════════════════════════════
#  Lifespan — load/unload models
# ═══════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load models on startup, release on shutdown."""
    global models

    logger.info("Loading models from %s …", MODEL_DIR)

    # ── LightGBM ──────────────────────────────────────────
    lgb_path = MODEL_DIR / "lightgbm_detector.pkl"
    if lgb_path.exists():
        try:
            from src.models.lightgbm_model import LightGBMFraudDetector
            m = LightGBMFraudDetector.load(str(lgb_path))
            models["lightgbm"] = m
            logger.info("✅ LightGBM loaded")
        except Exception as e:
            logger.warning("⚠️  LightGBM load failed: %s", e)

    # ── XGBoost ───────────────────────────────────────────
    xgb_path = MODEL_DIR / "xgboost_detector.pkl"
    xgb_booster = MODEL_DIR / "xgboost_detector.xgb"
    if xgb_path.exists() and xgb_booster.exists():
        try:
            from src.models.xgboost_model import XGBoostFraudDetector
            m = XGBoostFraudDetector.load(str(xgb_path))
            models["xgboost"] = m
            logger.info("✅ XGBoost loaded")
        except Exception as e:
            logger.warning("⚠️  XGBoost load failed: %s", e)
    elif xgb_path.exists():
        logger.warning(
            "⚠️  XGBoost .pkl found but .xgb booster missing — skipping. "
            "Download xgboost_detector.xgb from Kaggle Output."
        )

    # ── CatBoost ──────────────────────────────────────────
    cat_path = MODEL_DIR / "catboost_detector.cbm"
    if cat_path.exists():
        try:
            from src.models.catboost_model import CatBoostFraudDetector
            m = CatBoostFraudDetector.load(str(cat_path))
            models["catboost"] = m
            logger.info("✅ CatBoost loaded")
        except Exception as e:
            logger.warning("⚠️  CatBoost load failed: %s", e)

    if models:
        logger.info("🚀 Models ready: %s", list(models.keys()))
    else:
        logger.warning("⚠️  No models loaded — API will use heuristic scoring")

    yield  # application runs here

    models.clear()
    logger.info("Models released")


# ═══════════════════════════════════════════════════════════
#  FastAPI App
# ═══════════════════════════════════════════════════════════
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time financial fraud detection using ensemble ML models (LightGBM / XGBoost / CatBoost)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════
#  Pydantic Schemas
# ═══════════════════════════════════════════════════════════
class Transaction(BaseModel):
    """Single transaction for scoring"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User/card identifier")
    transaction_amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_id: str = Field(..., description="Merchant identifier")
    product_code: str = Field(default="W", description="Product code (W, H, C, S, R)")
    card_type: str = Field(default="visa", description="Card type")
    device_info: Optional[str] = Field(None, description="Device information")
    email_domain: Optional[str] = Field(None, description="Email domain")
    transaction_timestamp: datetime = Field(default_factory=datetime.now)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "transaction_id": "TXN123456",
                "user_id": "USER001",
                "transaction_amount": 299.99,
                "merchant_id": "MERCHANT_789",
                "product_code": "W",
                "card_type": "visa",
                "device_info": "iOS Device",
                "email_domain": "gmail.com",
                "transaction_timestamp": "2026-02-12T01:00:00",
            }]
        }
    }


class FraudResponse(BaseModel):
    transaction_id: str
    fraud_score: float = Field(..., ge=0, le=1)
    is_fraud: bool
    risk_level: str
    model_used: str
    processing_time_ms: float
    confidence: float
    reasons: Optional[List[str]] = None


class BatchRequest(BaseModel):
    transactions: List[Transaction]


class BatchResponse(BaseModel):
    total_transactions: int
    fraud_count: int
    results: List[FraudResponse]
    total_processing_time_ms: float


class ModelStats(BaseModel):
    models_loaded: List[str]
    primary_model: str
    total_predictions: int
    average_latency_ms: float
    last_updated: datetime





# ═══════════════════════════════════════════════════════════
#  Helper — score one transaction
# ═══════════════════════════════════════════════════════════
def _score(transaction: Transaction) -> tuple:
    """
    Return (fraud_probability, model_name, reasons).

    Because the trained models expect the full 847-feature vector that comes
    from FraudFeatureEngineer, and the API only receives a handful of fields,
    we use a *heuristic + model-ensemble* approach:
      • If models are loaded we build a minimal dummy feature row and ask the
        model for *directional* guidance, then blend with heuristics.
      • If no models are loaded, we use pure heuristics.
    """
    reasons: list[str] = []
    amt = transaction.transaction_amount

    # ---- heuristic base score ----
    # Sigmoid-like mapping of amount to base score
    base = 1 / (1 + np.exp(-0.003 * (amt - 500)))

    if amt > 5000:
        base = min(base + 0.15, 0.95)
        reasons.append(f"Very high amount (${amt:,.2f})")
    elif amt > 1000:
        base = min(base + 0.08, 0.85)
        reasons.append(f"High amount (${amt:,.2f})")

    if transaction.device_info is None:
        base = min(base + 0.05, 0.95)
        reasons.append("Missing device information")

    if transaction.product_code not in ("W", "H", "C", "S", "R"):
        base = min(base + 0.05, 0.95)
        reasons.append(f"Unusual product code: {transaction.product_code}")

    hour = transaction.transaction_timestamp.hour
    if hour < 5 or hour > 23:
        base = min(base + 0.07, 0.95)
        reasons.append(f"Unusual transaction hour ({hour}:00)")

    model_name = "Heuristic"

    # ---- model score (if available) ----
    # Prefer lightgbm → catboost → xgboost
    chosen_key = None
    for k in ("lightgbm", "catboost", "xgboost"):
        if k in models:
            chosen_key = k
            break

    if chosen_key is not None:
        try:
            detector = models[chosen_key]
            # Build a minimal feature row that the booster can consume.
            # We know from training the model expects 847 numeric columns.
            # We put TransactionAmt in the right spot and zeros elsewhere.
            # The model will give a *rough* probability that we blend with
            # the heuristic for a more useful API response.
            if hasattr(detector, "feature_importance") and detector.feature_importance is not None:
                feat_names = list(detector.feature_importance["feature"])
            elif hasattr(detector, "model"):
                # LightGBM booster supplies feature_name()
                if hasattr(detector.model, "feature_name"):
                    feat_names = detector.model.feature_name()
                else:
                    feat_names = None
            else:
                feat_names = None

            if feat_names is not None:
                row = pd.DataFrame(np.zeros((1, len(feat_names))), columns=feat_names)
                if "TransactionAmt" in row.columns:
                    row["TransactionAmt"] = amt
                if "log_amount" in row.columns:
                    row["log_amount"] = np.log1p(amt)
                if "amount_bin" in row.columns:
                    row["amount_bin"] = pd.cut(
                        [amt],
                        bins=[-np.inf, 50, 100, 200, 500, 1000, 5000, np.inf],
                        labels=[0, 1, 2, 3, 4, 5, 6],
                    ).codes[0]
                if "transaction_hour" in row.columns:
                    row["transaction_hour"] = hour
                if "transaction_dayofweek" in row.columns:
                    row["transaction_dayofweek"] = transaction.transaction_timestamp.weekday()

                model_prob = float(detector.predict(row)[0])
                # Blend: 40 % model, 60 % heuristic
                # (model only has partial features so we don't fully trust it)
                fraud_prob = 0.4 * model_prob + 0.6 * base
                model_name = chosen_key.upper()
                if model_prob > 0.5:
                    reasons.append(f"ML model flagged ({model_prob:.2%})")
            else:
                fraud_prob = base
        except Exception as exc:
            logger.warning("Model scoring failed (%s): %s", chosen_key, exc)
            fraud_prob = base
    else:
        fraud_prob = base

    fraud_prob = float(np.clip(fraud_prob, 0.0, 1.0))
    return fraud_prob, model_name, reasons


# ═══════════════════════════════════════════════════════════
#  Endpoints
# ═══════════════════════════════════════════════════════════
@app.get("/", tags=["Root"])
async def root():
    """Redirect to the interactive dashboard."""
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard():
    """Serve the interactive fraud detection dashboard."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api-info", tags=["Root"])
async def api_info():
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "dashboard": "/dashboard",
        "models_loaded": list(models.keys()),
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/score", response_model=FraudResponse, tags=["Scoring"])
async def score_transaction(transaction: Transaction):
    """Score a single transaction for fraud."""
    global prediction_count, total_latency
    t0 = time.time()

    fraud_prob, model_name, reasons = _score(transaction)

    # Risk bucketing
    if fraud_prob >= 0.8:
        risk, is_fraud = "CRITICAL", True
    elif fraud_prob >= 0.6:
        risk, is_fraud = "HIGH", True
    elif fraud_prob >= 0.4:
        risk, is_fraud = "MEDIUM", False
    elif fraud_prob >= 0.2:
        risk, is_fraud = "LOW", False
    else:
        risk, is_fraud = "MINIMAL", False

    ms = (time.time() - t0) * 1000
    prediction_count += 1
    total_latency += ms

    return FraudResponse(
        transaction_id=transaction.transaction_id,
        fraud_score=round(fraud_prob, 4),
        is_fraud=is_fraud,
        risk_level=risk,
        model_used=model_name,
        processing_time_ms=round(ms, 2),
        confidence=round(abs(fraud_prob - 0.5) * 2, 4),
        reasons=reasons or None,
    )


@app.post("/score/batch", response_model=BatchResponse, tags=["Scoring"])
async def score_batch(request: BatchRequest):
    """Score multiple transactions in one call."""
    t0 = time.time()
    results = []
    fraud_count = 0
    for txn in request.transactions:
        r = await score_transaction(txn)
        results.append(r)
        if r.is_fraud:
            fraud_count += 1
    return BatchResponse(
        total_transactions=len(results),
        fraud_count=fraud_count,
        results=results,
        total_processing_time_ms=round((time.time() - t0) * 1000, 2),
    )


@app.get("/stats", response_model=ModelStats, tags=["Admin"])
async def get_model_stats():
    avg = (total_latency / prediction_count) if prediction_count else 0.0
    return ModelStats(
        models_loaded=list(models.keys()) or ["none"],
        primary_model=next(
            (k.upper() for k in ("lightgbm", "catboost", "xgboost") if k in models),
            "Heuristic",
        ),
        total_predictions=prediction_count,
        average_latency_ms=round(avg, 2),
        last_updated=datetime.now(),
    )


@app.get("/models", tags=["Admin"])
async def get_models_info():
    info = {}
    for name, m in models.items():
        meta = {"loaded": True, "type": type(m).__name__}
        if hasattr(m, "feature_importance") and m.feature_importance is not None:
            meta["n_features"] = len(m.feature_importance)
        if hasattr(m, "best_iteration"):
            meta["best_iteration"] = m.best_iteration
        info[name] = meta
    return {"models": info, "total_loaded": len(models)}


# ── Static files ──────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
