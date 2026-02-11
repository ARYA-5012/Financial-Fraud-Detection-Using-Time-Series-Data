# 📖 User Guide — Fraud Detection System

This guide walks you through setting up, running, and using the Fraud Detection API and its interactive dashboard.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Obtaining Trained Models](#2-obtaining-trained-models)
3. [Starting the API Server](#3-starting-the-api-server)
4. [Using the Dashboard](#4-using-the-dashboard)
5. [Using the REST API](#5-using-the-rest-api)
6. [Understanding Results](#6-understanding-results)
7. [Training Your Own Models](#7-training-your-own-models)
8. [Configuration](#8-configuration)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Installation

### System Requirements

- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- **4 GB RAM** minimum (model loading + inference)
- **Windows / macOS / Linux** (any OS)

### Steps

```bash
# Clone the repository
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection

# Create a virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Obtaining Trained Models

The API requires pre-trained model files in the `models/` directory. These are **not included** in the repository (too large for Git).

### Option A: Train on Kaggle (Recommended)

1. Open `notebooks/04_kaggle_training.ipynb` in [Kaggle](https://www.kaggle.com)
2. Upload the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset
3. Run all cells — the notebook will train 3 models and export them
4. Download these files from the notebook output:
   - `lightgbm_detector.pkl`
   - `catboost_detector.cbm`
   - `xgboost_detector.pkl` + `xgboost_detector.xgb`
5. Place them in the `models/` directory

### Option B: Train Locally

```bash
# Ensure raw data is in data/raw/
# (train_transaction.csv and train_identity.csv)
python train_pipeline.py
```

> ⚠️ Local training requires ~16 GB RAM and the full dataset (~1.5 GB).

---

## 3. Starting the API Server

```bash
uvicorn api.main:app --reload --port 8000
```

You should see:
```
INFO:     🚀 Models ready: ['lightgbm', 'catboost']
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

> If no model files are found, the server starts in **heuristic-only mode** — all scoring uses rule-based logic instead of ML models.

---

## 4. Using the Dashboard

### Opening the Dashboard

Navigate to **http://localhost:8000** in your browser. You'll be redirected to the dashboard automatically.

### Scoring a Transaction

1. **Fill the form** on the left panel:
   - **Transaction ID** — Any unique identifier (e.g., `TXN-001`)
   - **User ID** — Cardholder identifier (e.g., `USER-123`)
   - **Amount ($)** — Transaction amount (must be > 0)
   - **Merchant ID** — Merchant identifier
   - **Product Code** — Select from W, H, C, S, or R
   - **Card Type** — visa, mastercard, amex, etc.
   - **Device Info** — Optional; leaving blank slightly increases risk
   - **Email Domain** — Optional; e.g., `gmail.com`

2. **Click "Analyze Transaction"** or press Enter

3. **Read the results** on the right panel:
   - The **gauge** shows the fraud probability (0–100%)
   - The **risk badge** shows the risk category
   - The **reasons panel** lists why the score is what it is
   - The result is added to the **prediction history** table below

### Quick Presets

Use the **preset buttons** to quickly test different scenarios:

| Preset | Amount | Description |
|--------|--------|-------------|
| 🟢 Normal | $29.99 | Typical everyday purchase |
| 🟡 Medium | $1,200 | Mid-range transaction |
| 🟠 Suspicious | $8,500 | High amount, unusual device |
| 🔴 High Risk | $15,000 | Very high amount, no device info |

### Stats Bar

The bottom bar shows live statistics:
- **Models** — Number of loaded models
- **Predictions** — Total predictions in this session
- **Avg Latency** — Average response time in milliseconds

---

## 5. Using the REST API

### Score a Single Transaction

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN-001",
    "user_id": "USER-123",
    "transaction_amount": 499.99,
    "merchant_id": "MERCHANT-456",
    "product_code": "W",
    "card_type": "visa",
    "device_info": "Chrome on Windows",
    "email_domain": "gmail.com"
  }'
```

**Response:**
```json
{
  "transaction_id": "TXN-001",
  "fraud_score": 0.2431,
  "is_fraud": false,
  "risk_level": "LOW",
  "model_used": "LIGHTGBM",
  "processing_time_ms": 5.27,
  "confidence": 0.5138,
  "reasons": ["Missing device information"]
}
```

### Score a Batch

```bash
curl -X POST http://localhost:8000/score/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"transaction_id": "T1", "user_id": "U1", "transaction_amount": 50, "merchant_id": "M1"},
      {"transaction_id": "T2", "user_id": "U2", "transaction_amount": 9999, "merchant_id": "M2"}
    ]
  }'
```

### Check Server Health

```bash
curl http://localhost:8000/health
```

### View Swagger Docs

Open **http://localhost:8000/docs** for interactive API documentation where you can test all endpoints directly.

---

## 6. Understanding Results

### Fraud Score (0.0–1.0)

The fraud score is a probability estimate between 0 and 1:
- **0.0** = Definitely legitimate
- **1.0** = Definitely fraudulent

### Risk Levels

| Score | Level | Color | Suggested Action |
|-------|-------|-------|-----------------|
| 0.00–0.19 | MINIMAL | Green | Auto-approve |
| 0.20–0.39 | LOW | Blue | Approve |
| 0.40–0.59 | MEDIUM | Yellow | Manual review |
| 0.60–0.79 | HIGH | Orange | Hold & investigate |
| 0.80–1.00 | CRITICAL | Red | Block immediately |

### Confidence Score

The confidence score (0.0–1.0) measures how certain the model is. A score near 1.0 means the prediction is highly confident (strongly flagged or strongly cleared). A score near 0.0 means borderline — the transaction could go either way.

### Risk Reasons

Common reasons include:
- **Very high amount** — Amount exceeds $5,000
- **Missing device information** — No device data provided
- **Unusual transaction hour** — Transaction outside 5:00 AM – 11:00 PM
- **ML model flagged** — The machine learning model gave a high probability

---

## 7. Training Your Own Models

### On Kaggle (Recommended)

1. Upload `notebooks/04_kaggle_training.ipynb` to Kaggle
2. Attach the IEEE-CIS Fraud Detection dataset
3. Run all cells — training takes ~15 minutes on Kaggle's free tier
4. Download model outputs and place in `models/`

### Locally

1. Download the IEEE-CIS dataset from Kaggle
2. Place CSV files in `data/raw/`
3. Run the training pipeline:
   ```bash
   python train_pipeline.py
   ```
4. Models are saved to `models/` automatically

---

## 8. Configuration

All hyperparameters and settings are in `config/config.yaml`:

```yaml
# Key settings you might want to adjust:
api:
  host: "0.0.0.0"
  port: 8000

thresholds:
  low_risk: 0.2
  medium_risk: 0.4
  high_risk: 0.7
  critical_risk: 0.9
```

---

## 9. Troubleshooting

### Server won't start
```
ModuleNotFoundError: No module named 'lightgbm'
```
→ Run `pip install -r requirements.txt`

### "No models loaded — API will use heuristic scoring"
→ Place model files (`.pkl`, `.cbm`) in the `models/` directory. See [Section 2](#2-obtaining-trained-models).

### XGBoost model not loading
```
⚠️ XGBoost .pkl found but .xgb booster missing
```
→ You need both `xgboost_detector.pkl` AND `xgboost_detector.xgb`. Download both from Kaggle output.

### Port already in use
```
ERROR: [Errno 10048] Address already in use
```
→ Use a different port: `uvicorn api.main:app --reload --port 8001`

### Slow first request
The first `/score` request after startup may take ~500 ms as the model warms up. Subsequent requests are ~5 ms.

---

*For more details, see the [README](README.md) or open an issue on GitHub.*
