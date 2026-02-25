<div align="center">

# �️ Financial Fraud Detection System

**Real-time transaction fraud detection using ensemble gradient-boosting models and an interactive web dashboard.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-9ACD32)](https://lightgbm.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebooks-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/aryyaaaaa/notebook0f058ba2e7)

</div>

---

## 📌 Overview

An end-to-end machine learning pipeline that detects fraudulent credit card transactions in **real-time**. The system is trained on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset (~590K transactions) and serves predictions through a FastAPI backend with an interactive web dashboard.

### Key Highlights

| Metric | Value |
|--------|-------|
| **Dataset** | 590,540 transactions (3.5% fraud) |
| **Features Engineered** | 847 (temporal, velocity, behavioral, device) |
| **Models** | LightGBM · XGBoost · CatBoost |
| **Inference Latency** | ~5–10 ms per transaction |
| **Serving** | FastAPI + Interactive Dashboard |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web Dashboard                        │
│  (Transaction Form → Gauge → Risk Badge → History)      │
└──────────────────────┬──────────────────────────────────┘
                       │  HTTP / JSON
┌──────────────────────▼──────────────────────────────────┐
│                   FastAPI Server                        │
│  /score  /score/batch  /health  /stats  /models         │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  LightGBM   │  │   XGBoost    │  │   CatBoost   │   │
│  │  (primary)   │  │  (optional)  │  │  (fallback)  │   │
│  └─────────────┘  └──────────────┘  └──────────────┘   │
│               Heuristic + Model Blend                   │
└─────────────────────────────────────────────────────────┘
                       ▲
┌──────────────────────┴──────────────────────────────────┐
│              Training Pipeline (Kaggle)                  │
│  Raw Data → Preprocessing → Feature Engineering →       │
│  Model Training → Evaluation → Save Artifacts           │
└─────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
fraud-detection/
├── api/
│   ├── main.py                 # FastAPI application (endpoints, scoring, lifespan)
│   └── static/
│       └── index.html          # Interactive web dashboard
├── src/
│   ├── data/
│   │   └── preprocessor.py     # Missing value handling, time-based splitting
│   ├── features/
│   │   ├── engineer.py         # 847-feature engineering pipeline
│   │   └── sequence_builder.py # LSTM sequence construction
│   ├── models/
│   │   ├── lightgbm_model.py   # LightGBM detector
│   │   ├── xgboost_model.py    # XGBoost detector
│   │   ├── catboost_model.py   # CatBoost detector
│   │   ├── lstm_model.py       # LSTM with attention (optional)
│   │   └── ensemble.py         # Stacking ensemble
│   ├── evaluation/
│   │   ├── metrics.py          # AUC, PR-AUC, cost analysis, threshold optimization
│   │   └── visualizations.py   # ROC/PR curves, confusion matrix, feature importance
│   └── utils/
│       └── helpers.py          # Config loader, logging, seed management
├── config/
│   └── config.yaml             # Hyperparameters and thresholds
├── models/                     # Trained model artifacts (.pkl, .cbm, .xgb)
├── notebooks/
│   ├── 03_kaggle_eda.ipynb     # Exploratory data analysis
│   └── 04_kaggle_training.ipynb # Full training pipeline on Kaggle
├── deployment/
│   ├── Dockerfile
│   └── docker-compose.yml
├── train_pipeline.py           # Local training script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Trained model files in `models/` directory

### 1. Clone & Install

```bash
git clone https://github.com/ARYA-5012/Financial-Fraud-Detection-Using-Time-Series-Data.git
cd Financial-Fraud-Detection-Using-Time-Series-Data
pip install -r requirements.txt
```

### 2. Add Model Files

Place trained model artifacts in the `models/` directory:

| File | Model | Size |
|------|-------|------|
| `lightgbm_detector.pkl` | LightGBM (primary) | ~414 KB |
| `catboost_detector.cbm` | CatBoost | ~479 KB |
| `xgboost_detector.pkl` + `.xgb` | XGBoost (optional) | ~12 KB + booster |

> Models are trained on Kaggle using `notebooks/04_kaggle_training.ipynb` and must be downloaded from the Kaggle notebook output.

### 3. Run the API

```bash
# Option A: Run FastAPI backend (for REST API)
uvicorn api.main:app --reload --port 8000

# Option B: Run Streamlit Dashboard (standalone)
streamlit run streamlit_app.py
```

### 4. Open the Dashboard

Navigate to **http://localhost:8000** in your browser.

---

## 🎯 Features

### Interactive Dashboard
- **Transaction Form** — Input amount, card type, device, timestamp, and more
- **Quick Presets** — One-click buttons: Normal $30, Medium $1.2K, Suspicious $8K, High Risk $15K
- **Animated Gauge** — Smooth arc animation showing fraud probability
- **Risk Badges** — Color-coded MINIMAL / LOW / MEDIUM / HIGH / CRITICAL
- **Risk Factors** — Human-readable reasons why a transaction was flagged
- **Prediction History** — Session-based table of all scored transactions
- **Live Stats** — Model count, prediction totals, average latency

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Redirects to dashboard |
| `GET`  | `/dashboard` | Interactive web UI |
| `GET`  | `/health` | Server health check |
| `POST` | `/score` | Score a single transaction |
| `POST` | `/score/batch` | Score multiple transactions |
| `GET`  | `/stats` | Prediction statistics |
| `GET`  | `/models` | Loaded model details |
| `GET`  | `/docs` | Swagger API documentation |
| `GET`  | `/api-info` | API metadata |

### Feature Engineering (847 Features)

| Category | Examples | Count |
|----------|----------|-------|
| **Temporal** | Hour, day-of-week, cyclical sin/cos, weekend/night flags | ~15 |
| **Amount** | Log-transform, z-score, percentile, round-number flags | ~10 |
| **User Behavior** | Rolling mean/std/min/max amounts, transaction counts | ~20 |
| **Velocity** | Spending rate, acceleration, burst detection | ~6 |
| **Merchant** | Fraud rate, first-transaction flag, amount deviation | ~7 |
| **Device** | Multi-device flag, new-device detection, email domain counts | ~5 |
| **Missing Indicators** | Binary flags for missing V-columns | ~250+ |
| **Encoded Categoricals** | Label-encoded card, product, email, device columns | ~500+ |

---

## 🔬 Training Pipeline

Training is done on **Kaggle** (free GPU/CPU) using the provided notebooks:

| Notebook | Kaggle Link | Description |
|----------|-------------|-------------|
| **EDA** | [📊 Open on Kaggle](https://www.kaggle.com/code/aryyaaaaa/notebookbb73263c07) | Data exploration, class imbalance analysis, feature distributions |
| **Training** | [🚀 Open on Kaggle](https://www.kaggle.com/code/aryyaaaaa/notebook0f058ba2e7) | Full pipeline: preprocess → features → train 3 models → evaluate → export |

### Training Results (on test set)

| Model | ROC-AUC | PR-AUC | Strengths |
|-------|---------|--------|-----------|
| **LightGBM** | ~0.94 | ~0.60 | Fastest, best generalization |
| **XGBoost** | ~0.93 | ~0.58 | Strong regularization |
| **CatBoost** | ~0.93 | ~0.57 | Best with categorical features |

---

## 🧪 Scoring Logic

The API uses a **blended scoring approach**:

```
final_score = 0.4 × model_probability + 0.6 × heuristic_score
```

- **Model probability** — Raw output from the loaded ML model (LightGBM preferred)
- **Heuristic score** — Rule-based scoring from transaction metadata (amount, time, device presence, product code)

This blend exists because the API receives ~8 features while the model was trained on 847. The heuristic provides stable baseline signals; the model contributes learned patterns.

### Risk Levels

| Score Range | Risk Level | Action |
|-------------|-----------|--------|
| 0.00–0.19 | MINIMAL | ✅ Auto-approve |
| 0.20–0.39 | LOW | ✅ Approve |
| 0.40–0.59 | MEDIUM | ⚠️ Review |
| 0.60–0.79 | HIGH | 🔶 Flag for review |
| 0.80–1.00 | CRITICAL | 🚨 Block transaction |

---

## 🐳 Docker Deployment

```bash
cd deployment
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

## ☁️ Streamlit Cloud Deployment

1. Fork this repository to your GitHub.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Connect your GitHub account and select this repository.
4. Set the main file path to `streamlit_app.py`.
5. Click **Deploy**! 🚀

---

## � License

This project is licensed under the MIT License.

---

## � Acknowledgments

- **IEEE-CIS Fraud Detection** dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)
- [LightGBM](https://lightgbm.readthedocs.io/), [XGBoost](https://xgboost.readthedocs.io/), [CatBoost](https://catboost.ai/)
- [FastAPI](https://fastapi.tiangolo.com/) for the serving framework

---

## 👨‍💻 Author

<table>
<tr>
<td align="center">
<strong>Arya Yadav</strong><br>
Bennett University<br>
<a href="mailto:aryayadav5012@gmail.com">📧 Email</a> |
<a href="https://github.com/yourusername">🐙 GitHub</a>
</td>
</tr>
</table>

---
