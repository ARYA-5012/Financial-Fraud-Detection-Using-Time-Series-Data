"""
Main Training Pipeline Script
Runs the complete fraud detection pipeline: EDA -> Features -> Training -> Evaluation
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path
import time

print("="*60)
print("FINANCIAL FRAUD DETECTION - TRAINING PIPELINE")
print("="*60)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n📁 STEP 1: Loading Data...")
start_time = time.time()

data_dir = Path("data/raw")
processed_dir = Path("data/processed")
features_dir = Path("data/features")
models_dir = Path("models")
reports_dir = Path("reports")

# Create directories
for d in [processed_dir, features_dir, models_dir, reports_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Load training data
print("  Loading train_transaction.csv...")
train_txn = pd.read_csv(data_dir / "train_transaction.csv")
print(f"    Transactions loaded: {len(train_txn):,}")

# Load identity data if exists
identity_path = data_dir / "train_identity.csv"
if identity_path.exists():
    print("  Loading train_identity.csv...")
    train_id = pd.read_csv(identity_path)
    print(f"    Identity records loaded: {len(train_id):,}")
    
    # Merge
    print("  Merging transaction and identity data...")
    df = train_txn.merge(train_id, on='TransactionID', how='left')
else:
    df = train_txn

print(f"\n  Final dataset shape: {df.shape}")
print(f"  Fraud rate: {df['isFraud'].mean():.4%}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"  Time: {time.time() - start_time:.1f}s")

# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n📊 STEP 2: Exploratory Data Analysis...")

print("\n  Target Distribution:")
fraud_counts = df['isFraud'].value_counts()
print(f"    Legitimate: {fraud_counts[0]:,} ({fraud_counts[0]/len(df):.2%})")
print(f"    Fraudulent: {fraud_counts[1]:,} ({fraud_counts[1]/len(df):.2%})")
print(f"    Imbalance ratio: 1:{int(fraud_counts[0]/fraud_counts[1])}")

print("\n  Transaction Amount Stats:")
for label, group in df.groupby('isFraud'):
    label_name = "Fraud" if label == 1 else "Legit"
    print(f"    {label_name}: mean=${group['TransactionAmt'].mean():.2f}, "
          f"median=${group['TransactionAmt'].median():.2f}, "
          f"max=${group['TransactionAmt'].max():.2f}")

print("\n  Missing Values (top 5):")
missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
for col, pct in missing.head(5).items():
    if pct > 0:
        print(f"    {col}: {pct:.1f}%")

# ============================================================
# STEP 3: DATA PREPROCESSING
# ============================================================
print("\n🔧 STEP 3: Data Preprocessing...")
start_time = time.time()

from src.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()

# Clean data
print("  Cleaning data...")
df_clean = preprocessor.clean_data(df)

# Time-based split
print("  Creating time-based train/val/test split...")
train_df, val_df, test_df = preprocessor.time_based_split(df_clean)

# Save splits
print("  Saving processed data...")
train_df.to_csv(processed_dir / "train.csv", index=False)
val_df.to_csv(processed_dir / "val.csv", index=False)
test_df.to_csv(processed_dir / "test.csv", index=False)

print(f"  Time: {time.time() - start_time:.1f}s")

# ============================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================
print("\n⚙️ STEP 4: Feature Engineering...")
start_time = time.time()

from src.features.engineer import FraudFeatureEngineer

engineer = FraudFeatureEngineer(lookback_windows=[1, 7, 30])

# Apply feature engineering
print("  Engineering features for training set...")
train_features = engineer.fit_transform(train_df, train_df=train_df)
print(f"    Train features shape: {train_features.shape}")

print("  Engineering features for validation set...")
val_features = engineer.fit_transform(val_df, train_df=train_df)
print(f"    Val features shape: {val_features.shape}")

print("  Engineering features for test set...")
test_features = engineer.fit_transform(test_df, train_df=train_df)
print(f"    Test features shape: {test_features.shape}")

# Save features
print("  Saving engineered features...")
train_features.to_csv(features_dir / "train_features.csv", index=False)
val_features.to_csv(features_dir / "val_features.csv", index=False)
test_features.to_csv(features_dir / "test_features.csv", index=False)

print(f"  Time: {time.time() - start_time:.1f}s")

# ============================================================
# STEP 5: PREPARE TRAINING DATA
# ============================================================
print("\n📋 STEP 5: Preparing Training Data...")

# Get feature columns
feature_cols = [col for col in train_features.columns if col not in 
                ['TransactionID', 'isFraud', 'TransactionDT', 'transaction_datetime',
                 'card1', 'card2', 'FraudLabel', 'datetime', 'hour', 'day_of_week', 
                 'day', 'day_of_month', 'week_of_year', 'month']]

# Filter to numeric only
numeric_cols = []
for col in feature_cols:
    if train_features[col].dtype in ['int64', 'float64', 'int32', 'float32']:
        numeric_cols.append(col)

feature_cols = numeric_cols
print(f"  Using {len(feature_cols)} numeric features")

X_train = train_features[feature_cols].fillna(-999)
y_train = train_features['isFraud']

X_val = val_features[feature_cols].fillna(-999)
y_val = val_features['isFraud']

X_test = test_features[feature_cols].fillna(-999)
y_test = test_features['isFraud']

print(f"  X_train: {X_train.shape}")
print(f"  X_val: {X_val.shape}")
print(f"  X_test: {X_test.shape}")

# ============================================================
# STEP 6: TRAIN LIGHTGBM
# ============================================================
print("\n🚀 STEP 6: Training LightGBM...")
start_time = time.time()

from src.models.lightgbm_model import LightGBMFraudDetector

lgb_detector = LightGBMFraudDetector()
lgb_results = lgb_detector.train(
    X_train, y_train,
    X_val, y_val,
    num_boost_round=500,
    early_stopping_rounds=50
)

# Save model
lgb_detector.save(models_dir / "lightgbm_detector.pkl")
print(f"  Time: {time.time() - start_time:.1f}s")

# ============================================================
# STEP 7: TRAIN XGBOOST
# ============================================================
print("\n🚀 STEP 7: Training XGBoost...")
start_time = time.time()

from src.models.xgboost_model import XGBoostFraudDetector

xgb_detector = XGBoostFraudDetector()
xgb_results = xgb_detector.train(
    X_train, y_train,
    X_val, y_val,
    num_boost_round=500,
    early_stopping_rounds=50
)

# Save model
xgb_detector.save(models_dir / "xgboost_detector.pkl")
print(f"  Time: {time.time() - start_time:.1f}s")

# ============================================================
# STEP 8: TRAIN CATBOOST
# ============================================================
print("\n🚀 STEP 8: Training CatBoost...")
start_time = time.time()

from src.models.catboost_model import CatBoostFraudDetector

cat_detector = CatBoostFraudDetector()
cat_params = cat_detector.get_default_params()
cat_params['iterations'] = 500
cat_params['verbose'] = 100
cat_detector.params = cat_params

cat_results = cat_detector.train(
    X_train, y_train,
    X_val, y_val
)

# Save model
cat_detector.save(models_dir / "catboost_detector.cbm")
print(f"  Time: {time.time() - start_time:.1f}s")

# ============================================================
# STEP 9: EVALUATE ALL MODELS
# ============================================================
print("\n📈 STEP 9: Final Evaluation on Test Set...")

from src.evaluation.metrics import evaluate_fraud_model, compare_models

# Get predictions
lgb_pred = lgb_detector.predict(X_test)
xgb_pred = xgb_detector.predict(X_test)
cat_pred = cat_detector.predict(X_test)

# Compare models
predictions = {
    'LightGBM': lgb_pred,
    'XGBoost': xgb_pred,
    'CatBoost': cat_pred
}

comparison = compare_models(y_test, predictions)

print("\n" + "="*60)
print("MODEL COMPARISON (Test Set)")
print("="*60)
print(comparison.to_string(index=False))
print("="*60)

# Save comparison
comparison.to_csv(reports_dir / "model_comparison.csv", index=False)

# Best model evaluation
print("\n📊 Best Model Detailed Evaluation:")
best_model_name = comparison.iloc[0]['Model']
best_pred = predictions[best_model_name]
evaluate_fraud_model(y_test, best_pred)

# ============================================================
# STEP 10: FEATURE IMPORTANCE
# ============================================================
print("\n🔍 STEP 10: Top 20 Important Features (LightGBM):")
importance = lgb_detector.get_feature_importance(20)
for rank, (_, row) in enumerate(importance.iterrows(), start=1):
    print(f"  {rank:2d}. {row['feature']}: {row['importance']:.0f}")

# Save feature importance
importance.to_csv(reports_dir / "feature_importance.csv", index=False)

# ============================================================
# COMPLETE
# ============================================================
print("\n" + "="*60)
print("✅ TRAINING PIPELINE COMPLETE!")
print("="*60)
print(f"""
Results saved to:
  - Models: {models_dir}/
  - Reports: {reports_dir}/
  - Features: {features_dir}/

Best Model: {comparison.iloc[0]['Model']}
  ROC-AUC: {comparison.iloc[0]['ROC-AUC']:.4f}
  PR-AUC: {comparison.iloc[0]['PR-AUC']:.4f}

To start the API:
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
""")
print("="*60)
