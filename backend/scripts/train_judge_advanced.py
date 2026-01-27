import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import joblib
import json

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = "backend/data/judge_training_data.csv"
MODEL_PATH = "backend/models/judge.json"
METRICS_PATH = "backend/models/judge_metrics.json"

def train_judge():
    print("--- Training The Judge (XGBoost Meta-Model) ---")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data not found at {DATA_PATH}. Run generate_judge_data.py first.")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Target: 
    # 'true_label' is 0 (Neutral), 1 (Buy), 2 (Sell).
    # 'pred_label' is what Chronos predicted.
    # We want to predict: "Is Chronos Correct?" OR "Is this a Profitable Signal?"
    
    # Approach A: Binary Classification "Is Correct?"
    # If Chronos says Buy (1) and True is Buy (1) -> Correct (1)
    # If Chronos says Buy (1) and True is Not Buy (0 or 2) -> Incorrect (0)
    
    # Approach B: Probability of Profit (Directional)
    # If signal is Buy, prob that True is Buy.
    # If signal is Sell, prob that True is Sell.
    
    # Let's filter for ACTIVE signals (Buy/Sell) from Chronos.
    # We don't need a Judge for Neutral.
    
    active_mask = df['pred_label'].isin([1, 2])
    print(f"Filtering for active signals... {active_mask.sum()} / {len(df)} rows.")
    df_active = df[active_mask].copy()
    
    if len(df_active) < 100:
        print("Not enough active signals to train Judge.")
        return

    # Define Target: Success
    # If Pred=1 (Buy), Success if True=1.
    # If Pred=2 (Sell), Success if True=2 (Sell).
    
    conditions = [
        (df_active['pred_label'] == 1) & (df_active['true_label'] == 1),
        (df_active['pred_label'] == 2) & (df_active['true_label'] == 2) 
    ]
    # Note: true_label might be 0 (Neutral/TimeOut). considered Failure.
    # true_label 2 mixed with pred 1 is Failure.
    
    df_active['success'] = np.select(conditions, [1, 1], default=0)
    
    print(f"Success Rate in Data: {df_active['success'].mean():.2%}")
    
    # Features
    # Context + Confidence
    features = [
        'conf_sell', 'prob_neutral', 'prob_buy', 'prob_sell',
        'bpi', 'ad_line', 'rs', 'rsi',
        'volatility_proxy', 
        # Add spread?
        # 'spread' = prob_buy - prob_sell?
    ]
    
    # Valid feats check
    features = [f for f in features if f in df_active.columns]
    print(f"Features: {features}")
    
    X = df_active[features]
    y = df_active['success']
    
    # Train/Test Split (Time Series Split preferred, but KFold ok for now if purged)
    # XGBoost
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        tree_method="hist", # CPU fast hist
        device="cpu" 
    )
    
    print("Training...")
    model.fit(X, y)
    
    # Eval
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    print("\n--- Judge Performance (In-Sample) ---")
    print(classification_report(y, preds))
    print(f"AUC: {roc_auc_score(y, probs):.4f}")
    
    # Feature Importance
    print("\nFeature Importance:")
    fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(fi)
    
    # Save
    model.save_model(MODEL_PATH)
    print(f"Judge Model saved to {MODEL_PATH}")
    
    # Save Thresholds/Metadata?
    meta = {
        "features": features,
        "auc": roc_auc_score(y, probs),
        "threshold": 0.5 # Default, can be tuned
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(meta, f, indent=4)

if __name__ == "__main__":
    train_judge()
