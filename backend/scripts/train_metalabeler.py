import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

DATA_PATH = "backend/data/judge_training_data.csv"
MODEL_PATH = "backend/models/judge_xgb.json"

def train_metalabeler():
    print("--- Training The Judge (XGBoost Metalabeler) ---")

    if not os.path.exists(DATA_PATH):
        print("Judge training data missing. Run generate_judge_data.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples.")

    # 1. Define Target
    # We want to predict if the Specialist's PRIMARY signal is correct.
    # Primary Signal = pred_label != 0 (Neutral)
    # AND Confidence is high? No, Judge decides confidence.
    # Let's target: Is (Pred_Label == True_Label) AND (Pred_Label != 0)?
    # Or just: Given the State + Specialist Opinion, will price go where Specialist says?

    # Simple Metalabel:
    # y = 1 if (pred_label == true_label) else 0
    df['is_correct'] = (df['pred_label'] == df['true_label']).astype(int)

    # Filter: Train only on Active Signals?
    # If Specialist says Neutral, we don't trade anyway.
    # But usually we want Judge to filter "Bad Buys".
    # So filter for pred_label != 0

    active_mask = df['pred_label'] != 0
    # Map 0=Neutral, 1=Buy, 2=Sell (Check consistency with TripleBarrier)
    # Assumption: 0=Neutral.

    train_df = df[active_mask].copy()
    print(f"Active Signals (Buy/Sell) to Audit: {len(train_df)}")

    if len(train_df) < 100:
        print("Not enough active signals to train Judge.")
        return

    # 2. Features
    # Inputs: [conf_sell, prob_neutral, prob_buy, prob_sell, bpi, ad_line, rs, rsi]
    # And maybe 'pred_label' (as categorical?) or implied direction.

    features = ['prob_buy', 'prob_sell', 'prob_neutral', 'bpi', 'ad_line', 'rs', 'rsi']
    target = 'is_correct'

    X = train_df[features]
    y = train_df[target]

    # 3. Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 4. Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
        eval_metric="logloss"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # 5. Evaluate
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds)
    auc = roc_auc_score(y_val, probs)

    print(f"\nJudge Results:")
    print(f"Accuracy: {acc:.2%}")
    print(f"Precision (Hit Rate Filtered): {prec:.2%}")
    print(f"AUC: {auc:.4f}")

    # Save
    model.save_model(MODEL_PATH)
    print(f"Judge saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_metalabeler()
