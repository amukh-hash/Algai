import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from itertools import combinations
from tqdm import tqdm
import torch
from chronos import ChronosPipeline

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

DATA_CACHE_PATH = os.path.join("backend", "data_cache_alpaca", "universal_dataset_cache.pkl")
MODEL_PATH = "backend/models/chronos_physics_lora"
JUDGE_PATH = "backend/models/judge_xgboost_advanced.json"

def combinatorial_purged_cv(n_splits=6, n_test_splits=2, samples_info=None, embargo_pct=0.01):
    """
    Generates Train/Test indices for CPCV.
    Divides timeline into N chunks.
    Test set is formed by k chunks.
    Train set is remaining N-k chunks.
    Purges samples overlapping the test set borders.
    """
    # Assuming samples are sorted by time? UniversalDataset samples are per-ticker.
    # We need GLOBAL time index or per-ticker time index.
    # For simplicity, we just split by sample index if randomized? 
    # NO, we must respect Time.
    # Universal Dataset is [Ticker1 ... TickerN].
    # We should split EACH ticker's timeline.
    
    # Simplified Implementation: Standard K-Fold but grouped by Ticker?
    # No, we want time splits.
    # Let's assume we can split each ticker's data into N segments.
    
    # We return a generator of (train_idx, test_idx).
    pass # Implementation requires complex indexing. 
    # For MVP, we will use Standard TimeSeriesSplit (Walk-Forward) but Purged.
    # Train: [0...t], Purge: [t...t+p], Test: [t+p...end]
    # But CPCV is better.
    
    return []

def generate_physics_signals(limit=50000):
    """
    Loads trained Chronos model and runs inference on the dataset to generate features.
    Returns: DataFrame with [TrueLabel, ChronosDir, ChronosConf, VIX, ...]
    """
    print(f"Generating Physics Signals (Inference) - Limit: {limit}...")
    
    if not os.path.exists(MODEL_PATH):
        print("Physics Model not found. Train Chronos first.")
        return None

    # Load Model
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-large", # Base
        device_map="auto",
        torch_dtype=torch.float16
    )
    # Load LoRA
    # Unwrap ChronosModel wrapper to access the T5 model
    target_model = pipeline.model
    if hasattr(target_model, "model"):
        target_model = target_model.model
        
    print(f"Loading Adapter from {MODEL_PATH}...")
    target_model.load_adapter(MODEL_PATH)

    # Load Data
    cached = joblib.load(DATA_CACHE_PATH)
    samples = cached['samples']
    dfs = cached['dfs']
    
    # Subsample if needed
    if len(samples) > limit:
        print(f"Subsampling {limit} from {len(samples)} total samples...")
        import random
        # Random sample to get diverse regimes
        indices = sorted(random.sample(range(len(samples)), limit))
        samples = [samples[i] for i in indices]
    
    results = []
    
    # Inference Loop (Batched?)
    # Pipeline 'predict' handles batches.
    
    batch_size = 32 # Increased batch size
    context_len = 64
    pred_len = 16
    
    for i in tqdm(range(0, len(samples), batch_size)):
        batch_samples = samples[i:i+batch_size]
        contexts = []
        true_labels = []
        
        for f_idx, end_idx, label, vol in batch_samples:
            df = dfs[f_idx]
            start = end_idx - context_len + 1
            if start < 0: continue
            
            # Context
            ctx = torch.tensor(df[start:end_idx+1, 3]) # Close
            contexts.append(ctx)
            true_labels.append(label)
            
        if not contexts: continue
        
        # Forecast
        # pipeline.predict returns (B, NumSamples, Horizon)
        forecasts = pipeline.predict(contexts, prediction_length=pred_len, num_samples=20)
        
        # Analyze Forecast
        # Median path
        median_path = forecasts.quantile(0.5, dim=1) # (B, H)
        low_path = forecasts.quantile(0.1, dim=1)
        high_path = forecasts.quantile(0.9, dim=1)
        
        # Extract Features
        for b in range(len(batch_samples)):
            entry_price = contexts[b][-1].item()
            future_price = median_path[b, -1].item() # End of horizon
            
            # Direction
            pred_return = (future_price / entry_price) - 1
            pred_dir = 1 if pred_return > 0 else -1
            
            # Confidence (Spread)
            spread = (high_path[b, -1] - low_path[b, -1]).item()
            conf = 1.0 / (spread + 1e-6) # Narrow spread = High conf
            
            results.append({
                "true_label": true_labels[b],
                "pred_dir": pred_dir,
                "pred_conf": conf,
                "pred_return": pred_return
            })
            
    return pd.DataFrame(results)

def train_judge():
    print("--- Training Advanced Judge (XGBoost) ---")
    
    # 1. Get Features
    # Check cache first
    signal_cache = "backend/data/physics_signals_cache.csv"
    if os.path.exists(signal_cache):
        print("Loading cached signals...")
        df = pd.read_csv(signal_cache)
    else:
        df = generate_physics_signals()
        if df is not None:
            df.to_csv(signal_cache, index=False)
        else:
            return

    # 2. Prepare Data
    # Target: Did Physics prediction match True Direction?
    # Be careful: TrueLabel might be 0, 1, 2. (Neutral, Buy, Sell)
    # Map 0->Neutral, 1->Buy, 2->Sell.
    # PredDir is 1 or -1.
    
    # Logic:
    # If Pred=Buy (1) and Label=Buy (1) -> Win
    # If Pred=Sell (-1) and Label=Sell (2) -> Win
    # Else -> Loss
    
    df['target'] = 0
    df.loc[(df['pred_dir'] == 1) & (df['true_label'] == 1), 'target'] = 1
    df.loc[(df['pred_dir'] == -1) & (df['true_label'] == 2), 'target'] = 1
    
    X = df[['pred_dir', 'pred_conf', 'pred_return']] # Add Regime features later
    y = df['target']
    
    # 3. Train XGBoost (Simple Split for MVP)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    
    print(f"Judge Accuracy: {acc:.2%}")
    print(f"Judge Precision: {prec:.2%}")
    
    model.save_model(JUDGE_PATH)
    print("Judge Saved.")

if __name__ == "__main__":
    train_judge()
