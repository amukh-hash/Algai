import os
import sys
import json
import joblib
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_global import InMemoryGlobalDataset, HybridPatchTST

MANIFEST_PATH = "backend/models/ensemble_manifest.json"
STACKER_PATH = "backend/models/stacker.pkl"
DATA_DIR = "backend/data_cache_alpaca"

def run_prediction(ticker="AAPL"):
    print(f"\n--- ENSEMBLE PREDICTION TEST: {ticker} ---")

    # 1. Load Registry & Stacker
    if not os.path.exists(MANIFEST_PATH):
        print("Manifest missing!") ; return
    if not os.path.exists(STACKER_PATH):
        print("Judge (stacker.pkl) missing! Please run Option 4 first.") ; return

    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)

    stacker = joblib.load(STACKER_PATH)

    # 2. Get Recent Data
    fpath = os.path.join(DATA_DIR, f"{ticker}_1m.parquet")
    if not os.path.exists(fpath):
        print(f"Data for {ticker} not found in {DATA_DIR}") ; return

    df = pd.read_parquet(fpath)
    print(f"Loaded {len(df)} minutes of history.")

    # Use most recent valid window (e.g. end of data)
    # We need at least 365 days of padding if lookback is 365
    target_idx = len(df) - 1

    # Summary of Opinions
    opinions = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use InMemory helper to get features efficiently
    dummy_ds = InMemoryGlobalDataset(DATA_DIR, lookback=365)

    # 3. Query all models
    print(f"Asking {len(manifest)} models for their verdict...")

    results_raw = []

    for entry in manifest:
        mid = entry['id']
        path = entry['path']
        cfg = entry['config']

        # Load Model
        try:
            lb = cfg['lookback']
            fs = cfg['feature_set']

            # Slice recent window
            start_in = target_idx - lb
            calc_start = max(0, start_in - 50)
            chunk = df.iloc[calc_start : target_idx].copy()

            # Feature Gen (Using the Logic from Dataset)
            if fs == 'tech':
                chunk['rsi'] = _compute_rsi(chunk['close'])
                m, s = _compute_macd(chunk['close'])
                chunk['macd'], chunk['macd_signal'] = m, s
                chunk['sma'] = chunk['close'].rolling(20).mean()
                chunk = chunk.bfill().ffill()
                feats = chunk.iloc[-lb:][['close', 'high', 'low', 'open', 'volume', 'rsi', 'macd', 'macd_signal', 'sma']].values
            elif fs == 'sentiment':
                chunk['volatility_proxy'] = (chunk['high'] - chunk['low']) / chunk['close']
                chunk['volume_shock'] = chunk['volume'] / chunk['volume'].rolling(20).mean()
                chunk = chunk.fillna(0)
                feats = chunk.iloc[-lb:][['close', 'high', 'low', 'open', 'volume', 'volatility_proxy', 'volume_shock']].values
            else:
                feats = chunk.iloc[-lb:][['close', 'high', 'low', 'open', 'volume']].values

            # Predict
            num_feats = get_num(fs)
            model = HybridPatchTST(
                num_input_features=num_feats,
                lookback_window=lb,
                n_heads=cfg['n_heads'],
                d_model=64  # Match Training!
            )
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()

            x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_dir, _ = model(x)
                prob = pred_dir.item()
                results_raw.append(prob)

        except Exception as e:
            print(f"Model {mid} crashed: {e}")
            results_raw.append(0.5)

    # 4. Final Stacking
    X = np.array(results_raw).reshape(1, -1)
    final_prob = stacker.predict(X)[0]

    # 5. Output
    print("\n" + "="*40)
    print(f"TICKER: {ticker}")
    print(f"MODELS POLLED: {len(results_raw)}")
    print(f"AVERAGE OPINION: {np.mean(results_raw):.4f}")
    print(f"JUDGE'S DECISION: {final_prob:.4f}")

    verdict = "BULLISH" if final_prob > 0.55 else ("BEARISH" if final_prob < 0.45 else "NEUTRAL")
    confidence = abs(final_prob - 0.5) * 200 # 0-100 scale

    print(f"FINAL VERDICT: {verdict} (Confidence: {confidence:.1f}%)")
    print("="*40)

# Local helpers to avoid heavy Dataset instantiation
def _compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def _compute_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def get_num(fs):
    if fs == 'tech': return 9
    if fs == 'sentiment': return 7
    return 5

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    run_prediction(ticker)
