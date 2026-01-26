import os
import sys
import json
import joblib
import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_global import InMemoryGlobalDataset, HybridPatchTST
from app.core.trainer import Trainer

MANIFEST_PATH = "backend/models/ensemble_manifest.json"
STACKER_PATH = "backend/models/stacker.pkl"

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            return json.load(f)
    return []

def run_judge():
    print("ALL RISE. The Judge is now in session.")
    
    # 1. Load Registry
    manifest = load_manifest()
    if not manifest:
        print("No models found in manifest. The Overlord has not done its job.")
        return

    print(f"Found {len(manifest)} candidates for the ensemble.")
    
    # 2. Prepare Validation Data (Unified)
    # We need a consistent validation set to evaluate all models.
    # We'll pick one config (e.g. Lookback 64) just to get the data format,
    # but strictly we should re-create dataset for each model if feature sets differ.
    # BETTER STRATEGY: 
    # Iterate over dataset ONCE. For each sample, generate predictions from ALL models.
    # But models have different input shapes/features.
    # So we must Iterate models, generate preds on THEIR version of the val set, ensure alignment?
    # Alignment is tricky with random sampling.
    # SIMPLIFICATION:
    # We will load a fixed set of files (last 20% of files) as "Holdout".
    # We will iterate over this Holdout.
    
    DATA_DIR = os.path.join("backend", "data_cache")
    # For stacking, we need ALIGNED predictions.
    # So we can't use random sampling dataset. We need a deterministic evaluator.
    # Let's simple use GlobalMarketDataset but with fixed seed or sequential?
    # For this POC, we will cheat slightly:
    # We will generate 1000 random samples. We will start with RAW features.
    # We will create "views" of this sample for other feature sets.
    
    print("Generating aligned validation set...")
    # Base dataset just to access file list
    base_ds = InMemoryGlobalDataset(DATA_DIR, lookback=365, feature_set='raw')
    
    val_preds = [] # (Num_Samples, Num_Models)
    val_targets = [] # (Num_Samples)
    
    # Check if models exist
    valid_models = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Feature count cache to avoid redundant dataset instantiation
    feature_count_cache = {}
    
    print(f"Loading {len(manifest)} models...")
    from tqdm import tqdm
    for entry in tqdm(manifest, desc="Loading Models"):
        path = entry['path']
        config = entry['config']
        
        if not os.path.exists(path):
            continue
            
        # Instantiate Model
        try:
            fs = config['feature_set']
            if fs not in feature_count_cache:
                # instantiate dataset just once to get num features
                dummy_ds = InMemoryGlobalDataset(DATA_DIR, feature_set=fs)
                feature_count_cache[fs] = dummy_ds._get_num_features()
            
            num_feats = feature_count_cache[fs]
            
            model = HybridPatchTST(
                num_input_features=num_feats,
                lookback_window=config['lookback'],
                d_model=64,
                n_layers=2,
                n_heads=config['n_heads']
            )
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()
            
            valid_models.append((entry['id'], model, config))
            
        except Exception as e:
            print(f"\nCandidate {entry['id']} crashed during load: {e}")
            continue

    if not valid_models:
        print("No valid models.")
        return

    print("Running Grand Tribunal (Batched Alignment & Prediction)...")
    from tqdm import tqdm
    
    TOTAL_SAMPLES = 100000
    BATCH_SIZE = 512
    X_stack = []
    y_stack = []
    
    # Pre-select files to reduce disk hit frequency
    common_ds = InMemoryGlobalDataset(DATA_DIR, lookback=365)
    all_files = common_ds.all_files
    
    # Pre-cache some files if they are small enough, or just use a rolling cache
    # The current InMemoryGlobalDataset already uses a .cache dict if we call it right.
    
    pbar = tqdm(total=TOTAL_SAMPLES, desc="Generating Meta-Data")
    
    while len(X_stack) < TOTAL_SAMPLES:
        current_batch_size = min(BATCH_SIZE, TOTAL_SAMPLES - len(X_stack))
        
        batch_samples = [] # List of (df, target_idx, true_dir, feature_views)
        
        # 1. Collect a batch of valid scenarios and PRE-CALCULATE features
        while len(batch_samples) < current_batch_size:
            f_idx = np.random.randint(0, len(all_files))
            fpath = all_files[f_idx]
            
            if fpath in common_ds.cache:
                df = common_ds.cache[fpath]
            else:
                df = pd.read_parquet(fpath)
                common_ds.cache[fpath] = df
            
            if len(df) < 500: continue
            
            max_lb = 365
            if len(df) <= max_lb + 2: continue
            
            start_idx = np.random.randint(50, len(df) - max_lb - 1)
            target_idx = start_idx + max_lb
            
            curr = df.iloc[target_idx - 1]['close']
            future = df.iloc[target_idx]['close']
            true_dir = 1.0 if future > curr else 0.0
            
            # Pre-calc all views needed by all models
            views = {}
            for fs_type in ['raw', 'tech', 'sentiment']:
                # Note: We need to know lookback for the view? 
                # No, we can store the whole calc_df or just the max lookback slice.
                # Since models have different lookbacks, we'll store the calc_df or enough for calc.
                # Actually, simpler: Since we know the models' lookbacks, we can just slice then.
                # But indicators need buffer.
                calc_start = max(0, target_idx - 365 - 50)
                calc_df = df.iloc[calc_start : target_idx].copy()
                
                if fs_type == 'tech':
                    calc_df['rsi'] = common_ds._compute_rsi(calc_df['close'])
                    m, s = common_ds._compute_macd(calc_df['close'])
                    calc_df['macd'] = m
                    calc_df['macd_signal'] = s
                    calc_df['sma'] = calc_df['close'].rolling(20).mean()
                    calc_df = calc_df.ffill().bfill()
                elif fs_type == 'sentiment':
                    calc_df['volatility_proxy'] = (calc_df['high'] - calc_df['low']) / calc_df['close']
                    calc_df['volume_shock'] = calc_df['volume'] / calc_df['volume'].rolling(20).mean()
                    calc_df = calc_df.fillna(0)
                
                views[fs_type] = calc_df
                
            batch_samples.append((target_idx, true_dir, views))

        # 2. For each model, generate predictions
        batch_preds = np.zeros((current_batch_size, len(valid_models)))
        
        for m_idx, (mid, model, cfg) in enumerate(valid_models):
            lb = cfg['lookback']
            fs = cfg['feature_set']
            
            model_batch_tensors = []
            for _, _, views in batch_samples:
                calc_df = views[fs]
                w = calc_df.iloc[-lb:]
                
                if fs == 'tech':
                    feats = w[['close', 'high', 'low', 'open', 'volume', 'rsi', 'macd', 'macd_signal', 'sma']].values
                elif fs == 'sentiment':
                    feats = w[['close', 'high', 'low', 'open', 'volume', 'volatility_proxy', 'volume_shock']].values
                else:
                    feats = w[['close', 'high', 'low', 'open', 'volume']].values
                
                model_batch_tensors.append(torch.tensor(feats, dtype=torch.float32))
            
            x_batch = torch.stack(model_batch_tensors).to(device)
            with torch.no_grad():
                d, v = model(x_batch)
                batch_preds[:, m_idx] = d.squeeze().cpu().numpy()

        # Add to main collection
        for i in range(current_batch_size):
            X_stack.append(batch_preds[i].tolist())
            y_stack.append(batch_samples[i][1])
            
        pbar.update(current_batch_size)
        
    pbar.close()
    
    # Train Stacker
    print("Training Meta-Judge (Linear Regression)...")
    X = np.array(X_stack) # (Samples, Models)
    y = np.array(y_stack)
    
    meta_model = LinearRegression()
    meta_model.fit(X, y)
    
    print("Judge Training Complete.")
    print("Model Weights (Trust Scores):")
    for (mid, _, _), coef in zip(valid_models, meta_model.coef_):
        print(f" - {mid}: {coef:.4f}")
        
    joblib.dump(meta_model, STACKER_PATH)
    print(f"Judge saved to {STACKER_PATH}")

if __name__ == "__main__":
    run_judge()
