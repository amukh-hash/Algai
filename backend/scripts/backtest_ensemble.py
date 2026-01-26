import os
import sys
import json
import joblib
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.models.patchtst import HybridPatchTST

MANIFEST_PATH = "backend/models/ensemble_manifest.json"
STACKER_PATH = "backend/models/stacker.pkl"
DATA_DIR = "backend/data_cache_alpaca"

class SimpleAuditDataset(Dataset):
    """
    Just returns indices. Actual data slicing happens in the main loop
    to allow grouping models by configuration.
    """
    def __init__(self, num_samples, valid_samples):
        self.num_samples = num_samples
        self.valid_samples = valid_samples # list of (file_idx, start_row_idx)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return the indices needed to fetch data
        file_idx, start_idx = self.valid_samples[idx]
        return file_idx, start_idx, idx # include idx to map back to results

def compute_technical_features(df):
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # SMA
    df['sma'] = df['close'].rolling(window=20).mean()

    # Sentiment / Volatility Proxies
    df['volatility_proxy'] = (df['high'] - df['low']) / df['close']
    df['volume_shock'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)

    return df.bfill().ffill().fillna(0) # Safety fill

def get_feature_cols(fs):
    if fs == 'tech':
        return ['close', 'high', 'low', 'open', 'volume', 'rsi', 'macd', 'macd_signal', 'sma']
    elif fs == 'sentiment':
        return ['close', 'high', 'low', 'open', 'volume', 'volatility_proxy', 'volume_shock']
    else: # raw
        return ['close', 'high', 'low', 'open', 'volume']

def run_audit(num_samples=10000):
    print(f"--- STARTING THE TURBO AUDIT (Batch Optimized, {num_samples} Samples) ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Registry & Group Models
    if not os.path.exists(MANIFEST_PATH):
        print("Manifest missing!"); return
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)

    # Sort/Group models by (lookback, feature_set) to share data batches
    model_groups = {}
    for i, entry in enumerate(manifest):
        key = (entry['config']['lookback'], entry['config']['feature_set'])
        if key not in model_groups: model_groups[key] = []
        model_groups[key].append((i, entry))

    print(f"Grouped {len(manifest)} models into {len(model_groups)} configurations.")

    # 2. Data Loading & Pre-processing
    print("Pre-loading data...")
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".parquet")]
    if not files: print("No data!"); return

    dfs = [] # List of DataFrames
    valid_ranges = [] # Lists of (file_idx, start_row_idx) for sampling

    max_lookback = 365
    horizon = 60

    for i, fpath in enumerate(tqdm(files)):
        try:
            df = pd.read_parquet(fpath)
            if len(df) < max_lookback + horizon + 20: continue

            # Compute ALL features once
            df = compute_technical_features(df)

            # Convert to float32 numpy for speed
            # We store the WHOLE df in memory.
            # Optimization: keep columns in a dict of numpy arrays for fast slicing?
            # or just DataFrame. Pandas slicing is fast enough if batched properly.
            dfs.append(df)

            # Record valid start indices
            # Valid start: from max_lookback to len-horizon
            # We want to be safe for any lookback <= 365
            start_min = 365
            start_max = len(df) - horizon - 1

            if start_max > start_min:
                # Add a block of potential indices
                # We'll sample from this later
                # For now just store metadata to sample from
                valid_ranges.append((len(dfs)-1, start_min, start_max))
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    if not dfs: print("No valid data!"); return

    # 3. Sample Generation
    print(f"Generating {num_samples} test points...")
    samples = []
    # Weighted sampling based on file length? Or uniform file choice?
    # Simple uniform file choice
    for _ in range(num_samples):
        # Pick random file from valid ones
        f_idx, s_min, s_max = valid_ranges[np.random.randint(len(valid_ranges))]
        # Pick random start
        s_idx = np.random.randint(s_min, s_max)
        samples.append((f_idx, s_idx))

    # Dataset
    ds = SimpleAuditDataset(num_samples, samples)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    # Output Storage
    # matrix: [num_samples, num_models]
    all_preds = np.zeros((num_samples, len(manifest)), dtype=np.float32)
    ground_truths = np.zeros((num_samples, 3), dtype=np.int8) # 5m, 30m, 60m

    # 4. Collection of Ground Truth (Only need to do once)
    print("Collecting Ground Truths...")
    # We can do this iterativley or just quickly via simple loop since no GPU involved
    # But let's use the loader to keep alignment perfect
    for f_idxs, s_idxs, result_idxs in loader:
        # f_idxs, s_idxs are tensors
        f_idxs = f_idxs.numpy()
        s_idxs = s_idxs.numpy()
        r_idxs = result_idxs.numpy()

        for k in range(len(f_idxs)):
            f_idx = f_idxs[k]
            s_idx = s_idxs[k]
            r_idx = r_idxs[k]

            df = dfs[f_idx]
            curr = df['close'].iloc[s_idx]

            # Vectorized 'iloc' is not possible easily across different DF sizes
            # manual lookup is fine for 10k samples (CPU fast enough)
            p5 = df['close'].iloc[s_idx + 5]
            p30 = df['close'].iloc[s_idx + 30]
            p60 = df['close'].iloc[s_idx + 60]

            ground_truths[r_idx, 0] = 1 if p5 > curr else 0
            ground_truths[r_idx, 1] = 1 if p30 > curr else 0
            ground_truths[r_idx, 2] = 1 if p60 > curr else 0

    # 5. Inference Loops (Grouped)
    print("Running Inference...")

    # Pre-extract data buffers? No, too much RAM.
    # Extract batches on the fly.

    for (lb, fs), group_entries in model_groups.items():
        print(f"Processing Group: LB={lb}, FS={fs} ({len(group_entries)} Models)")

        # A. Prepare Models
        loaded_models = []
        for global_idx, entry in group_entries:
             num_feats = 9 if fs == 'tech' else (7 if fs == 'sentiment' else 5)
             m = HybridPatchTST(
                 num_input_features=num_feats,
                 lookback_window=lb,
                 n_heads=entry['config']['n_heads'],
                 d_model=64
             ).to(device)
             m.load_state_dict(torch.load(entry['path'], map_location=device))
             m.eval()
             loaded_models.append((global_idx, m))

        # B. Iterate Batches
        feature_cols = get_feature_cols(fs)

        with torch.no_grad():
            for f_idxs, s_idxs, result_idxs in tqdm(loader, leave=False):
                # Construct Batch Tensor
                # indices are tensors
                f_idxs = f_idxs.numpy()
                s_idxs = s_idxs.numpy()

                # Fetch windows from DFs
                batch_wins = []
                for k in range(len(f_idxs)):
                    f = f_idxs[k]
                    s = s_idxs[k]
                    # df slice
                    # slice: [s - lb : s] (s is current, so s-lb to s.
                    # Wait, lookback includes current? Usually yes.
                    # standard: df.iloc[s - lb + 1 : s + 1] -> includes s
                    win = dfs[f][feature_cols].iloc[s - lb + 1 : s + 1].values
                    batch_wins.append(win)

                # Stack
                X = torch.tensor(np.array(batch_wins), dtype=torch.float32).to(device)

                # Inference for all models in this group
                for g_idx, model in loaded_models:
                    out, _ = model(X)
                    # out: [Batch, 1]
                    preds = out.squeeze().cpu().numpy()

                    # unique indices for this batch
                    # result_idxs maps to the global all_preds row
                    all_preds[result_idxs, g_idx] = preds

        # Clean up models
        del loaded_models
        torch.cuda.empty_cache()

    # 6. Judge
    print("Judging...")
    stacker = joblib.load(STACKER_PATH)
    judge_preds = stacker.predict(all_preds)
    avg_preds = np.mean(all_preds, axis=1)

    # 7. Metrics
    stats = {h: {'avg': 0, 'judge': 0} for h in ['5m', '30m', '60m']}

    for i in range(num_samples):
        # 0=5m, 1=30m, 2=60m
        for h_i, h_name in enumerate(['5m', '30m', '60m']):
            truth = ground_truths[i, h_i]

            p_avg = 1 if avg_preds[i] > 0.5 else 0
            p_jdg = 1 if judge_preds[i] > 0.5 else 0

            if p_avg == truth: stats[h_name]['avg'] += 1
            if p_jdg == truth: stats[h_name]['judge'] += 1

    # Report
    print("\n" + "="*60)
    print(f"--- TURBO AUDIT REPORT (Final, {num_samples} Samples) ---")
    print(f"{'Horizon':<10} | {'Ensemble Avg Accuracy':<25} | {'Judge Accuracy':<20}")
    print("-" * 60)
    for h in ['5m', '30m', '60m']:
        avg_acc = (stats[h]['avg'] / num_samples) * 100
        jdg_acc = (stats[h]['judge'] / num_samples) * 100
        diff = jdg_acc - avg_acc
        sign = "+" if diff >= 0 else ""
        print(f"{h:<10} | {avg_acc:>22.2f}% | {jdg_acc:>17.2f}% ({sign}{diff:.2f})")
    print("="*60)

if __name__ == "__main__":
    c = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    run_audit(c)
