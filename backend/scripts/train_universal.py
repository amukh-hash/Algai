import os
import sys
import glob
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.models.patchtst import HybridPatchTST
from app.core.trainer import Trainer
from app.core.loss import UniversalLoss
from app.targets.triple_barrier import get_daily_vol, apply_triple_barrier, get_purged_indices

import joblib

DATA_DIR = os.path.join("backend", "data_cache_alpaca")
CONTEXT_PATH = os.path.join(DATA_DIR, "market_context_1m.parquet")
CACHE_PATH = os.path.join(DATA_DIR, "universal_dataset_cache.pkl")
SAVE_PATH = "backend/models/pretrained_market_physics.pt"

class UniversalDataset(Dataset):
    def __init__(self, tickers=None, lookback=64, barrier_horizon=120, barrier_mult=1.5, force_rebuild=False):
        self.lookback = lookback

        # 0. Check Cache
        if not force_rebuild and os.path.exists(CACHE_PATH):
            print(f"Loading Dataset from Cache: {CACHE_PATH}")
            try:
                cached_data = joblib.load(CACHE_PATH)
                self.samples = cached_data['samples']
                self.dfs = cached_data['dfs']
                self.num_features = cached_data['num_features']
                print(f"Loaded {len(self.samples)} samples from API.")
                return
            except Exception as e:
                print(f"Cache load failed ({e}), rebuilding...")

        # 1. Load Context
        if not os.path.exists(CONTEXT_PATH):
            raise FileNotFoundError("Context data missing. Run context.py first.")

        print("Loading Market Context...")
        self.context_df = pd.read_parquet(CONTEXT_PATH)
        # Columns: ad_line, bpi, spy_close

        # 2. Load Tickers
        if tickers is None:
            # Load all in manifest or dir
            files = glob.glob(os.path.join(DATA_DIR, "*_1m.parquet"))
            # Filter context file
            files = [f for f in files if "market_context" not in f]
        else:
            files = [os.path.join(DATA_DIR, f"{t}_1m.parquet") for t in tickers]

        print(f"Loading {len(files)} tickers for Universal Pre-Training...")

        self.samples = [] # List of (df_idx, start_idx, label, vol_target)
        self.dfs = [] # List of processed DataFrames

        # We process each file sequentially to save RAM during processing
        # Then store purely numpy arrays?

        for f_idx, fpath in enumerate(tqdm(files)):
            try:
                # Load
                df = pd.read_parquet(fpath)
                if len(df) < lookback + barrier_horizon + 10: continue

                # Align with Context
                # Left join to stock index to keep stock's timeline
                df = df.join(self.context_df, how='inner')
                if len(df) < lookback: continue

                # Feature Engineering (Microstructure + RS + Tech)
                # 1. Rel Strength
                if 'spy_close' in df.columns and 'close' in df.columns:
                    df['rs'] = np.log(df['close'] + 1e-9) - np.log(df['spy_close'] + 1e-9)
                else:
                    df['rs'] = 0.0

                # 2. Tech Indicators (Standard Set for Universal)
                # RSI, MACD, SMA
                # Minimal set to not explode feature dim
                close = df['close']

                # Fast RSI
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).fillna(0)
                loss = (-delta.where(delta < 0, 0)).fillna(0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / (avg_loss + 1e-9)
                df['rsi'] = 100 - (100 / (1 + rs))

                # SMA Distance
                df['sma_dist'] = close / (close.rolling(20).mean() + 1e-9) - 1

                # Fill NaNs
                df = df.fillna(0)

                # 3. Triple Barrier Labeling
                vol = get_daily_vol(close)
                # apply barrier
                tb_df = apply_triple_barrier(close, vol, barrier_horizon, barrier_mult)

                df['label'] = tb_df['label']
                df['trade_end_idx'] = tb_df['trade_end_idx']
                df['vol_target'] = vol # For auxiliary head

                # 4. Purging
                valid_indices = get_purged_indices(tb_df)

                # Store processed DF (float32 to save RAM)
                # Columns: Open, High, Low, Close, Volume, BPI, AD_Line, RS, RSI, SMA_Dist
                # 5 + 2 + 1 + 2 = 10 Features
                # We normalize logic later? RevIN does it.
                # Just keep values.

                FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume',
                                'bpi', 'ad_line', 'rs', 'rsi', 'sma_dist']

                data_matrix = df[FEATURE_COLS].astype(np.float32).values
                label_arr = df['label'].values.astype(np.int8)
                vol_arr = df['vol_target'].values.astype(np.float32)

                self.dfs.append(data_matrix)

                # Add valid samples
                # (df_index, start_row_index)
                # valid_indices are the START of the window?
                # apply_triple_barrier iterates 'i'. 'i' is the current time.
                # We need [i - lookback + 1 : i + 1]
                # So valid 'i' must be >= lookback.

                for i in valid_indices:
                    if i >= lookback:
                         self.samples.append((f_idx, i, label_arr[i], vol_arr[i]))

            except Exception as e:
                print(f"Skipping {fpath}: {e}")

        print(f"Dataset Ready. Total Purged Samples: {len(self.samples)}")
        self.num_features = 10

        # Save Cache
        print(f"Saving Cache to {CACHE_PATH}...")
        joblib.dump({
            'samples': self.samples,
            'dfs': self.dfs,
            'num_features': self.num_features
        }, CACHE_PATH, compress=3)
        print("Cache Saved.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_idx, end_idx, label, vol_target = self.samples[idx]
        data = self.dfs[f_idx]

        # Window: end_idx is inclusive?
        # Standard: i is the point of decision.
        # Window is [i - lookback + 1 : i + 1]

        start_idx = end_idx - self.lookback + 1
        x = data[start_idx : end_idx + 1]

        # Convert to tensor
        x_t = torch.tensor(x, dtype=torch.float32)

        # Target: (Label, Volatility)
        # Label is 0, 1, 2

        targets = torch.tensor([label, vol_target], dtype=torch.float32)
        return x_t, targets

def train_universal():
    print("Initializing Universal Pre-Training (Phase A)...")

    # 1. Dataset
    # Use default 64 lookback for universal physics
    ds = UniversalDataset(lookback=64)
    if len(ds) == 0:
        print("Dataset empty. Check data/context.")
        return

    print("\n" + "="*50)
    print(f"Dataset Loaded: {len(ds)} Samples.")
    print("PAUSED: Waiting for user confirmation to start training.")
    print("Press ENTER to begin training...")
    input()
    print("RESUMING: Starting Training Loop...")
    print("="*50 + "\n")

    # Config
    BATCH_SIZE = 512
    EPOCHS = 20 # Pre-training can be long, but we have big data
    LR = 1e-4

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # 0 for simplicity/RAM


    # 2. Model
    # 10 Input Features
    model = HybridPatchTST(
        num_input_features=10,
        lookback_window=64,
        n_heads=8,
        n_layers=4, # Deeper for universal
        d_model=128,
        n_classes=3 # Sell/Neutral/Buy
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}...")

    # 3. Loss & Opt
    criterion = UniversalLoss(num_classes=3, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    # 4. Trainer
    # Trainer expects validation loader usually. Split?
    # For Pre-training, we just want to soak up data. Validation on last 10%?
    # Simple explicit loop or use Trainer? Trainer is nice.
    # Let's split ds.

    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    trainer = Trainer(model, criterion, optimizer, device=device)
    trainer.fit(train_loader, val_loader, epochs=EPOCHS, patience=5)

    # 5. Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Universal Weights saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_universal()
