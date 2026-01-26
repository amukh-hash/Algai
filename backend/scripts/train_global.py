import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import glob
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.models.patchtst import HybridPatchTST
from app.core.trainer import Trainer

import psutil

class InMemoryGlobalDataset(Dataset):
    """
    High-Performance RAM Dataset with Hybrid Fallback.
    Loads data into RAM until cap (default 20GB) is reached.
    Remaining files are read from disk (slower but safe).
    """
    def __init__(self, data_dir, lookback=64, samples_per_epoch=10000, feature_set='raw', ram_cap_gb=20):
        # Prefer Alpaca directory if exists, else default
        alpaca_dir = os.path.join(os.path.dirname(data_dir), "data_cache_alpaca")
        if os.path.exists(alpaca_dir) and len(os.listdir(alpaca_dir)) > 0:
            self.data_dir = alpaca_dir
            print(f"Dataset Source: Alpaca History ({alpaca_dir})")
        else:
            self.data_dir = data_dir
            print(f"Dataset Source: Generic Cache ({data_dir})")

        self.files = glob.glob(os.path.join(self.data_dir, "*.parquet"))
        self.lookback = lookback
        self.samples_per_epoch = samples_per_epoch
        self.feature_set = feature_set
        
        # In-Memory Cache
        self.cache = {} # path -> dataframe
        self.ram_files = [] # list of paths in RAM
        self.disk_files = [] # list of paths on Disk
        
        print(f"Loading Dataset into RAM (Cap: {ram_cap_gb} GB)...")
        
        for fpath in self.files:
            # Check RAM
            mem = psutil.virtual_memory()
            used_gb = (mem.total - mem.available) / (1024**3)
            # Or just check process memory? simpler to check system available
            # If available < 4GB, stop
            if mem.available < 4 * (1024**3) or used_gb > ram_cap_gb:
                self.disk_files.append(fpath)
            else:
                try:
                    df = self._load_and_process(fpath)
                    if df is not None:
                        self.cache[fpath] = df
                        self.ram_files.append(fpath)
                    else:
                        # Too short
                        pass
                except Exception as e:
                    print(f"Failed to load {fpath}: {e}")

        print(f"RAM Loaded: {len(self.ram_files)} files. Disk Fallback: {len(self.disk_files)} files.")
        self.all_files = self.ram_files + self.disk_files

    def __len__(self):
        return self.samples_per_epoch

    def _load_and_process(self, fpath):
        """
        Reads parquet, computes features, returns DF.
        """
        df = pd.read_parquet(fpath)
        
        # Buffer for tech indicators
        buffer = 50
        if len(df) < self.lookback + buffer:
             return None
             
        # Feature Engineering (Once at load time for RAM files!)
        # IMPORTANT: efficient storage.
        # If we augment here, we augment ONCE.
        # Ideally we compute raw indicators here, and augment in __getitem__.
        
        if self.feature_set == 'tech':
            df['rsi'] = self._compute_rsi(df['close'])
            macd, sig = self._compute_macd(df['close'])
            df['macd'] = macd
            df['macd_signal'] = sig
            df['sma'] = df['close'].rolling(window=20).mean()
            df = df.bfill().ffill()
            
        elif self.feature_set == 'sentiment':
            df['volatility_proxy'] = (df['high'] - df['low']) / df['close']
            df['volume_shock'] = df['volume'] / df['volume'].rolling(20).mean()
            df = df.fillna(0)
            
        return df

    def _compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _compute_macd(self, series):
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
        
    def __getitem__(self, idx):
        # 1. Pick random file
        fpath = self.all_files[idx % len(self.all_files)]
        
        # 2. Get Data (RAM or Disk)
        if fpath in self.cache:
            df = self.cache[fpath]
        else:
            # Disk Read (Hybrid Fallback)
            try:
                df = self._load_and_process(fpath)
                if df is None: return self._empty_sample()
            except:
                return self._empty_sample()

        try:
            # 3. Sampling
            # We already processed features in _load_and_process
            # Just slice
            
            # Valid range
            buffer = 50 # Already handled in _load_and_process check, but double check
            if len(df) < self.lookback + 1: return self._empty_sample()
            
            # Strided Sampling logic
            stride = self.lookback * 2
            max_start = len(df) - self.lookback
            # We can start anywhere since indicators are computed
            # But stick to stride for separation compliance
            valid_starts = range(0, max_start, stride)
             
            if len(valid_starts) == 0: return self._empty_sample()
            
            rand_k = torch.randint(0, len(valid_starts), (1,)).item()
            start_idx = valid_starts[rand_k]
            
            window = df.iloc[start_idx : start_idx + self.lookback]
            
            # Select Columns
            if self.feature_set == 'tech':
                feats = window[['close', 'high', 'low', 'open', 'volume', 'rsi', 'macd', 'macd_signal', 'sma']].values
            elif self.feature_set == 'sentiment':
                feats = window[['close', 'high', 'low', 'open', 'volume', 'volatility_proxy', 'volume_shock']].values
            else:
                feats = window[['close', 'high', 'low', 'open', 'volume']].values
            
            # 4. Augmentation (Runtime)
            noise = np.random.normal(0, 0.01, feats.shape)
            feats = feats * (1 + noise)
            
            # 5. Target
            next_step = df.iloc[start_idx + self.lookback]
            curr_close = window.iloc[-1]['close']
            next_close = next_step['close']
            
            ret = (next_close - curr_close) / curr_close
            direction_soft = torch.sigmoid(torch.tensor(ret * 200)).item()
            volatility = abs(ret)
            
            return torch.tensor(feats, dtype=torch.float32), torch.tensor([direction_soft, volatility], dtype=torch.float32)
            
        except Exception as e:
            return self._empty_sample()

    def _empty_sample(self):
        return torch.zeros(self.lookback, self._get_num_features(), dtype=torch.float32), torch.tensor([0.5, 0.0], dtype=torch.float32)

    def _get_num_features(self):
        if self.feature_set == 'tech': return 9
        if self.feature_set == 'sentiment': return 7
        return 5

def train_model(config=None):
    if config is None:
        config = {
            'lookback': 64,
            'feature_set': 'raw',
            'n_heads': 4,
            'save_path': "backend/models/global_base.pth"
        }
        
    DATA_DIR = os.path.join("backend", "data_cache")
    
    # 1. Dataset
    dataset = InMemoryGlobalDataset(
        DATA_DIR, 
        lookback=config['lookback'], 
        samples_per_epoch=10000, 
        feature_set=config['feature_set']
    )
    
    # Batch Size 512 + 6 Workers for high throughput
    dataloader = DataLoader(
        dataset, 
        batch_size=512, 
        shuffle=True, 
        num_workers=6, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    num_feats = dataset._get_num_features()
    
    # 2. Model
    model = HybridPatchTST(
        num_input_features=num_feats, 
        lookback_window=config['lookback'], 
        d_model=64, 
        n_layers=2,
        n_heads=config['n_heads'],
        dropout=0.2 
    )
    
    # 3. Training
    class HybridLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.mse = nn.MSELoss()
            
        def forward(self, predictions, targets):
            pred_dir, pred_vol = predictions
            target_dir = targets[:, 0].unsqueeze(1)
            target_vol = targets[:, 1].unsqueeze(1)
            # Simple sum of MSEs
            return self.mse(pred_dir, target_dir) + self.mse(pred_vol, target_vol)

    criterion = HybridLoss()
    
    # Weight Decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device} | Config: {config}")
    trainer = Trainer(model, criterion, optimizer, device=device)
    
    # 1000 Epochs, Patience 50
    trainer.fit(dataloader, dataloader, epochs=1000, patience=50)
    
    # 4. Save
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
    torch.save(model.state_dict(), config['save_path'])
    print(f"Model Saved to {config['save_path']}")

if __name__ == "__main__":
    train_model()
