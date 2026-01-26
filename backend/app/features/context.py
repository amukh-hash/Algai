import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

DATA_DIR = os.path.join("backend", "data_cache_alpaca")
MANIFEST_PATH = os.path.join("backend", "data", "microcosm_manifest.json")

def load_microcosm_data():
    """
    Loads all 100 tickers into a dictionary of DataFrames.
    Optimized to load only necessary columns for context calculation.
    """
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError("Microcosm manifest not found.")
        
    with open(MANIFEST_PATH, 'r') as f:
        data = json.load(f)
        tickers = list(set(data['leaders'] + data['vol_beasts'] + data['liquidity_proxies']))
    
    dfs = {}
    print(f"Loading {len(tickers)} Microcosm tickers for Context Engineering...")
    
    for ticker in tqdm(tickers):
        fpath = os.path.join(DATA_DIR, f"{ticker}_1m.parquet")
        if os.path.exists(fpath):
            # Load Open, High, Low, Close for BPI and RS
            try:
                df = pd.read_parquet(fpath, columns=['open', 'high', 'low', 'close'])
                # Resample or align? 
                # Alpaca 1m data might have gaps. We need a unified index.
                dfs[ticker] = df
            except Exception as e:
                print(f"Error loading {ticker}: {e}")
        else:
            print(f"Warning: Missing data for {ticker}")
            
    return dfs

def create_unified_index(dfs):
    """
    Creates a master DatetimeIndex covering the range of all data.
    """
    min_date = min([df.index.min() for df in dfs.values()])
    max_date = max([df.index.max() for df in dfs.values()])
    # Market minutes only? Or just 1T resample?
    # Simple 1T resample for now, forward fill is risky for gap detection but okay for breadth.
    return pd.date_range(min_date, max_date, freq='1min')

def calculate_market_breadth(dfs, master_index):
    """
    Calculates Advance/Decline Line (AD_Line).
    (Count Up - Count Down) / Total
    """
    print("Calculating Market Breadth (AD Line)...")
    
    # We need a DataFrame where columns are Tickers, rows are Close prices
    # This might be RAM intensive for 100 * 10 years.
    # 10 years * 1.5M rows * 100 floats = 1.2GB. Safe.
    
    closes = pd.DataFrame(index=master_index)
    for t, df in dfs.items():
        # Reindex to master, ffill limited to 5 mins (don't fill overnight gaps)
        closes[t] = df['close'].reindex(master_index).ffill(limit=5)
        
    # Periodic Returns (1 min)
    returns = closes.diff()
    
    up_counts = (returns > 0).sum(axis=1)
    down_counts = (returns < 0).sum(axis=1)
    # total valid for that minute (not NaN)
    valid_counts = returns.count(axis=1)
    
    ad_line = (up_counts - down_counts) / (valid_counts + 1e-9)
    return ad_line

def calculate_bpi(dfs, master_index):
    """
    Calculates Buying Pressure Index (BPI) using CLV.
    Mean CLV of all 100 stocks.
    """
    print("Calculating Buying Pressure Index (Pseudo-Tick)...")
    
    # We need Close, High, Low
    # Iterate row by row? Too slow.
    # Matrix operations.
    
    clv_sum = pd.Series(0.0, index=master_index)
    count = pd.Series(0, index=master_index)
    
    for t, df in tqdm(dfs.items()):
        # Align
        aligned = df.reindex(master_index).ffill(limit=5)
        
        C = aligned['close']
        H = aligned['high']
        L = aligned['low']
        
        # CLV = ((C - L) - (H - C)) / (H - L)
        # Avoid div by zero (H=L)
        rnge = (H - L).replace(0, np.nan)
        clv = ((C - L) - (H - C)) / rnge
        
        clv = clv.fillna(0) # No range = Neutral pressure
        
        clv_sum += clv
        count += 1 # If we assume all loaded are valid. 
        # Strictly should check if aligned was NaN.
        # But ffill handles small gaps. Major gaps are 0.
    
    bpi = clv_sum / count
    return bpi

def build_context_layer():
    dfs = load_microcosm_data()
    if not dfs: return
    
    master_index = create_unified_index(dfs)
    
    # 1. Breadth
    ad_line = calculate_market_breadth(dfs, master_index)
    
    # 2. BPI
    bpi = calculate_bpi(dfs, master_index)
    
    # 3. SPY Baseline for Relative Strength
    # (RS is calculated per stock at runtime or training time, 
    # but we need to ensure SPY is available globally)
    # Check if SPY is in dfs
    if 'SPY' in dfs:
        spy_close = dfs['SPY']['close'].reindex(master_index).ffill()
    else:
        print("CRITICAL: SPY not found. Cannot calculate Relative Strength.")
        spy_close = pd.Series(1.0, index=master_index) # dummy
    
    # Save Context Artifacts
    context_df = pd.DataFrame({
        'ad_line': ad_line,
        'bpi': bpi,
        'spy_close': spy_close
    }, index=master_index)
    
    # Save to efficient parquet
    # Ensure standard filename
    out_path = os.path.join(DATA_DIR, "market_context_1m.parquet")
    context_df.to_parquet(out_path)
    print(f"Context Layer built and saved to {out_path}")

if __name__ == "__main__":
    build_context_layer()
