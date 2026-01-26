import os
import glob
import numpy as np
import pandas as pd
import polars as pl
import databento as db
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from concurrent.futures import ProcessPoolExecutor
from fredapi import Fred
from tqdm import tqdm

# Config
DATA_DIR = r"C:\Users\Aishik\Documents\Workshop\Algai\Data\Extracted"
OUTPUT_FILE = r"backend\data\processed\orthogonal_features_final.parquet"
FRED_API_KEY = os.getenv("FRED_API_KEY")

def fast_fracdiff(series, d=0.4, window=100):
    """
    Vectorized Fixed-Window Fractional Differentiation.
    Preserves memory/trends while making data stationary.
    """
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (d - k + 1) / k)
    weights = np.array(weights[::-1]) # Reverse for convolution

    # Convolution
    # Same as 'valid' mode but we want same length with NaNs at start
    res = np.convolve(series, weights, mode='valid')
    # Pad input with NaNs
    res = np.concatenate([np.full(window-1, np.nan), res])
    return res

import gc

def process_daily_dbn(file_path):
    """
    Worker function to process a single day's DBN file.
    Runs on individual cores of the 9800X3D.
    """
    # Disable Polars threading in workers to prevent contention/OOM
    os.environ["POLARS_MAX_THREADS"] = "1"

    try:
        # Load DBN and convert to Polars (faster for microstructure)
        stored_data = db.DBNStore.from_file(file_path)
        df_pd = stored_data.to_df()
        df = pl.from_pandas(df_pd)

        # Free memory immediately
        del stored_data
        del df_pd
        gc.collect()

        # Databento MBP-10 cols: bid_sz_00, ask_sz_00, bid_px_00, ask_px_00, etc.
        # Check col names
        if "bid_sz_00" not in df.columns:
            # Maybe flat format?
            return None

        # Calculate Multi-Level Order Flow Imbalance (OFI)
        # We look at the change in size at the best bid/ask
        # OFI = e_t * q_t
        # Simplified vectorized:
        df = df.with_columns([
            (pl.col("bid_sz_00").diff().fill_null(0) -
             pl.col("ask_sz_00").diff().fill_null(0)).alias("OFI_L1"),
            ((pl.col("bid_px_00") * pl.col("ask_sz_00") +
              pl.col("ask_px_00") * pl.col("bid_sz_00")) /
             (pl.col("bid_sz_00") + pl.col("ask_sz_00"))).alias("microprice")
        ])

        # Resample to 1-second bars
        # 'ts_event' is nanoseconds int usually in Databento pandas
        # Convert to datetime if int
        if df["ts_event"].dtype != pl.Datetime:
             df = df.with_columns(pl.from_epoch(pl.col("ts_event"), time_unit="ns"))

        # Ensure sorted for group_by_dynamic
        df = df.sort("ts_event")

        df_1s = df.group_by_dynamic("ts_event", every="1s").agg([
            pl.col("OFI_L1").sum(),
            pl.col("microprice").mean(),
            pl.col("price").last().alias("close")
        ])

        return df_1s.to_pandas()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def get_macro_regimes(start_date, end_date):
    """Pull VIX and Yield Curve from FRED"""
    if not FRED_API_KEY:
        print("WARNING: FRED_API_KEY not found in .env. Skipping Macro Features.")
        # Return dummy DataFrame to allow join
        dates = pd.date_range(start_date, end_date)
        return pd.DataFrame({'vix': 0.0, 'yield_spread': 0.0}, index=dates)

    try:
        fred = Fred(api_key=FRED_API_KEY)
        vix = fred.get_series('VIXCLS', start_date, end_date)
        yield_curve = fred.get_series('T10Y2Y', start_date, end_date)
        macro = pd.DataFrame({'vix': vix, 'yield_spread': yield_curve}).ffill()
        return macro
    except Exception as e:
        print(f"Error fetching Macro data: {e}. Using placeholders.")
        dates = pd.date_range(start_date, end_date)
        return pd.DataFrame({'vix': 0.0, 'yield_spread': 0.0}, index=dates)

def select_orthogonal_features(df, threshold=0.5):
    """
    Groups features by correlation and selects one representative per cluster.
    """
    print("Performing Feature Orthogonality Check...")
    # 1. Calculate the Correlation Matrix (Spearman handles non-linear better)
    corr = df.corr(method='spearman').fillna(0)

    # 2. Convert correlation to a distance matrix
    dist_matrix = 1 - np.abs(corr)
    dist_matrix = np.clip(dist_matrix, 0, 1) # Ensure valid range
    np.fill_diagonal(dist_matrix, 0) # Fix floating point errors causing Scipy crash

    # 3. Perform Ward's Linkage Clustering
    linkage_matrix = sch.ward(squareform(dist_matrix))

    # 4. Extract Clusters
    cluster_labels = sch.fcluster(linkage_matrix, threshold, criterion='distance')

    selected_features = []
    for cluster_id in np.unique(cluster_labels):
        features_in_cluster = corr.columns[cluster_labels == cluster_id]
        # Heuristic: Pick feature with highest variance or simply first
        selected_features.append(features_in_cluster[0])

    print(f"Reduced {len(corr.columns)} features to {len(selected_features)} orthogonal features.")
    print(f"Selected: {selected_features}")
    return df[selected_features]

def main():
    print(f"Scanning {DATA_DIR} for DBN files...")
    # Recursively find all .dbn files (including compressed)
    files = glob.glob(os.path.join(DATA_DIR, "**", "*.dbn*"), recursive=True)
    if not files:
        # Fallback to the zip name subdir?
        files = glob.glob(os.path.join(DATA_DIR, "*", "*.dbn*"), recursive=True)

    if not files:
        print("No .dbn files found. Check extraction content.")
        # Debug: list dirs
        print("Dirs found:", [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
        return

    print(f"Found {len(files)} files. Processing with 1 worker (Serial Mode for Memory Safety)...")

    results = []
    # Reduced workers to avoid OOM/Contention
    with ProcessPoolExecutor(max_workers=1) as executor:
        for res in tqdm(executor.map(process_daily_dbn, files), total=len(files)):
            if res is not None and not res.empty:
                results.append(res)

    if not results:
        print("No data processed successfully.")
        return

    print("Concatenating results...")
    full_df = pd.concat(results).sort_values("ts_event")
    full_df.set_index("ts_event", inplace=True)

    print("Calculating Fractional Differentiation (d=0.4)...")
    full_df = full_df.dropna()
    full_df['close_frac'] = fast_fracdiff(full_df['close'].values, d=0.4, window=100)

    print("Fetching Macro Data...")
    macro_data = get_macro_regimes(full_df.index.min(), full_df.index.max())

    # Merge on date
    full_df['date'] = full_df.index.date
    # Reset index to merge, then set back? Or merge on column.

    # Macro index is Datetime usually Daily.
    # Convert macro index to Date
    macro_data.index = pd.to_datetime(macro_data.index).date
    macro_data.index.name = 'date'

    full_df = full_df.reset_index().merge(macro_data, on='date', how='left').set_index("ts_event")
    full_df.drop(columns=['date'], inplace=True)
    full_df = full_df.ffill().fillna(0) # Fill missing macro

    # Orthogonality
    final_df = select_orthogonal_features(full_df)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"Saving {len(final_df)} rows to {OUTPUT_FILE}...")
    final_df.to_parquet(OUTPUT_FILE)
    print("Done.")

if __name__ == "__main__":
    if hasattr(os, 'fork'):
        # Unix fix for spawn
        pass
    main()
