import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from chronos import ChronosPipeline

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Constants
DATA_PATH = "backend/data/processed/orthogonal_features_final.parquet"
OUTPUT_PATH = "backend/data/judge_training_data.csv"
BASE_MODEL_ID = "amazon/chronos-t5-large"
CHECKPOINT_DIR = "backend/models/chronos_physics_phase2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_best_model_path():
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            latest = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
            print(f"Found local checkpoint: {latest}")
            return latest
    return BASE_MODEL_ID

def generate_data():
    model_path = get_best_model_path()
    print(f"--- Generating Judge Data using {model_path} on {DEVICE} ---")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data not found at {DATA_PATH}")
        return

    print("Loading Data...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Columns: {df.columns.tolist()}")

    # Fix Column Access
    price_col = 'microprice' if 'microprice' in df.columns else 'close'
    if price_col not in df.columns:
        print(f"CRITICAL: Could not find price column (checked 'microprice', 'close').")
        return
    print(f"Using Price Column: {price_col}")

    if 'ticker' not in df.columns:
        print("Warning: No 'ticker' column. Treating as single sequence.")
        tickers = ['UNKNOWN']
        df['ticker'] = 'UNKNOWN'
    else:
        tickers = df['ticker'].unique()
        
    print(f"Loading Chronos Base: {BASE_MODEL_ID}...")
    
    # 1. Load Full Pipeline (Standard)
    # This gives us the correct wrapper structure (ChronosModel)
    pipeline = ChronosPipeline.from_pretrained(
        BASE_MODEL_ID,
        device_map=DEVICE,
        torch_dtype=torch.float32
    )

    # 2. Inject LoRA Adapters
    if model_path != BASE_MODEL_ID:
        print(f"Loading LoRA adapters from {model_path}...")
        from peft import PeftModel
        try:
            # Inspection: ChronosPipeline.model is likely the ChronosModel wrapper
            # ChronosModel.model is the HuggingFace T5 model (based on error logs)
            
            # Target the HF model inside the wrapper
            # valid hierarchy: pipeline -> .model (ChronosModel) -> .model (T5ForConditionalGeneration)
            
            t5_model = pipeline.model.model 
            
            # Apply LoRA
            peft_model = PeftModel.from_pretrained(t5_model, model_path)
            merged_model = peft_model.merge_and_unload()
            
            # Inject back
            pipeline.model.model = merged_model
            
            print("LoRA Adapters merged and injected successfully.")
        except Exception as e:
            print(f"WARNING: LoRA Injection Failed: {e}")
            print("Debug: valid attrs of pipeline.model:", dir(pipeline.model))
            print(" proceeding with Base Model.")

    all_rows = []
    
    # Parameters
    context_len = 512
    pred_len = 64
    stride = 1024 # Sparse Sampling (was 64) - Reduces data by 16x
    BATCH_SIZE = 8 # Safe batch size
    NUM_SAMPLES = 5 # Reduce paths (was 10) - Reduces compute by 2x
    
    # Import TBM
    from app.targets.triple_barrier import get_daily_vol, apply_triple_barrier

    # Check for existing data to resume
    if os.path.exists(OUTPUT_PATH):
        print(f"Resuming: Found existing data at {OUTPUT_PATH}")
        existing_df = pd.read_csv(OUTPUT_PATH)
        processed_count = len(existing_df)
    else:
        processed_count = 0
        with open(OUTPUT_PATH, 'w') as f:
            f.write("ticker,true_label,true_ret,pred_label,conf_score,prob_buy,prob_sell,prob_neutral,rsi,volatility_proxy,bpi,ad_line,fold\n")

    total_rows_generated = processed_count
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        sub_df = df[df['ticker'] == ticker].copy()
        
        # Prices as Series for TBM
        prices_series = sub_df[price_col]
        prices = prices_series.values
        
        # 1. Compute Volatility & Triple Barrier Labels
        # We do this upfront for the whole series = Fast
        print(f"  Calculating Triple Barriers...")
        vol = get_daily_vol(prices_series, span=100)
        events = apply_triple_barrier(
            prices_series, 
            vol, 
            vertical_barrier_steps=pred_len, # 64
            barrier_width_multiplier=2.0 
        )
        # events columns: label, trade_end_idx, ret
        # aligned with df index
        
        n = len(prices)
        if n < context_len + pred_len:
            continue
            
        # Collect Valid Indices
        valid_indices = []
        for i in range(context_len, n - pred_len, stride):
            idx_entry = i - 1
            if idx_entry < len(events):
                valid_indices.append(i)

        # Batch Processing
        for batch_start in tqdm(range(0, len(valid_indices), BATCH_SIZE), desc=f"{ticker}"):
            batch_idxs = valid_indices[batch_start : batch_start + BATCH_SIZE]
            
            # Prepare Batch Input
            contexts = []
            for i in batch_idxs:
                # Context Window
                ctx = prices[i-context_len : i]
                contexts.append(torch.tensor(ctx, dtype=torch.float32))
            
            if not contexts: continue
            
            # Chronos Inference (Batched)
            try:
                # List of tensors
                forecasts = pipeline.predict(
                    contexts,
                    prediction_length=pred_len,
                    num_samples=NUM_SAMPLES
                ) # (B, num_samples, pred_len)
            except Exception as e:
                print(f"Batch Error: {e}")
                continue
                
            # Process Batch Results
            batch_rows = []
            for b, i in enumerate(batch_idxs):
                idx_entry = i - 1
                row_label = events['label'].iloc[idx_entry]
                row_ret = events['ret'].iloc[idx_entry]
                true_label = int(row_label)
                ctx_row = sub_df.iloc[i-1]
                
                # Forecast Analysis
                forecast = forecasts[b] # (num_samples, pred_len)
                
                # Median path return
                median_path = torch.median(forecast, dim=0).values.numpy() # dim=0 across samples
                pred_return = (median_path[-1] - prices[i-1]) / prices[i-1]
                
                # Quantiles
                low_path = torch.quantile(forecast, 0.1, dim=0).numpy()
                high_path = torch.quantile(forecast, 0.9, dim=0).numpy()
                spread = np.mean(high_path - low_path)
                 
                # Pred Label
                thresh = 0.001
                if pred_return > thresh: pred_label = 1
                elif pred_return < -thresh: pred_label = 2
                else: pred_label = 0
                
                # Probabilities
                paths = forecast.numpy() # (num_samples, 64)
                final_rets = (paths[:, -1] - prices[i-1]) / prices[i-1]
                prob_buy = np.mean(final_rets > thresh)
                prob_sell = np.mean(final_rets < -thresh)
                prob_neutral = 1.0 - prob_buy - prob_sell
                
                row_str = f"{ticker},{true_label},{row_ret:.6f},{pred_label},{1.0 - (spread / prices[i-1]):.4f},{prob_buy:.2f},{prob_sell:.2f},{prob_neutral:.2f},{ctx_row.get('rsi', 50)},{ctx_row.get('volatility_proxy', 0)},{ctx_row.get('bpi', 50)},{ctx_row.get('ad_line', 0)},0\n"
                batch_rows.append(row_str)
            
            # Incremental Save
            with open(OUTPUT_PATH, 'a') as f:
                for r in batch_rows:
                    f.write(r)
            
            total_rows_generated += len(batch_rows)

    print(f"Finished. Total rows: {total_rows_generated}")

if __name__ == "__main__":
    generate_data()
