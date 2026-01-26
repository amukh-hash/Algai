import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.models.patchtst import HybridPatchTST
from app.core.trainer import Trainer
from app.core.loss import UniversalLoss
from scripts.train_universal import UniversalDataset

def generate_metalabeling_data(tickers, k_folds=5, epochs_per_fold=3):
    print(f"--- Generating Judge Data for {len(tickers)} tickers ({k_folds}-Fold CV) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Global collection
    all_rows = []
    
    for ticker in tickers:
        print(f"Processing {ticker}...")
        ds = UniversalDataset(tickers=[ticker], lookback=64)
        if len(ds) < 500:
            print(f"Skipping {ticker} (Too small: {len(ds)})")
            continue
            
        # K-Fold Split
        # We assume samples are independent enough due to purging.
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        indices = np.arange(len(ds))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            # 1. Train Specialist on Train Fold
            # For speed, we initialize from Pretrained Physics if available (Transfer Learning)
            # OR random init if we want pure specialist test. 
            # Plan says: "Train Specialist... Pred on Val".
            # So we load Universal, Fine-tune on Train, Pred on Val.
            
            model = HybridPatchTST(num_input_features=10, lookback_window=64, n_heads=8, d_model=128, n_classes=3)
            univ_path = "backend/models/pretrained_market_physics.pt"
            if os.path.exists(univ_path):
                model.load_state_dict(torch.load(univ_path, map_location=device), strict=False)
            model.to(device)
            
            # Freeze extraction layers for speed/stability
            for p in model.patch_embedding.parameters(): p.requires_grad = False
            for p in model.backbone.encoder.layers[:2].parameters(): p.requires_grad = False
            
            train_loader = DataLoader(Subset(ds, train_idx), batch_size=256, shuffle=True)
            # Validation Loader (to predict)
            val_loader = DataLoader(Subset(ds, val_idx), batch_size=256, shuffle=False)
            
            criterion = UniversalLoss(num_classes=3)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
            
            # Short Training
            # No trainer? Use manual loop for tight control or Trainer.
            # Trainer is fine.
            trainer = Trainer(model, criterion, optimizer, device=device, verbose=False)
            trainer.fit(train_loader, None, epochs=epochs_per_fold, patience=1)
            
            # 2. Predict on Val Fold
            model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    # targets: (label, vol)
                    true_labels = targets[:, 0].numpy()
                    
                    # Forward
                    logits, pred_vol = model(inputs)
                    
                    # Probabilities
                    probs = torch.softmax(logits, dim=1).cpu().numpy() # (B, 3)
                    conf = np.max(probs, axis=1)
                    pred_class = np.argmax(probs, axis=1)
                    
                    # Extract Context Features (BPI, Breadth, VIX/RS?)
                    # Inputs structure: [OHLCV, BPI, AD, RS, RSI, SMA]
                    # indices: 0..4, 5=BPI, 6=AD, 7=RS, 8-9...
                    # We take the LAST time step of the input window for context?
                    # shape: (B, Lookback, Feats)
                    last_step = inputs[:, -1, :].cpu().numpy()
                    bpi = last_step[:, 5]
                    ad_line = last_step[:, 6]
                    rs = last_step[:, 7]
                    rsi_val = last_step[:, 8]
                    
                    for i in range(len(true_labels)):
                        row = {
                            'ticker': ticker,
                            'fold': fold_idx,
                            'true_label': true_labels[i],
                            'pred_label': pred_class[i],
                            'conf_sell': probs[i, 0], # label 0 = Sell? No 0=Neutral? 
                            # Triple Barrier: 0=Neutral, 1=Buy, 2=Sell (mapped via PyTorch indexing?)
                            # Wait, apply_triple_barrier returns 0, 1, 2. 
                            # 0=Neutral, 1=Buy, 2=Sell.
                            # So Class 0: Neutral, Class 1: Buy, Class 2: Sell.
                            
                            'prob_neutral': probs[i, 0],
                            'prob_buy': probs[i, 1],
                            'prob_sell': probs[i, 2],
                            
                            'bpi': bpi[i],
                            'ad_line': ad_line[i],
                            'rs': rs[i],
                            'rsi': rsi_val[i]
                        }
                        all_rows.append(row)
            
        # Clean up per ticker
        del ds
        torch.cuda.empty_cache()
        
    # Save
    df = pd.DataFrame(all_rows)
    out_path = "backend/data/judge_training_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Judge Data Generated: {len(df)} rows. Saved to {out_path}")

if __name__ == "__main__":
    # Representative subset for the Judge
    # Ideally all 100, but takes time.
    # Let's pick Top 5 Liquid + Top 5 Vol
    targets = ["NVDA", "TSLA", "AMD", "COIN", "MSTR", "SPY", "QQQ", "AAPL", "MSFT", "IWM"]
    generate_metalabeling_data(targets, k_folds=3, epochs_per_fold=1) # Fast run for testing
