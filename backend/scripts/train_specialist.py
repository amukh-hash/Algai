import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.models.patchtst import HybridPatchTST
from app.core.trainer import Trainer
from app.core.loss import UniversalLoss
from scripts.train_universal import UniversalDataset

UNIVERSAL_PATH = "backend/models/pretrained_market_physics.pt"

def train_specialist(ticker, epochs=5, lr=5e-5, freeze_ratio=0.5):
    print(f"--- Training Specialist for {ticker} ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Dataset (Single Ticker)
    specialist_ds = UniversalDataset(tickers=[ticker], lookback=64)
    if len(specialist_ds) == 0:
        print(f"No data for {ticker}")
        return

    # Split
    train_size = int(0.9 * len(specialist_ds))
    val_size = len(specialist_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(specialist_ds, [train_size, val_size])

    loader_train = DataLoader(train_ds, batch_size=256, shuffle=True)
    loader_val = DataLoader(val_ds, batch_size=256)

    # 2. Model (Universal Architecture)
    model = HybridPatchTST(
        num_input_features=10,
        lookback_window=64,
        n_heads=8,
        n_layers=4,
        d_model=128,
        n_classes=3
    )

    # 3. Load Pretrained Weights
    if os.path.exists(UNIVERSAL_PATH):
        print(f"Loading Universal Physics from {UNIVERSAL_PATH}")
        # Use strict=False if we change heads, but here arch is same
        model.load_state_dict(torch.load(UNIVERSAL_PATH, map_location=device))
    else:
        print("Universal weights not found! Training from scratch (Not Recommended).")

    model.to(device)

    # 4. Freeze Low-Level Layers (Transfer Learning)
    # Freeze PatchEmbedding
    for p in model.patch_embedding.parameters():
        p.requires_grad = False

    # Freeze lower Encoder layers
    # model.backbone.encoder is nn.TransformerEncoder
    num_layers = len(model.backbone.encoder.layers)
    num_freeze = int(num_layers * freeze_ratio)

    print(f"Freezing {num_freeze}/{num_layers} Encoder Layers...")
    for i in range(num_freeze):
        for p in model.backbone.encoder.layers[i].parameters():
            p.requires_grad = False

    # Optimizer (Only optimize trainable)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Loss
    criterion = UniversalLoss(num_classes=3, gamma=2.0)

    # 5. Fine-Tune
    trainer = Trainer(model, criterion, optimizer, device=device)
    trainer.fit(loader_train, loader_val, epochs=epochs, patience=2)

    # 6. Save Specialist
    save_path = f"backend/models/specialists/specialist_{ticker}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Specialist saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", type=str, help="Ticker symbol to fine-tune on")
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs")
    args = parser.parse_args()

    train_specialist(args.ticker, epochs=args.epochs)
