import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.models.patchtst import HybridPatchTST
from app.core.trainer import Trainer

class SingleStockDataset(Dataset):
    """
    Loads data for a single stock for fine-tuning.
    """
    def __init__(self, file_path, lookback=64):
        self.df = pd.read_parquet(file_path)
        self.lookback = lookback
        # Normalize columns if needed, here assuming pre-normalized or raw
        # Using simple OHLCV
        self.features = self.df[['close', 'high', 'low', 'open', 'volume']].values

    def __len__(self):
        return len(self.df) - self.lookback

    def __getitem__(self, idx):
        window = self.features[idx : idx + self.lookback]

        # Target
        curr_close = self.features[idx + self.lookback - 1][0] # Close is col 0
        next_close = self.features[idx + self.lookback][0]

        direction = 1.0 if next_close > curr_close else 0.0
        volatility = abs((next_close - curr_close) / curr_close)

        return torch.tensor(window, dtype=torch.float32), torch.tensor([direction, volatility], dtype=torch.float32)

def train_finetune(symbol="AAPL"):
    DATA_PATH = os.path.join("backend", "data_cache", f"{symbol}_1h.parquet")
    BASE_MODEL_PATH = "backend/models/global_base.pth"

    if not os.path.exists(DATA_PATH):
        print(f"Data for {symbol} not found at {DATA_PATH}")
        return

    if not os.path.exists(BASE_MODEL_PATH):
        print("Global Base Model not found! Please run train_global.py first.")
        return

    # 1. Dataset
    dataset = SingleStockDataset(DATA_PATH, lookback=64)
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # 2. Model (Init with LoRA enabled)
    model = HybridPatchTST(
        num_input_features=5,
        lookback_window=64,
        d_model=64,
        n_layers=2,
        dropout=0.1,
        use_lora=True, # Enable LoRA architecture
        lora_rank=4
    )

    # 3. Load Base Weights
    # We need to filter out LoRA keys if they weren't in base,
    # OR we load base first then inject LoRA.
    # The model._inject_lora() is called in __init__ if use_lora=True.
    # So the state_dict keys will possess lora params.
    # The base model checkpoint won't have them.
    # We load with strict=False to ignore missing LoRA keys in checkpoint.
    state_dict = torch.load(BASE_MODEL_PATH)
    model.load_state_dict(state_dict, strict=False)
    print("Loaded Global Base Model.")

    # 4. Freeze Non-LoRA parameters
    # Mark only LoRA layers as trainable
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if "lora" in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    print(f"Trainable Params: {trainable_params} / {all_params} ({(trainable_params/all_params):.2%})")

    # 5. Training
    class HybridLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.mse = nn.MSELoss()

        def forward(self, predictions, targets):
            pred_dir, pred_vol = predictions
            target_dir = targets[:, 0].unsqueeze(1)
            target_vol = targets[:, 1].unsqueeze(1)
            return self.mse(pred_dir, target_dir) + self.mse(pred_vol, target_vol)

    criterion = HybridLoss()
    # Weight Decay
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0.01)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    trainer = Trainer(model, criterion, optimizer, device=device)

    print(f"Starting Fine-Tuning for {symbol}...")
    trainer.fit(train_loader, val_loader, epochs=100, patience=20)

    # 6. Save Adapter Only
    formatted_symbol = symbol.replace("=", "").replace(";", "") # Sanitize
    save_path = f"backend/models/lora_{formatted_symbol}.pth"

    # Save only state dict items with 'lora' in name
    lora_state = {k: v for k, v in model.state_dict().items() if "lora" in k}
    torch.save(lora_state, save_path)
    print(f"Specialist Adapter saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_finetune(sys.argv[1])
    else:
        train_finetune("AAPL")
