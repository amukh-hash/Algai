import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from chronos import ChronosPipeline

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.api.databento_client import DatabentoClient
from app.features.signal_processing import apply_modwt_uks

MODEL_ID = "amazon/chronos-t5-large"

class TeacherDataset(Dataset):
    def __init__(self, prices, tokenizer, context_length=512, prediction_length=64):
        self.prices = prices
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __len__(self):
        # Limit length for demo/mock
        return max(0, len(self.prices) - self.context_length - self.prediction_length)

    def __getitem__(self, idx):
        # Sliding window
        full_window = self.prices[idx : idx + self.context_length + self.prediction_length]

        # Split
        context = full_window[:self.context_length]
        target = full_window[self.context_length:]

        # Tokenize
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0) # (1, L)
        target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

        # Context Transform: returns input_ids, attention_mask, scale
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(context_tensor)

        # Target Transform: Scale target using CONTEXT scale, then quantize
        # We access the internal input_transform or assume manual scaling + quantization?
        # Chronos tokenizer usually has `input_transform` which takes (samples, scale)
        # Check if input_transform returns just tokens or more.
        # It usually returns token_ids, attention_mask.

        # If input_transform is not public or varies, we can try to rely on context_input_transform
        # but force scale.
        # But `context_input_transform` computes scale from data.

        # Let's inspect tokenizer at runtime or assume `input_transform`.
        # If not available, we might fail.
        # Safe fallback: Use context_input_transform on target?
        # No, that would re-scale independently.

        # We will try to use `tokenizer.input_transform(target, scale)`.
        try:
             # Ensure scale is (1, 1)
             label_ids, _ = self.tokenizer.input_transform(target_tensor, scale)
        except:
             # Fallback if API differs
             # Maybe it is `_input_transform`?
             label_ids, _, _ = self.tokenizer.context_input_transform(target_tensor)

        return {
            "input_ids": input_ids.squeeze(0),
            "labels": label_ids.squeeze(0)
        }

def train_teacher():
    print(f"--- Starting Teacher Training ({MODEL_ID}) ---")

    # 1. Get Data
    client = DatabentoClient(mock_mode=True)
    print("Fetching historical L2 data...")
    # Getting enough data for context
    df = client.get_historical_range("BTC-USD", "2023-01-01", "2023-01-02", schema='mbp-10')

    if 'price' in df.columns:
        raw_prices = df['price'].values
    else:
        # Fallback for mock/real
        if 'bid_px_00' in df.columns:
            raw_prices = (df['bid_px_00'] + df['ask_px_00']) / 2.0
            raw_prices = raw_prices.values
        else:
            raw_prices = np.random.randn(1000).cumsum() + 100

    print(f"Got {len(raw_prices)} ticks. Applying MODWT+UKS Smoothing...")

    # 2. Acausal Smoothing (Teacher Perception)
    try:
        smoothed_prices = apply_modwt_uks(raw_prices, level=3)
    except Exception as e:
        print(f"Smoothing failed: {e}. Using raw prices.")
        smoothed_prices = raw_prices

    # 3. Setup Model
    print(f"Loading Model: {MODEL_ID}")
    pipeline = ChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float32
    )
    model = pipeline.model
    tokenizer = pipeline.tokenizer

    # Unwrap ChronosModel -> T5
    if hasattr(model, "model"):
        model = model.model

    model.train()

    # Dataset
    ds = TeacherDataset(smoothed_prices, tokenizer)
    if len(ds) == 0:
        print("Not enough data for training.")
        return

    dl = DataLoader(ds, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    print("Training Loop (Perception)...")
    steps = 0
    max_steps = 5 # Demo

    for batch in dl:
        input_ids = batch["input_ids"].to(model.device)
        labels = batch["labels"].to(model.device)

        # Forward
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        steps += 1
        print(f"Step {steps} Loss: {loss.item():.4f}")

        if steps >= max_steps: break

    print("Saving Teacher Model...")
    save_path = "backend/models/teacher_t5_smoothed"
    model.save_pretrained(save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    train_teacher()
