import os
import sys
import torch
import numpy as np
import pandas as pd
from collections import deque
import riskfolio as rp
from chronos import ChronosPipeline
import xgboost as xgb

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.api.databento_client import DatabentoClient
from app.features.signal_processing import apply_sliding_wavelet_ukf

STUDENT_ID = "amazon/chronos-bolt-small"
# STUDENT_ID = "amazon/chronos-t5-small" # Fallback for environment compatibility

def perform_daytime_update(model, tokenizer, window_data, optimizer):
    """
    SimTS Adaptation: Updates encoder embeddings.
    """
    try:
        # Augmentation
        v1 = window_data * (1 + np.random.normal(0, 0.005, len(window_data)))
        v2 = window_data * (1 + np.random.normal(0, 0.005, len(window_data)))

        t1 = torch.tensor(v1, dtype=torch.float32).unsqueeze(0)
        t2 = torch.tensor(v2, dtype=torch.float32).unsqueeze(0)

        # Tokenize
        id1, _, _ = tokenizer.context_input_transform(t1)
        id2, _, _ = tokenizer.context_input_transform(t2)

        id1 = id1.to(model.device)
        id2 = id2.to(model.device)

        # Forward Encoder
        # T5 has `encoder` or `model.encoder`
        if hasattr(model, "encoder"):
            enc = model.encoder
        elif hasattr(model, "model") and hasattr(model.model, "encoder"):
            enc = model.model.encoder
        else:
            return # Can't find encoder

        out1 = enc(input_ids=id1).last_hidden_state # (B, L, D)
        out2 = enc(input_ids=id2).last_hidden_state

        # Maximize similarity (MSE of embeddings or Cosine)
        # SimTS usually uses Cosine Similarity on pooled representation
        p1 = out1.mean(dim=1)
        p2 = out2.mean(dim=1)

        loss = 1.0 - torch.nn.functional.cosine_similarity(p1, p2).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    except Exception as e:
        # print(f"Adaptation failed: {e}")
        pass

def get_meta_confidence(judge_model, features):
    if judge_model is None: return 0.6
    return 0.6

def allocate_portfolio(confidence):
    # Kelly
    p = confidence
    if p < 0.5: return 0.0

    b = 2.0
    q = 1.0 - p
    kelly = (p * b - q) / b
    return max(0.0, kelly * 0.5)

def run_live():
    print("--- Starting Live Execution (Bolt) ---")

    # 1. Init
    try:
        client = DatabentoClient(mock_mode=True)
        pipeline = ChronosPipeline.from_pretrained(STUDENT_ID, device_map="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=torch.float32)
        model = pipeline.model
        tokenizer = pipeline.tokenizer

        # Optimizer for adaptation (freeze all but embeddings? or encoder?)
        # For simplicity, optimize encoder last layer or similar.
        # Here we verify the loop, so just generic optimizer.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

        judge_model = None

        stream = client.start_live_stream("BTC-USD")
        window = deque(maxlen=64)
        ukf = None

        print("Listening for ticks...")

        count = 0
        for tick in stream:
            price = tick['price']
            window.append(price)

            if len(window) < 32: continue

            win_arr = np.array(window)

            # 2. Causal Filter
            filtered_price, ukf = apply_sliding_wavelet_ukf(win_arr, ukf)

            # 3. Inference
            tensor = torch.tensor(win_arr, dtype=torch.float32).unsqueeze(0)
            # Forecast 1 step
            forecast = pipeline.predict(tensor, prediction_length=1)
            pred_price = np.median(forecast.numpy()) # (1, samples, 1) -> scalar

            # 4. Meta & Allocation
            conf = get_meta_confidence(judge_model, [])
            size = allocate_portfolio(conf)

            action = "WAIT"
            if size > 0:
                if pred_price > filtered_price: action = "BUY"
                else: action = "SELL"

            print(f"Tick: {price:.2f} | Filt: {filtered_price:.2f} | Pred: {pred_price:.2f} | Act: {action} ({size:.2f})")

            # 5. Adaptation
            count += 1
            if count % 10 == 0:
                perform_daytime_update(model, tokenizer, win_arr, optimizer)

            if count > 20: break # Demo limit

    except Exception as e:
        print(f"Live Loop Error: {e}")

if __name__ == "__main__":
    run_live()
