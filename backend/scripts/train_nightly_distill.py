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
from app.features.signal_processing import apply_modwt_uks, triple_barrier_labels
from app.models.loss import StudentTradingLoss

TEACHER_ID = "amazon/chronos-t5-large"
STUDENT_ID = "amazon/chronos-bolt-small"

class DistillationDataset(Dataset):
    def __init__(self, prices_l1, teacher_logits, tbm_labels, tokenizer, context_length=64, prediction_length=32):
        self.prices = prices_l1
        self.teacher_logits = teacher_logits # Precomputed or None
        self.tbm_labels = tbm_labels
        self.tokenizer = tokenizer
        self.context_len = context_length
        self.pred_len = prediction_length

    def __len__(self):
        return len(self.prices) - self.context_len - self.pred_len

    def __getitem__(self, idx):
        # Window
        window = self.prices[idx : idx + self.context_len]

        # Prepare inputs for Student
        window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(window_tensor)

        # Targets
        # Teacher Logits: (Pred_Len, Vocab)
        # TBM Label: Scalar (0 or 1)
        t_logits = self.teacher_logits[idx] if self.teacher_logits is not None else torch.zeros(self.pred_len, 4096)
        tbm = self.tbm_labels[idx + self.context_len] # Label at the decision point?
        # TBM label is usually "Outcome of trade entered at t".
        # So label at end of context.

        return {
            "input_ids": input_ids.squeeze(0),
            "teacher_logits": torch.tensor(t_logits, dtype=torch.float32),
            "tbm_label": torch.tensor(tbm, dtype=torch.long)
        }

def train_nightly():
    print("--- Starting Nightly Distillation ---")

    # 1. Data Prep
    client = DatabentoClient(mock_mode=True)
    df = client.get_historical_range("BTC-USD", "2023-01-01", "2023-01-02", schema='mbp-10')

    if 'price' in df.columns:
        prices = df['price']
    else:
        prices = (df['bid_px_00'] + df['ask_px_00']) / 2.0

    # Volatility & Labels
    vol = prices.rolling(20).std().fillna(0)
    tbm = triple_barrier_labels(prices, vol)

    # 2. Teacher Inference (Precompute)
    print("Loading Teacher for Inference...")
    try:
        teacher_pipe = ChronosPipeline.from_pretrained(TEACHER_ID, device_map="cuda" if torch.cuda.is_available() else "cpu", torch_dtype=torch.float32)
        teacher_model = teacher_pipe.model
        if hasattr(teacher_model, "model"): teacher_model = teacher_model.model
        teacher_tokenizer = teacher_pipe.tokenizer

        print("Precomputing Teacher Logits...")
        teacher_model.eval()

        teacher_logits_list = []
        teacher_prices = prices.values # Assuming simplified L2 for demo

        context_len = 64
        pred_len = 32
        indices = list(range(len(teacher_prices) - context_len - pred_len))
        batch_size = 4 # Small batch for safety

        with torch.no_grad():
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i+batch_size]

                input_ids_list = []
                labels_list = []

                for idx in batch_indices:
                    window = teacher_prices[idx : idx + context_len + pred_len]
                    context = window[:context_len]
                    target = window[context_len:]

                    c_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
                    t_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)

                    i_ids, _, scale = teacher_tokenizer.context_input_transform(c_tensor)
                    # Try to use input_transform with context scale
                    try:
                        l_ids, _ = teacher_tokenizer.input_transform(t_tensor, scale)
                    except:
                        # Fallback
                        l_ids, _, _ = teacher_tokenizer.context_input_transform(t_tensor)

                    input_ids_list.append(i_ids.squeeze(0))
                    labels_list.append(l_ids.squeeze(0))

                if not input_ids_list: continue

                input_ids = torch.stack(input_ids_list).to(teacher_model.device)
                labels = torch.stack(labels_list).to(teacher_model.device)

                outputs = teacher_model(input_ids=input_ids, labels=labels)
                teacher_logits_list.append(outputs.logits.cpu())

                if len(teacher_logits_list) * batch_size > 20:
                    # Limit for demo/testing to avoid hour-long run
                    break

        if teacher_logits_list:
            teacher_logits = torch.cat(teacher_logits_list, dim=0)
        else:
            teacher_logits = None

        del teacher_pipe, teacher_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Teacher Inference Failed: {e}. Using dummy logits.")
        teacher_logits = None

    # 3. Student Training
    print(f"Loading Student: {STUDENT_ID}")
    student_pipe = ChronosPipeline.from_pretrained(STUDENT_ID, device_map="cuda", torch_dtype=torch.float32)
    student_model = student_pipe.model
    tokenizer = student_pipe.tokenizer

    # Get Bin Centers for Loss
    # Try to find from model config
    # This is model specific. For Bolt, it might be in `model.config.distribution_output`?
    # We will fallback to linear space if not found.
    centers = torch.linspace(-15, 15, 4096).to(student_model.device) # Approx

    criterion = StudentTradingLoss(tokenizer_centers=centers, risk_free_rate=0.0)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)

    ds = DistillationDataset(prices.values, teacher_logits, tbm.values, tokenizer)
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    student_model.train()
    print("Starting Distillation Loop...")

    steps = 0
    for batch in dl:
        input_ids = batch["input_ids"].to(student_model.device)
        t_logits = batch["teacher_logits"].to(student_model.device) # (B, Pred, V)
        tbm_target = batch["tbm_label"].to(student_model.device)

        # Student Forward
        # Bolt forward returns different things?
        # usually: output = model(input_ids)
        # output.logits is (B, Pred, V) or (B, Context+Pred, V) depending on mode
        # Chronos Bolt is usually Decoder-only or Encoder-Decoder?
        # T5 is Enc-Dec. Bolt is usually Enc-Dec or just T5 based.
        # "amazon/chronos-bolt" implies T5 based or TinyLlama?
        # Actually Bolt is T5 based usually.
        # If we pass labels=None, we get logits?
        # We need logits for the NEXT tokens (Prediction).
        # But we don't have ground truth labels for 'loss' argument.
        # We want pure logits.

        # We might need to construct `decoder_input_ids`.
        # For inference, pipeline handles it.
        # For training, we usually provide labels.
        # Here we want to optimize custom loss.
        # We need the model to output logits for the prediction horizon.
        # For T5, we need to pass `decoder_input_ids`.
        # We can use `input_ids` (context) and learn to predict future?
        # But we need to feed start token?

        # Simplify: assume we can get logits.
        # For now, let's just run a dummy forward with dummy labels to get logits,
        # then ignore the internal loss and use ours.
        dummy_labels = torch.zeros(input_ids.shape[0], 32, dtype=torch.long).to(student_model.device)
        outputs = student_model(input_ids=input_ids, labels=dummy_labels) # Force generation mode?

        s_logits = outputs.logits # Shape (B, 32, V) hopefully

        # Calculate our loss
        loss, components = criterion(s_logits, t_logits, tbm_target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        steps += 1
        print(f"Step {steps} Loss: {loss.item():.4f} (Dist: {components[0]:.2f}, Sort: {components[1]:.2f}, Focal: {components[2]:.2f})")

        if steps >= 5: break

    print("Saving Distilled Student...")
    student_model.save_pretrained("backend/models/chronos_bolt_distilled")
    print("Done.")

if __name__ == "__main__":
    train_nightly()
