import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
import joblib
from chronos import ChronosPipeline

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

DATA_CACHE_PATH = os.path.join("backend", "data_cache_alpaca", "universal_dataset_cache.pkl")
# Using T5-Large for Physics. T5-Small for testing.
MODEL_ID = "amazon/chronos-t5-large" 

class ChronosDataset(Dataset):
    def __init__(self, cache_path, tokenizer, context_length=512, prediction_length=64):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.tokenizer = tokenizer
        
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache not found at {cache_path}")
            
        print(f"Loading data from {cache_path}...")
        cached = joblib.load(cache_path)
        self.samples = cached.get('samples', [])
        # dfs is list of numpy arrays [Open, High, Low, Close, ...]
        self.dfs = cached.get('dfs', [])
        
        print(f"Loaded {len(self.samples)} samples. Adapting for Chronos...")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_idx, end_idx, _, _ = self.samples[idx]
        data = self.dfs[f_idx]
        
        # Window: [t - context : t + pred]
        # data is [Time, Features]. Col 3 is Close.
        
        total_len = self.context_length + self.prediction_length
        start_global = end_idx - self.context_length + 1
        end_global = end_idx + 1 + self.prediction_length
        
        if start_global < 0 or end_global > len(data):
            # Pad or Skip. Return dummies (filtered by collate or loader size)
            # Simple dummy for now
            return torch.zeros(1), torch.zeros(1) # Flags
        
        # Extract Series
        raw_series = data[start_global:end_global, 3] # Close Price
        series_torch = torch.tensor(raw_series, dtype=torch.float32).unsqueeze(0) # (1, L)
        
        # Chronos Transform
        # 1. Scale & Bin -> Token IDs
        # The pipeline methods expect specific shapes.
        # context_input_transform expects (Batch, Time, Dim) or (Batch, Time)
        # We need to manually split context and target?
        
        # Actually, for training T5, we feed (Input_Ids, Labels).
        
        # Chronos Tokenizer has `context_input_transform`.
        # It takes real values, scales them, and quantizes them.
        # It returns `token_ids`, `attention_mask`, `scale`.
        
        context_part = series_torch[:, :self.context_length]
        target_part = series_torch[:, self.context_length:]
        
        # Convert to tokens
        # Note: input_ids from context + target?
        # T5 training: input_ids = context tokens. labels = target tokens.
        
        # We use the tokenizer to process the WHOLE series? 
        # Or context separate?
        # Chronos inference: Context -> Generate Future.
        
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(context_part)
        label_ids, _, _ = self.tokenizer.context_input_transform(target_part) 
        # Note: label_ids logic slightly different (no scaling, use context scale?)
        # Chronos assumes stationary?
        # Actually `context_input_transform` computes scale. 
        # We should use the SAME scale for target. 
        # The library might expose `output_transform`? No that's inverse.
        
        # Manual Scaling for Target using Context Scale:
        # scale shape (1, 1).
        target_scaled = target_part / scale
        # Quantize manually?
        # tokenizer.input_transform(target_scaled)? 
        # It seems `_input_transform` does scale + quantize.
        # Check `tokenizer.quantize(target_scaled)`?
        # Let's hope `context_input_transform` handles generic input.
        
        # Assuming we just run fine-tuning on context -> target.
        # This dataset code runs on CPU worker.
        
        # RETURN:
        # input_ids: (L_ctx)
        # labels: (L_pred)
        
        return input_ids.squeeze(0), label_ids.squeeze(0)

def collate_fn(batch):
    # Filter dummies
    batch = [b for b in batch if b[0].ndim > 0 and b[0].shape[0] > 1]
    if not batch: return None
    
    input_ids = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    
    # Pad
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0) # 0 is usually pad in HF
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100) # -100 ignore index
    
    return {"input_ids": input_ids, "labels": labels}

def train_chronos():
    print("--- Initializing Chronos-Physics (LoRA) ---")
    
    # 1. Load Pipeline (CPU first to save VRAM, then move model)
    # Using 'cpu' mapping initially, then custom move.
    print(f"Loading Chronos Pipeline: {MODEL_ID}")
    pipeline = ChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map="cpu", # Load on CPU, we will move model to GPU manually wrapped in LoRA
        torch_dtype=torch.float32
    )
    
    model = pipeline.model
    tokenizer = pipeline.tokenizer
    
    # Unwrap ChronosModel if needed to get to T5
    if hasattr(model, "model"):
        print("Unwrapping ChronosModel -> T5")
        model = model.model
    
    # Enable Gradient Checkpointing for VRAM savings
    model.gradient_checkpointing_enable()
    
    # 2. Inject LoRA
    print("Injecting LoRA Adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"] # T5 attention
    )
    model = get_peft_model(model, peft_config)
    
    # 3. Move to GPU (Quantized ideally, but float16 is okay for 12GB if Batch is small)
    # To use 8bit, we needed 'load_in_8bit=True' in from_pretrained.
    # ChronosPipeline passes kwargs to AutoModel? Yes.
    # But we already loaded. Re-load model part?
    # Simpler: Just float16 on GPU. 700M params = 1.4GB. 12GB is plenty.
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.print_trainable_parameters()
    
    # 4. Data
    ds = ChronosDataset(DATA_CACHE_PATH, tokenizer, context_length=128, prediction_length=64)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    # 5. Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    print("\nStarting Training Loop...")
    for epoch in range(5):
        epoch_loss = 0
        steps = 0
        pbar_size = 100 # limit steps for demo
        
        for batch in loader:
            if batch is None: continue
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            steps += 1
            if steps % 10 == 0:
                print(f"Epoch {epoch} Step {steps} Loss: {loss.item():.4f}")
                
            if steps >= pbar_size: break
            
        print(f"Epoch {epoch} Avg Loss: {epoch_loss/steps:.4f}")
        
    save_path = "backend/models/chronos_physics_lora"
    model.save_pretrained(save_path)
    print(f"LoRA Adapters saved to {save_path}")

if __name__ == "__main__":
    train_chronos()
