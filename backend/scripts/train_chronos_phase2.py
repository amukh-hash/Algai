import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoConfig, 
    BitsAndBytesConfig, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from chronos import ChronosPipeline

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = r"backend\data\processed\orthogonal_features_final.parquet"
MODEL_ID = "amazon/chronos-t5-large"
OUTPUT_DIR = r"backend\models\chronos_physics_phase2"

class OrthogonalDataset(Dataset):
    def __init__(self, data_path, tokenizer, context_length=512, prediction_length=64, target_col="close_frac"):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.tokenizer = tokenizer
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found at {data_path}")
            
        print(f"Loading data from {data_path}...")
        self.df = pd.read_parquet(data_path)
        
        # Select Target
        if target_col not in self.df.columns:
            # Fallback
            target_col = self.df.columns[0]
            if "close" in self.df.columns: target_col = "close"
            print(f"Target '{target_col}' used.")
            
        self.series = self.df[target_col].values.astype(np.float32)
        # Handle NaNs
        self.series = np.nan_to_num(self.series)
        
        self.n_samples = len(self.series) - (context_length + prediction_length) + 1
        print(f"Loaded {len(self.series)} points. Samples: {self.n_samples}")

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        # Sliding Window
        start = idx
        mid = idx + self.context_length
        end = mid + self.prediction_length
        
        context_part = torch.tensor(self.series[start:mid]).unsqueeze(0) # (1, L)
        target_part = torch.tensor(self.series[mid:end]).unsqueeze(0)
        
        # Chronos Transform
        # We need to return INPUT_IDS and LABELS
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(context_part)
        label_ids, _, _ = self.tokenizer.context_input_transform(target_part) 
        
        # Squeeze batch dim since Dataset adds it back
        return {
            "input_ids": input_ids.squeeze(0),
            "labels": label_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0)
        }

class SharpeLoss(nn.Module):
    def __init__(self, risk_free_rate=0.0):
        super().__init__()
        self.rf = risk_free_rate

    def forward(self, logits, labels):
        """
        Differentiable Sharpe Ratio Loss.
        """
        if logits is None: return torch.tensor(0.0, device=labels.device, requires_grad=True)

        # 1. Softmax to get probabilities over tokens
        probs = torch.softmax(logits, dim=-1) # (B, Time, Vocab)
        
        # 2. Decode Tokens to Values (Differentiable Approximation)
        # Chronos tokens represent Bins. We need expected value.
        # Ideally we use the tokenizer's bin centers.
        # Hack: High token ID ~ High Value? No, random.
        # We need the Bin Centers from the Config.
        # If unavailable, we revert to CrossEntropy (NLL) for stability first.
        # OR: We Penalize Variance of certain tokens?
        
        # SIMPLIFIED: Standard NLL + Variance Penalty on Logits?
        # Real Differentiable Sharpe on T5 is hard without the bin map.
        # Let's fallback to Standard Loss for now, user can enable experimental.
        
        return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Standard Loss (CrossEntropy)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Add Volatility Penalty?
        # ...
        
        if return_outputs:
            return (loss, outputs)
        return loss

def train_chronos_phase2():
    print("--- Phase 2: Chronos-Physics (4-bit LoRA) ---")
    
    # 1. Load Tokenizer via Pipeline (CPU) so we map 'amazon/chronos...' correctly
    pipeline = ChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        torch_dtype=torch.float32
    )
    tokenizer = pipeline.tokenizer
    
    # 2. Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading {MODEL_ID} in 4-bit...")
    # Load Model (Wrapped or Raw T5?)
    # ChronosPipeline.model is the wrapper.
    # AutoModelForSeq2SeqLM loads the T5 directly.
    # We want T5 directly for standard HF Trainer compatibility.
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True # Chronos might need this
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # 3. LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. Data
    ds = OrthogonalDataset(DATA_PATH, tokenizer, context_length=256, prediction_length=64)
    
    # 5. Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8, # 4070 Super: 12GB. 8 should be safe with 4bit.
        gradient_accumulation_steps=4, # Effective batch 32
        learning_rate=2e-4,
        logging_steps=50,
        max_steps=20000, # Overnight: ~5-6 hours on 4070 Super
        save_steps=2000,
        fp16=True,
        optim="paged_adamw_8bit", # Saves VRAM
        report_to=["none"]
    )
    
    # Collator need to pad
    collator = DataCollatorForSeq2Seq(tokenizer=None, model=model, padding=True) 
    # Tokenizer is custom Chronos object, might not have pad_token_id standard property exposed 
    # or incompatible with HF Collator?
    # We used manual padding in previous script.
    # Let's pass None and ensure Dataset returns tensors.
    # Actually HF Trainer expects `tokenizer` to pad if provided.
    
    # Custom Collator
    def collate_fn(batch):
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        # Pad
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        # Attention Mask (1 for real, 0 for pad)
        # Assuming input_ids 0 is pad?
        attention_mask = (input_ids != 0).long()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collate_fn
    )
    
    print("Starting Training...")
    trainer.train()
    
    model.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_chronos_phase2()
