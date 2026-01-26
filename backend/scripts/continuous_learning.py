import os
import sys
import torch
import joblib
from torch.utils.data import DataLoader
from chronos import ChronosPipeline
from peft import get_peft_model, LoraConfig, TaskType

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.train_chronos_lora import ChronosDataset, collate_fn

DATA_CACHE_PATH = os.path.join("backend", "data_cache_alpaca", "universal_dataset_cache.pkl")
MODEL_PATH = "backend/models/chronos_physics_lora"
EWC_PATH = "backend/models/ewc_fisher.pt"

class EWCTrainer:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = {}
        self.optpar = {}

    def compute_fisher(self):
        """
        Calculate Fisher Information Matrix (Diagonal approximation).
        F = E[grad(L)^2]
        """
        print("Computing Fisher Information Matrix...")
        self.model.eval()
        
        # Initialize Fisher dict
        for n, p in self.params.items():
            self.fisher[n] = torch.zeros_like(p.data)
            self.optpar[n] = p.data.clone() # Store old weights

        count = 0
        for batch in self.dataloader:
            if batch is None: continue
            
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            # Accumulate grad^2
            for n, p in self.params.items():
                if p.grad is not None:
                    self.fisher[n] += p.grad.data ** 2
            
            count += 1
            if count >= 50: break # Estimate on subset for speed
            
        # Average
        for n in self.fisher:
            self.fisher[n] /= count
            
        print("Fisher Matrix Computed.")

    def ewc_loss(self, current_loss, lambda_ewc=1000):
        """
        L_final = L_task + (lambda/2) * Sum(F * (theta - theta_star)^2)
        """
        ewc_loss = 0
        for n, p in self.params.items():
            fisher = self.fisher[n]
            opt_par = self.optpar[n]
            ewc_loss += (fisher * (p - opt_par) ** 2).sum()
            
        return current_loss + (lambda_ewc / 2) * ewc_loss

    def save_fisher(self, path):
        torch.save({
            'fisher': self.fisher,
            'optpar': self.optpar
        }, path)
        print(f"EWC State saved to {path}")

def run_continuous_learning_setup():
    print("--- Setting up Continuous Learning (EWC) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Trained Model
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-large",
        device_map="cpu", # Move to GPU explicitly
        torch_dtype=torch.float32 
    )
    
    target_model = pipeline.model
    if hasattr(target_model, "model"):
        target_model = target_model.model
        
    print(f"Loading Adapter from {MODEL_PATH}...")
    target_model.load_adapter(MODEL_PATH)
    target_model.to(device)
    target_model.train() # Set to train mode for gradients
    
    # Ensure gradients are enabled for LoRA
    for n, p in target_model.named_parameters():
        if "lora" in n:
            p.requires_grad = True
    
    # 2. Load Old Data (for Fisher calc)
    tokenizer = pipeline.tokenizer
    ds = ChronosDataset(DATA_CACHE_PATH, tokenizer, context_length=128, prediction_length=64)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # 3. Compute FIM
    ewc_trainer = EWCTrainer(target_model, loader, device)
    ewc_trainer.compute_fisher()
    ewc_trainer.save_fisher(EWC_PATH)

if __name__ == "__main__":
    run_continuous_learning_setup()
