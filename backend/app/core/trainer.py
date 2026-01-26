import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import time
from tqdm import tqdm

class Trainer:
    """
    Standard Training Loop for PyTorch models.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = "cpu"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)

        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0

        # Helper for progress bar
        pbar = tqdm(dataloader, desc="Training", unit="batch")

        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            # Forward with Mixed Precision
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                output = self.model(x)
                loss = self.criterion(output, y)

            # Backward with Scaler
            self.scaler.scale(loss).backward()

            # Unscale for Clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)

            # Step with Scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            # Update pbar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 10,
            patience: int = 3) -> Dict[str, Any]:

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0

        start_time = time.time()

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save model logic here
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        total_time = time.time() - start_time
        return {
            'history': history,
            'best_val_loss': best_val_loss,
            'total_time': total_time
        }
