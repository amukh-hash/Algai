import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) Linear Layer.
    Wraps a frozen pre-trained linear layer and adds a trainable low-rank branch.
    output = Wx + BAx
    """
    def __init__(self, 
                 original_layer: nn.Linear, 
                 rank: int = 4, 
                 alpha: int = 1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Dimensions
        in_dim = original_layer.in_features
        out_dim = original_layer.out_features
        
        # LoRA Matrices
        # A: (Rank, In) initialized Gaussian
        # B: (Out, Rank) initialized Zero
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scaling = alpha / rank
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze Original
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Original branch
        out_orig = self.original_layer(x)
        
        # LoRA branch: x @ A.T @ B.T * scaling
        # (Batch, In) @ (In, Rank) -> (Batch, Rank)
        # (Batch, Rank) @ (Rank, Out) -> (Batch, Out)
        out_lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return out_orig + out_lora
