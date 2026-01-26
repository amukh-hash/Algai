import torch
import torch.nn as nn
from app.preprocessing.revin import RevIN
from app.models.layers import PatchEmbedding
from app.models.lora import LoRALinear

class PatchTSTBackbone(nn.Module):
    """
    Core Transformer Backbone for PatchTST.
    Processes a sequence of patches.
    """
    def __init__(self, 
                 num_patches: int, 
                 d_model: int, 
                 n_heads: int, 
                 n_layers: int, 
                 d_ff: int, 
                 dropout: float = 0.1,
                 head_dropout: float = 0.0):
        super().__init__()
        
        # Positional Encoding (Learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, num_patches, d_model))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_ff, 
            dropout=dropout, 
            activation="gelu",
            batch_first=True,
            norm_first=True 
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, x):
        """
        x: (Batch, Channels, Num_Patches, d_model)
        """
        B, C, N, D = x.shape
        x = x.reshape(B*C, N, D)
        x = x + self.pos_embedding.reshape(1, N, D)
        x = self.dropout(x)
        x = self.encoder(x)
        return x

class HybridPatchTST(nn.Module):
    """
    Refactored PatchTST with Dual Heads: Direction (Binary) and Volatility (Regression).
    Supports LoRA for Fine-Tuning.
    """
    def __init__(self,
                 num_input_features: int, 
                 lookback_window: int,
                 patch_len: int = 16,
                 stride: int = 8,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 use_lora: bool = False,
                 lora_rank: int = 4,
                 n_classes: int = 3):
        super().__init__()
        
        self.num_input_features = num_input_features
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        
        # 1. RevIN
        self.revin = RevIN(num_features=num_input_features, affine=True)
        
        # 2. Patching
        self.patch_embedding = PatchEmbedding(patch_len, stride, d_model)
        self.num_patches = int((lookback_window - patch_len) / stride + 1)
        
        # 3. Backbone
        self.backbone = PatchTSTBackbone(
            num_patches=self.num_patches,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # 4. Pooling / Flattening
        # Flatten patches: (N * d_model)
        self.flatten_dim = self.num_patches * d_model
        
        # 5. Heads
        self.global_flatten_dim = num_input_features * self.flatten_dim
        
        # Direction Head: Classification (Logits for CrossEntropy/FocalLoss)
        self.direction_head = nn.Sequential(
            nn.Linear(self.global_flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
            # No Sigmoid/Softmax! Return logits.
        )
        
        # Volatility Head: Expected Magnitude (Sigma or %)
        self.volatility_head = nn.Sequential(
            nn.Linear(self.global_flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Softplus() # Ensure positive volatility
        )
        
        if self.use_lora:
            self._inject_lora()

    def _inject_lora(self):
        """
        Replaces Linear layers in the backbone (FFN) with LoRALinear.
        Usually target Q/K/V in Attention or FFN.
        For PatchTST Encoder, simple way: target the final heads or Encoder layers?
        Encoder is nn.TransformerEncoder, hard to patch internal layers without copy-paste logic.
        
        Strategy: Apply LoRA to the Projection Heads.
        This allows adapting the 'Interpretation' of the backbone's features.
        """
        # Patch Direction Head [0] and [3] (Linear layers)
        if isinstance(self.direction_head[0], nn.Linear):
            self.direction_head[0] = LoRALinear(self.direction_head[0], rank=self.lora_rank)
        
        if isinstance(self.volatility_head[0], nn.Linear):
            self.volatility_head[0] = LoRALinear(self.volatility_head[0], rank=self.lora_rank)

    def load_adapter(self, adapter_path: str):
        """
        Loads LoRA weights.
        """
        if not self.use_lora:
            raise ValueError("Model initialized without LoRA support.")
        
        state_dict = torch.load(adapter_path)
        self.load_state_dict(state_dict, strict=False) # Load matches

    def forward(self, x):
        """
        x: (Batch, Lookback, Features)
        Returns:
            direction: (Batch, 1) probability [0, 1]
            volatility: (Batch, 1) magnitude > 0
        """
        B, L, F = x.shape
        
        # 1. RevIN
        x_norm, _, _ = self.revin.normalize(x)
        
        # 2. Patching
        x_emb = self.patch_embedding(x_norm) # (B, F, N, D)
        
        # 3. Backbone
        x_enc = self.backbone(x_emb) # (B*F, N, D)
        
        # 4. Global Flatten
        # Reshape back to (B, F, N, D)
        x_enc = x_enc.reshape(B, F, self.num_patches, -1)
        # Flatten everything
        x_flat = x_enc.reshape(B, -1)
        
        # 5. Heads
        direction = self.direction_head(x_flat)
        volatility = self.volatility_head(x_flat)
        
        return direction, volatility
