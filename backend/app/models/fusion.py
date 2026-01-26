import torch
import torch.nn as nn

class TextEmbeddingMock(nn.Module):
    """
    Mocks a FinBERT model.
    In production, this would be a real HuggingFace Transformer.
    Here we project random integers (tokens) to an embedding size.
    """
    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x):
        """
        x: (Batch, Sequence_Length) of token IDs.
        Returns: (Batch, Embedding_Dim) - Pooled output (e.g. CLS token or Mean)
        """
        emb = self.embedding(x) # (B, S, D)
        # Mean pooling
        return emb.mean(dim=1)

class GatedMultimodalUnit(nn.Module):
    """
    Fuses Price Embedding and Text Embedding using a Gated Mechanism.
    z = sigmoid(W_z * [E_text, E_price] + b_z)
    H = z * tanh(W_t * E_text) + (1-z) * tanh(W_p * E_price)
    """
    def __init__(self, price_dim: int, text_dim: int, hidden_dim: int):
        super().__init__()
        
        # Projections to hidden_dim
        self.W_t = nn.Linear(text_dim, hidden_dim)
        self.W_p = nn.Linear(price_dim, hidden_dim)
        
        # Gate
        self.W_z = nn.Linear(text_dim + price_dim, 1)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, price_emb, text_emb):
        """
        price_emb: (Batch, price_dim)
        text_emb: (Batch, text_dim)
        """
        # Project inputs
        h_text = self.tanh(self.W_t(text_emb))
        h_price = self.tanh(self.W_p(price_emb))
        
        # Compute Gate
        # Concatenate raw embeddings
        concat = torch.cat([price_emb, text_emb], dim=-1)
        z = self.sigmoid(self.W_z(concat)) # (Batch, 1)
        
        # Fuse
        # z determines how much weight to give to TEXT.
        # If z=1, fully text. If z=0, fully price.
        # Formula from requirements: z * text + (1-z) * price
        
        h_fused = z * h_text + (1 - z) * h_price
        
        return h_fused
