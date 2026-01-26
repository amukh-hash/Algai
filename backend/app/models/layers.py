import torch
import torch.nn as nn
import math

class Time2Vec(nn.Module):
    """
    Time2Vec: Learning a Vector Representation of Time.
    https://arxiv.org/abs/1907.05595

    Formula:
    t2v(tau)[i] = w_i * tau + phi_i, if i = 0
    t2v(tau)[i] = F(w_i * tau + phi_i), if 1 <= i <= k, F = Sin
    """
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

        # Weights and biases
        # Shape: (output_dim)
        self.w = nn.Parameter(torch.randn(output_dim))
        self.phi = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Time, 1) or (Batch, Time).
               This represents the time steps (e.g., indices or normalized timestamps).
        Returns:
            Tensor of shape (Batch, Time, output_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1) # (Batch, Time, 1)

        # Broadcast multiply: (Batch, Time, 1) * (output_dim) -> (Batch, Time, output_dim)
        # We need to reshape w and phi for broadcasting
        # w: (1, 1, output_dim)

        time_linear = self.w[0] * x + self.phi[0]
        time_periodic = torch.sin(self.w[1:] * x + self.phi[1:])

        # Concatenate
        # time_linear is (Batch, Time, 1)
        # time_periodic is (Batch, Time, output_dim-1)

        return torch.cat([time_linear, time_periodic], dim=-1)

class PatchEmbedding(nn.Module):
    """
    Patch Embedding layer for PatchTST.
    Segments the time series into overlapping patches.
    """
    def __init__(self, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # Linear projection to map patch_len to d_model
        # Note: If we treat each channel independently, this is applied per channel.
        # PatchTST typically shares the same embedding weights across all channels.
        self.projection = nn.Linear(patch_len, d_model)

    def forward(self, x):
        """
        Args:
            x: Input tensor (Batch, Time, Channels) (after RevIN).
               Note: PatchTST usually reshapes to (Batch * Channels, Time, 1) for channel independence.

        Returns:
            Tensor of shape (Batch, Channels, Num_Patches, d_model) or similar.
        """
        # 1. Unfold/Patching
        # x: (Batch, Time, Channels)
        # We want to patch along the Time dimension.
        # Permute to (Batch, Channels, Time) to use unfold on dim 2
        x = x.permute(0, 2, 1) # (B, C, T)

        # unfold(dimension, size, step)
        # Output: (B, C, Num_Patches, Patch_Len)
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)

        # 2. Projection
        # We want to project the last dimension (Patch_Len) to d_model
        # patches: (B, C, N, P)
        out = self.projection(patches) # (B, C, N, d_model)

        return out
