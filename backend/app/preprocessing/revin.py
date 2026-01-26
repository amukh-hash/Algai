import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN).
    Reference: https://openreview.net/pdf?id=cGDAkQo1C0p
    
    Attributes:
        num_features (int): Number of input features (channels).
        eps (float): Epsilon for stability.
        affine (bool): Whether to learn affine parameters (gamma, beta).
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self._init_params()

    def _init_params(self):
        # Gamma and Beta per feature
        # Shape: (1, 1, num_features) to broadcast over Batch and Time
        self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_features))

    def forward(self, x, mode: str = 'norm'):
        """
        Args:
            x: Input tensor of shape (Batch, Time, Features)
            mode: 'norm' or 'denorm'.
        """
        if mode == 'norm':
            return self.normalize(x)
        elif mode == 'denorm':
            # This is tricky: denorm needs the mean/std from the input.
            # If we just call forward(y, mode='denorm'), we don't have stats.
            # Usually RevIN is used as a wrapper or we store stats temporarily?
            # Storing stats in self is dangerous for multi-gpu/parallel.
            # Ideally, normalize returns (x_norm, stats).
            raise NotImplementedError("Use normalize() and denormalize() explicitly.")
    
    def normalize(self, x):
        """
        Normalize the input time series.
        Returns:
            x_norm: Normalized tensor.
            mean: Mean statistics.
            stdev: Std statistics.
        """
        # x: (Batch, Time, Features)
        # Compute mean/std over Time dimension (dim=1)
        
        mean = torch.mean(x, dim=1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        
        x_norm = (x - mean) / stdev
        
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias
            
        return x_norm, mean, stdev

    def denormalize(self, x_norm, mean, stdev):
        """
        Denormalize the input.
        Args:
            x_norm: Normalized tensor (usually model output).
            mean: Mean from input.
            stdev: Std from input.
        """
        if self.affine:
            x_norm = (x_norm - self.affine_bias) / (self.affine_weight + self.eps*1e-6)
            
        x_denorm = x_norm * stdev + mean
        return x_denorm

class RevINModelWrapper(nn.Module):
    """
    Wraps a backbone model with RevIN.
    """
    def __init__(self, backbone: nn.Module, num_features: int, affine: bool = True):
        super().__init__()
        self.backbone = backbone
        self.revin = RevIN(num_features, affine=affine)
        
    def forward(self, x):
        """
        x: (Batch, Time, Features)
        """
        x_norm, mean, stdev = self.revin.normalize(x)
        
        # Backbone forward
        y_hat_norm = self.backbone(x_norm)
        
        # Denormalize
        # Note: y_hat_norm might have different sequence length or features?
        # Usually for forecasting, Features is same (target variables).
        # If forecasting only a subset, we need careful index matching.
        # Assuming forecasting all features or same features as input.
        
        y_hat = self.revin.denormalize(y_hat_norm, mean, stdev)
        return y_hat
