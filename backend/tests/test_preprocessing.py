import pytest
import pandas as pd
import numpy as np
import torch
from app.preprocessing.fractional import FractionalDifferencer, frac_diff_ffd
from app.preprocessing.revin import RevIN

def test_frac_diff_ffd_logic():
    # Create a simple series: 1, 2, 3, 4, 5...
    data = np.arange(100, dtype=float)
    series = pd.Series(data)
    
    # Apply with d=1 (should be approx standard differencing)
    # Note: FFD with threshold might cut off weights, so it might not be exactly x[t] - x[t-1]
    # But let's check basic execution
    
    diff_series = frac_diff_ffd(series, d=0.5, thres=1e-3)
    
    assert isinstance(diff_series, pd.Series)
    # The first few elements should be NaN due to window width
    assert diff_series.isna().any()
    
    # Check valid elements exist
    assert diff_series.notna().sum() > 0

def test_fractional_differencer_class():
    # Needs sufficient length for the window
    df = pd.DataFrame({
        'close': np.cumprod(1 + np.random.normal(0, 0.01, 1000))
    })
    
    differ = FractionalDifferencer(d=0.4)
    out_df = differ.transform(df)
    
    assert 'close' in out_df.columns
    assert len(out_df) == len(df)
    assert out_df['close'].isna().sum() > 0

def test_revin_normalization():
    # Shape: (Batch, Time, Features)
    B, T, F = 32, 100, 4
    x = torch.randn(B, T, F) * 5 + 10 # Mean 10, Std 5
    
    revin = RevIN(num_features=F, affine=False)
    
    x_norm, mean, std = revin.normalize(x)
    
    # Check output stats
    # Mean over Time dim should be approx 0
    # Var over Time dim should be approx 1
    
    out_mean = x_norm.mean(dim=1)
    out_std = x_norm.std(dim=1, unbiased=False)
    
    assert torch.allclose(out_mean, torch.zeros_like(out_mean), atol=1e-5)
    assert torch.allclose(out_std, torch.ones_like(out_std), atol=1e-5)
    
    # Test Denormalization
    x_rec = revin.denormalize(x_norm, mean, std)
    assert torch.allclose(x, x_rec, atol=1e-5)

def test_revin_affine():
    B, T, F = 2, 10, 2
    x = torch.randn(B, T, F)
    revin = RevIN(num_features=F, affine=True)
    
    # Run forward pass (normalization)
    x_norm, mean, std = revin.normalize(x)
    
    # Verify parameter shapes
    assert revin.affine_weight.shape == (1, 1, F)
    assert revin.affine_bias.shape == (1, 1, F)
    
    # Denormalize
    x_rec = revin.denormalize(x_norm, mean, std)
    assert torch.allclose(x, x_rec, atol=1e-5)
