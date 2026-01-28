import pytest
import numpy as np
import pandas as pd
from backend.app.features.signal_processing import (
    apply_modwt_uks,
    apply_sliding_wavelet_ukf,
    trend_scanning_labels,
    triple_barrier_labels
)

def test_modwt_uks():
    # Generate noisy sine wave
    t = np.linspace(0, 10, 100)
    signal = np.sin(t)
    noise = np.random.normal(0, 0.1, 100)
    data = signal + noise

    smoothed = apply_modwt_uks(data, level=2)

    assert len(smoothed) == len(data)
    # Check if smoothed is less noisy (simple variance check of differences)
    # Diff of signal ~ 0.1 (smooth)
    # Diff of noise ~ large
    orig_diff_var = np.var(np.diff(data))
    smooth_diff_var = np.var(np.diff(smoothed))

    assert smooth_diff_var < orig_diff_var
    print(f"Original VarDiff: {orig_diff_var}, Smoothed VarDiff: {smooth_diff_var}")

def test_sliding_wavelet_ukf():
    data = np.random.randn(50).cumsum()
    ukf = None
    results = []

    # Needs window size >= something for wavelet
    window_size = 10
    for i in range(window_size, len(data)):
        window = data[i-window_size:i]
        val, ukf = apply_sliding_wavelet_ukf(window, ukf)
        results.append(val)

    assert len(results) == len(data) - window_size
    assert ukf is not None

def test_trend_scanning():
    # Linear trend
    data = np.linspace(0, 10, 100)
    labels = trend_scanning_labels(data, window_min=5, window_max=20)

    assert len(labels) == 100
    # Should be positive t-stats
    # Ignore end where scanning isn't possible
    valid_labels = labels[:-20]
    assert np.all(valid_labels > 0)

def test_triple_barrier():
    dates = pd.date_range('2023-01-01', periods=100)
    prices = pd.Series(np.linspace(100, 120, 100), index=dates) # Up trend (20% increase)
    vol = pd.Series(np.ones(100) * 0.01, index=dates) # 1% vol

    # 20% in 100 steps -> 0.2% per step.
    # Barrier is 1%. Takes 5 steps to hit.
    # Window is 10. Should hit.

    labels = triple_barrier_labels(prices, vol, vertical_barrier_window=10)

    assert len(labels) == 100
    # Should hit top (1) mostly
    assert (labels == 1).sum() > 0
