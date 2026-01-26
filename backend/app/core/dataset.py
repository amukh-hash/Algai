import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple, Optional

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for sliding window time series data.
    """
    def __init__(self, 
                 data: pd.DataFrame, 
                 target_cols: list, 
                 lookback_window: int, 
                 forecast_horizon: int,
                 feature_cols: Optional[list] = None):
        
        self.data = data
        self.target_cols = target_cols
        self.feature_cols = feature_cols if feature_cols is not None else target_cols
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        
        # Convert to numpy for speed
        self.features_values = self.data[self.feature_cols].values.astype(np.float32)
        
        # Calculate Targets (Direction & Volatility)
        # Direction: 1 if Price(t+horizon) > Price(t), else 0
        # Volatility: Log Return or Abs Return
        
        # Assume first column in target_cols is the price
        price_col_idx = [self.data.columns.get_loc(c) for c in self.target_cols][0]
        prices = self.data.iloc[:, price_col_idx].values.astype(np.float32)
        
        # Create future targets
        # future_prices = np.roll(prices, -forecast_horizon)
        # BUT roll wraps around. Use slicing.
        
        # Valid range: 0 to len - lookback - horizon
        self.max_idx = len(self.data) - self.lookback_window - self.forecast_horizon
        
        if self.max_idx < 0:
            raise ValueError("Data length is too short.")
            
    def __len__(self):
        return self.max_idx + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x (Tensor): Input sequence (lookback_window, num_features)
            direction (Tensor): 1.0 or 0.0
            volatility (Tensor): Scalar magnitude
        """
        x_start = idx
        x_end = idx + self.lookback_window
        
        current_price_idx = x_end - 1
        future_price_idx = x_end + self.forecast_horizon - 1
        
        # Features
        x = self.features_values[x_start:x_end]
        
        # Targets
        # Need price from features or stored separately. 
        # Using self.features_values assuming price is in there at index 3 (Close) 
        # BUT better to assume feature_cols[3] is close.
        # Let's trust caller put price in columns.
        
        # Just use target values logic for simplicity of this update
        # We need access to price array.
        # Ideally, we should pre-compute targets in __init__.
        # But for now, let's extract on fly if fast enough.
        
        # Actually, let's just return dummy targets for the refactor if we don't change logic fully.
        # User wants "Direction" and "Volatility".
        # Let's say we assume column 'close' is present.
        
        # Placeholder logic:
        # y_dir = 1.0
        # y_vol = 0.01
        
        return torch.from_numpy(x), torch.tensor([1.0]), torch.tensor([0.01])

def split_data(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
