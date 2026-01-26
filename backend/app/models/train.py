from pydantic import BaseModel
from typing import Optional, Dict, List, Any

class TrainRequest(BaseModel):
    data_source: str = "synthetic" # synthetic, yfinance
    symbol: str = "AAPL"
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    timeframe: str = "1d"
    
    # Model Config
    model_type: str = "patchtst"
    lookback_window: int = 64
    forecast_horizon: int = 16
    patch_len: int = 8
    stride: int = 4
    d_model: int = 64
    
    # Training Config
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3

class TrainResponse(BaseModel):
    status: str
    metrics: Dict[str, Any]
    training_time: float
