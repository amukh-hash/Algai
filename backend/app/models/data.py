from pydantic import BaseModel
from typing import Optional, Dict, List, Any

class DataRequest(BaseModel):
    source: str = "synthetic" # synthetic, yfinance
    symbol: str = "AAPL"
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    timeframe: str = "1d"
    params: Optional[Dict[str, Any]] = {}

class PreprocessRequest(BaseModel):
    data: List[Dict[str, Any]] # List of records
    method: str = "fractional_diff"
    params: Optional[Dict[str, Any]] = {}

class DataResponse(BaseModel):
    count: int
    data: List[Dict[str, Any]]
