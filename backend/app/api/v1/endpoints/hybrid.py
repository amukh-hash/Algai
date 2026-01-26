from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Any
from app.engine.pipe import InferencePipe
from app.data.parquet_manager import ParquetOptionsLoader, SyntheticOptionsGenerator
from app.engine.vectorized import VectorizedExecutioner
from app.models.patchtst import HybridPatchTST
import os

router = APIRouter()

class HybridRequest(BaseModel):
    symbol: str = "AAPL"
    start_date: str = "2023-01-01"
    end_date: str = "2023-01-05"

import pandas as pd

@router.post("/run", response_model=List[Any])
def run_hybrid_pipeline(request: HybridRequest):
    try:
        # Setup Data
        data_dir = "backend/data" # Local data dir
        gen = SyntheticOptionsGenerator(data_dir)
        # Generate some data if missing
        dt = pd.to_datetime(request.start_date)
        gen.generate_month(request.symbol, dt.year, dt.month)

        loader = ParquetOptionsLoader(data_dir)

        # Setup Model
        model = HybridPatchTST(num_input_features=5, lookback_window=64)
        model.eval()

        # Setup Executioner
        executioner = VectorizedExecutioner()

        # Pipe
        pipe = InferencePipe(model, loader, executioner, device="cpu") # Force CPU for dev env without GPU

        trades = pipe.run_pipeline(request.symbol, request.start_date, request.end_date)

        # Result is already a list of dicts
        return trades

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
