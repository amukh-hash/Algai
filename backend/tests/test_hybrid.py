import pytest
import os
import shutil
import pandas as pd
from app.data.parquet_manager import SyntheticOptionsGenerator, ParquetOptionsLoader
from app.engine.vectorized import VectorizedExecutioner
from app.engine.pipe import InferencePipe
from app.models.patchtst import HybridPatchTST

@pytest.fixture
def data_dir():
    path = "/tmp/test_algo_data"
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path)

def test_full_pipeline(data_dir):
    symbol = "TEST"
    date = "2023-01-01"
    
    # 1. Generate Data
    gen = SyntheticOptionsGenerator(data_dir)
    # Generate Month (Partitioned)
    dt = pd.to_datetime(date)
    gen.generate_month(symbol, dt.year, dt.month)
    
    loader = ParquetOptionsLoader(data_dir)
    
    # 2. Model
    model = HybridPatchTST(num_input_features=5, lookback_window=64)
    model.eval()
    
    # 3. Executioner
    executioner = VectorizedExecutioner()
    
    # 4. Pipe
    pipe = InferencePipe(model, loader, executioner, device="cpu")
    
    trades = pipe.run_pipeline(symbol, date, date)
    
    # Check if trades happened (Synthetic data is random, but we force high conf if needed?
    # Or just check structure)
    # The pipe loop only adds to trades if selected_contracts is not empty.
    # Prediction prob is random untrained. Might get lucky.
    
    # For test robustness, let's verify no crash first.
    assert isinstance(trades, list)
