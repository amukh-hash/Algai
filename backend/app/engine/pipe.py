import torch
import pandas as pd
import numpy as np
from app.data.parquet_manager import ParquetOptionsLoader
from app.engine.vectorized import VectorizedExecutioner
from app.models.patchtst import HybridPatchTST
import concurrent.futures
import threading

class InferencePipe:
    """
    Async Pipe connecting Forecaster (GPU) and Executioner (CPU).
    Uses ThreadPoolExecutor for IO (Data Loading) to keep GPU/CPU busy.
    """
    def __init__(self,
                 model: HybridPatchTST,
                 data_loader: ParquetOptionsLoader,
                 executioner: VectorizedExecutioner,
                 device: str = "cuda"):
        self.model = model
        self.data_loader = data_loader
        self.executioner = executioner
        self.device = device

        if torch.cuda.is_available() and device == "cuda":
            self.model.cuda()
            try:
                print("Compiling model...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                print(f"Compilation skipped: {e}")

    def run_pipeline(self, symbol: str, start_date: str, end_date: str):
        """
        Async Loop:
        1. Prefetch Data (Thread)
        2. Infer (GPU)
        3. Execute (CPU)
        """
        dates = pd.date_range(start=start_date, end=end_date)

        # Thread Pool for IO
        io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # State
        future_data = None

        # Pre-load first day
        first_date = dates[0].strftime('%Y-%m-%d')
        future_data = io_pool.submit(self._load_day_data, symbol, first_date)

        for i, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')

            # 1. Wait for Data (Blocking main thread only if IO is slower than compute)
            try:
                current_day_data = future_data.result()
            except FileNotFoundError:
                current_day_data = None
            except Exception as e:
                print(f"Error loading data: {e}")
                current_day_data = None

            # Trigger Next Load (Prefetch)
            if i + 1 < len(dates):
                next_date = dates[i+1].strftime('%Y-%m-%d')
                future_data = io_pool.submit(self._load_day_data, symbol, next_date)

            if current_day_data is None or current_day_data.empty:
                continue

            # 2. Update Positions (CPU)
            # Update P&L with new data
            self.executioner.update_positions(current_day_data)
            self.executioner.check_exits()

            # 3. GPU Inference
            # Mock Input (Batch=1, Lookback=64, Features=5)
            # In real system, load stock data separately.
            input_tensor = torch.randn(1, 64, 5).float()
            if self.device == "cuda":
                input_tensor = input_tensor.pin_memory().cuda(non_blocking=True)

            with torch.no_grad():
                direction, volatility = self.model(input_tensor)

            direction = direction.cpu().item()
            volatility = volatility.cpu().item()

            preds = pd.DataFrame([{
                'date': date_str,
                'direction_prob': direction,
                'vol_pred': volatility
            }])

            # 4. Execute (CPU)
            self.executioner.select_contracts(preds, current_day_data)

        io_pool.shutdown()
        return self.executioner.closed_trades + self.executioner.open_positions.to_dict(orient="records")

    def _load_day_data(self, symbol: str, date: str) -> pd.DataFrame:
        """
        Helper to load specific day data from Parquet partitions.
        Consumes the generator to return full day DataFrame for vectorized execution.
        """
        # Note: If day is huge, this might need sub-chunking.
        # But VectorizedExecutioner expects a DataFrame to merge against.
        # We assume 32GB RAM can hold 1 day of options for 1 ticker.
        chunks = list(self.data_loader.load_chunked(symbol, date))
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks)
