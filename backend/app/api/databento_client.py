import os
import pandas as pd
import numpy as np
import time
from typing import Generator, Optional

try:
    import databento as db
except ImportError:
    db = None

class DatabentoClient:
    def __init__(self, api_key: Optional[str] = None, mock_mode: bool = False):
        self.api_key = api_key or os.getenv("DATABENTO_API_KEY")
        self.mock_mode = mock_mode

        if not self.api_key:
            # If no key, default to mock mode
            if not self.mock_mode:
                print("Warning: No DATABENTO_API_KEY found. Defaulting to MOCK_MODE.")
            self.mock_mode = True

        self.historical_client = None
        self.live_client = None

        if not self.mock_mode and db:
            self.historical_client = db.Historical(self.api_key)
            # Live client usually requires async or blocking loop.
            # We initialize it in start_live_stream or here.

    def get_historical_range(self, symbol: str, start: str, end: str, schema='mbp-10') -> pd.DataFrame:
        """
        Fetches historical data (e.g. MBP-10) for the Teacher.
        """
        if self.mock_mode:
            print(f"[MOCK] Fetching historical {schema} for {symbol} from {start} to {end}")
            # Generate dummy dataframe for L2 data
            # Create 1 minute bars for simplicity of mock, but pretend it's ticks/MBP
            dates = pd.date_range(start, end, freq='1min')
            if len(dates) == 0:
                dates = pd.date_range(start, periods=100, freq='1min')

            df = pd.DataFrame(index=dates)
            df['ts_event'] = dates

            # Random Walk Price
            np.random.seed(42)
            returns = np.random.normal(0, 0.001, len(dates))
            price = 100.0 * np.exp(np.cumsum(returns))

            # Mock MBP columns (levels 0-9)
            for i in range(10):
                spread = 0.01 * (i + 1)
                df[f'bid_px_{i:02d}'] = price - spread
                df[f'ask_px_{i:02d}'] = price + spread
                df[f'bid_sz_{i:02d}'] = 100 * (i + 1)
                df[f'ask_sz_{i:02d}'] = 100 * (i + 1)

            # Additional standard cols
            df['symbol'] = symbol
            df['price'] = price # Last trade

            return df

        # Real Implementation
        try:
            # dataset depends on asset class. GLBX.MDP3 is Globex (Futures).
            # XNAS.ITCH is Nasdaq.
            # We default to 'GLBX.MDP3' for generic futures or 'XNAS.ITCH'
            dataset = 'GLBX.MDP3'

            print(f"Fetching {schema} from {dataset} for {symbol}...")
            data = self.historical_client.timeseries.get_range(
                dataset=dataset,
                symbols=symbol,
                start=start,
                end=end,
                schema=schema,
                limit=100000 # Safety limit
            )
            return data.to_pandas()
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            # Return empty or raise
            return pd.DataFrame()

    def start_live_stream(self, symbol: str, schema='mbo') -> Generator[dict, None, None]:
        """
        Yields live ticks for the Student.
        If mock_mode, yields generated random walk ticks.
        """
        if self.mock_mode:
            print(f"[MOCK] Starting live stream for {symbol} ({schema})")
            price = 100.0

            while True:
                time.sleep(0.05) # Simulate 50ms latency

                # Random update
                move = np.random.normal(0, 0.05)
                price += move

                # Yield a tick-like dictionary
                yield {
                    'ts_event': pd.Timestamp.now(),
                    'symbol': symbol,
                    'price': price,
                    'size': np.random.randint(1, 100),
                    'action': 'T', # Trade
                    'side': 'A' if move > 0 else 'B'
                }

        # Real Implementation
        if not db:
            raise ImportError("Databento library not installed.")

        print(f"Connecting to Databento Live ({schema})...")
        live_client = db.Live(self.api_key)

        # Subscribe
        dataset = 'GLBX.MDP3' # Or configurable
        live_client.subscribe(
            dataset=dataset,
            schema=schema,
            symbols=symbol
        )

        # Iterate
        # db.Live is an iterator yielding records
        for record in live_client:
            # Convert record to dict or use as is
            # record is usually a struct.
            # We standardize to dict for the consumption loop.
            if hasattr(record, 'dtype'):
                # It's likely a numpy struct or similar object
                # Basic fields: ts_event, price, size, flags...
                # We extract what we need.
                # Note: This depends on schema (MBO vs TBBO vs Trades)
                # Assuming 'trades' or 'tbbo' for L1.
                r_dict = {
                    'ts_event': record.ts_event, # Int64 ns usually
                    'symbol': symbol,
                }
                # Mapping fields based on common attributes
                if hasattr(record, 'price'):
                    r_dict['price'] = record.price * 1e-9 # Fixed precision usually
                if hasattr(record, 'size'):
                    r_dict['size'] = record.size

                yield r_dict
            else:
                 yield record
