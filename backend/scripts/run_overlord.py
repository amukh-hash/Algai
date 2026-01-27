import os
import sys
import time
import json
import logging
import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.models.patchtst import HybridPatchTST
# from app.utils.chronos_adapter import ChronosAdapter # We need to implementing this adapter properly if not exists
# or just direct loading logic

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backend/logs/overlord.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("OVERLORD")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

class Overlord:
    """
    The Overlord: Infinite Execution Loop.
    Orchestrates Data -> Physics -> Judge -> Execution.
    """
    def __init__(self, mode="paper", interval_seconds=60):
        self.mode = mode
        self.interval = interval_seconds
        self.tickers = ["NVDA", "SPY", "QQQ", "IWM", "TSLA"] 
        
        logger.info(f"Initializing OVERLORD in {mode.upper()} mode...")
        
        # Load Config
        load_dotenv()
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            logger.critical("Alpaca Credentials NOT FOUND in .env")
            raise ValueError("Missing Credentials")
            
        # Initialize Alpaca Clients
        self.paper = True if mode == "paper" else False
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        # Verify Account
        try:
            account = self.trading_client.get_account()
            logger.info(f"Connected to Alpaca. ID: {account.id}")
            logger.info(f"Buying Power: ${account.buying_power} | Cash: ${account.cash}")
            if account.trading_blocked:
                logger.warning("Account is BLOCKED from trading!")
        except Exception as e:
            logger.critical(f"Failed to connect to Alpaca: {e}")
            raise e
        
        # Initialize Models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_physics_engine()
        self._load_the_judge()
        
        logger.info("Systems Online.")

    def _load_physics_engine(self):
        logger.info("Loading Physics Engine (Chronos)...")
        # Placeholder: In production we load the pipeline here
        self.physics_ready = True

    def _load_the_judge(self):
        logger.info("Loading The Judge (XGBoost)...")
        path = "backend/models/judge.json"
        if os.path.exists(path):
            self.judge = xgb.Booster()
            self.judge.load_model(path)
            self.judge_ready = True
            logger.info("Judge Loaded.")
        else:
            logger.warning("Judge Model NOT FOUND. Running in LAWLESS mode (Physics Only).")
            self.judge = None
            self.judge_ready = False

    def fetch_live_data(self, ticker):
        """
        Fetch last 64 minutes of data for Chronos context.
        """
        try:
            # We need 64 bars + context? 
            # Chronos needs context_length (e.g. 512) to predict.
            # Only fetching 1000 to be safe.
            now = datetime.now()
            start = now - timedelta(days=5) # 5 days should cover weekends/holidays gap for 1m bars
            
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=start,
                limit=1000,
                adjustment='split'
            )
            
            bars = self.data_client.get_stock_bars(req)
            if not bars.data:
                logger.warning(f"[{ticker}] No data found.")
                return pd.DataFrame()
                
            df = bars.df.reset_index()
            # Columns: symbol, timestamp, open, high, low, close, volume, trade_count, vwap
            
            # Format for Physics Engine
            # We need specific columns.
            # Chronos needs a univariate series usually, or multivariate 'close', 'open', 'high', 'low', 'volume'
            
            # Filter for just this ticker (though request was specific)
            ticker_df = df[df['symbol'] == ticker].copy()
            ticker_df.set_index('timestamp', inplace=True)
            
            # Resample to ensure 1 min grid (fill gaps)
            # ticker_df = ticker_df.resample('1T').ffill()
            
            return ticker_df.tail(600) # Return enough context
            
        except Exception as e:
            logger.error(f"[{ticker}] Data Fetch Failed: {e}")
            return pd.DataFrame()

    def run(self):
        logger.info(f"--- ENTERING INFINITE LOOP ({self.interval}s interval) ---")
        try:
            while True:
                start_time = time.time()
                
                self.tick()
                
                # Sleep remainder of interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("Overlord Shutting Down...")
        except Exception as e:
            logger.critical(f"CRITICAL FAILURE: {e}", exc_info=True)
            # Restart logic?

    def tick(self):
        """
        One Unified Heartbeat.
        """
        if not self.is_market_open():
            logger.info("Market Closed. Sleeping...")
            return

        logger.info(f"Tick: {datetime.now()}")
        
        for ticker in self.tickers:
            self.process_ticker(ticker)

    def is_market_open(self):
        # Apply simple time check or Alpaca Clock
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except:
            return True # Fallback for Paper testing

    def process_ticker(self, ticker):
        # 1. Get Data
        df = self.fetch_live_data(ticker)
        if df.empty or len(df) < 64:
            logger.warning(f"[{ticker}] Insufficient Data ({len(df)} rows).")
            return

        # 2. Physics Inference (Forecast)
        # Mocking signal for now until Chronos Adapter is connected
        # signal = self.physics.predict(df)
        signal = {"dir": 0, "conf": 0.0} 
        
        # 3. Judge Review
        if self.judge_ready:
            # DMatrix creation mocked
            # prob_success = self.judge.predict(...)
            pass
        
        # 4. Execution Logic
        # Simple strategy: 
        # If Dir=1 (Buy) and No Position -> Buy
        # If Dir=2 (Sell) and Has Position -> Sell
        
        try:
            # Check Position
            try:
                pos = self.trading_client.get_open_position(ticker)
                has_position = float(pos.qty) > 0
            except:
                has_position = False
            
            # Logic
            if signal['dir'] == 1 and not has_position:
                if signal['conf'] > 0.7:
                    logger.info(f"[{ticker}] BUY SIGNAL (Conf {signal['conf']:.2f}). Executing...")
                    self.execute_order(ticker, OrderSide.BUY)
            
            elif signal['dir'] == 2 and has_position:
                logger.info(f"[{ticker}] SELL SIGNAL. Closing Position.")
                self.execute_order(ticker, OrderSide.SELL)
                
        except Exception as e:
            logger.error(f"[{ticker}] Execution Logic Failed: {e}")

    def execute_order(self, ticker, side):
        req = MarketOrderRequest(
            symbol=ticker,
            qty=1, # Fixed qty for safety
            side=side,
            time_in_force=TimeInForce.DAY
        )
        
        try:
            order = self.trading_client.submit_order(req)
            logger.info(f"[{ticker}] ORDER SUBMITTED: {side} 1. ID: {order.id}")
        except Exception as e:
            logger.error(f"[{ticker}] ORDER FAILED: {e}")

if __name__ == "__main__":
    overlord = Overlord(mode="paper")
    overlord.run()
