import pytest
import pandas as pd
import numpy as np
from app.engine.vectorized import VectorizedExecutioner

def test_vectorized_lifecycle():
    exec = VectorizedExecutioner(initial_capital=10000)
    
    # Day 1: Signal to Buy
    # Market Data Day 1
    # AAPL Call 150 Exp 2023-01-20
    # Ask = 5.0
    day1 = pd.DataFrame([{
        "date": "2023-01-01",
        "root": "AAPL",
        "expiration": "2023-01-20",
        "strike": 150,
        "right": "C",
        "bid": 4.8,
        "ask": 5.0,
        "delta": 0.3 # Target delta
    }])
    
    preds = pd.DataFrame([{
        "date": "2023-01-01",
        "direction_prob": 0.9, # High conf
        "vol_pred": 0.2
    }])
    
    # Select
    exec.select_contracts(preds, day1)
    assert len(exec.open_positions) == 1
    assert exec.open_positions.iloc[0]["entry_price"] == 5.0
    assert exec.cash == 10000 - 500
    
    # Day 2: Price moves UP
    # Call Price increases to 7.0
    day2 = pd.DataFrame([{
        "date": "2023-01-02",
        "root": "AAPL",
        "expiration": "2023-01-20",
        "strike": 150,
        "right": "C",
        "bid": 7.0, # Bid is exit price
        "ask": 7.2
    }])
    
    # Update
    exec.update_positions(day2)
    # Check PnL
    # (7.0 - 5.0) * 100 * 1 = 200
    assert np.isclose(exec.open_positions.iloc[0]["pnl"], 200.0)
    assert np.isclose(exec.open_positions.iloc[0]["pnl_pct"], 0.4) # 40%
    
    # Check Exits (TP=0.5, SL=-0.2)
    # 40% < 50%, no exit
    exec.check_exits(take_profit=0.5)
    assert len(exec.open_positions) == 1
    
    # Day 3: Price MOONS
    # Call Price -> 10.0
    day3 = pd.DataFrame([{
        "date": "2023-01-03",
        "root": "AAPL",
        "expiration": "2023-01-20",
        "strike": 150,
        "right": "C",
        "bid": 10.0,
        "ask": 10.2
    }])
    
    exec.update_positions(day3)
    # PnL% = (10-5)/5 = 1.0 (100%)
    # Should exit
    exec.check_exits(take_profit=0.5)
    
    assert len(exec.open_positions) == 0
    assert len(exec.closed_trades) == 1
    assert exec.closed_trades[0]["exit_reason"] == "TP"
    assert exec.cash == 9500 + 1000 # 10500
