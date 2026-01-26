import pandas as pd
import numpy as np

class VectorizedExecutioner:
    """
    The Executioner.
    Matches AI predictions to Option Chains and computes P&L using vectorized ops.
    """
    def __init__(self, initial_capital: float = 100000.0):
        self.capital = initial_capital
        self.cash = initial_capital
        
        # DataFrame columns: [entry_date, root, expiration, strike, right, entry_price, quantity, current_price, pnl, status]
        self.open_positions = pd.DataFrame(columns=[
            "entry_date", "root", "expiration", "strike", "right", 
            "entry_price", "quantity", "current_price", "pnl", "status", "max_pnl_pct"
        ])
        
        self.closed_trades = []

    def update_positions(self, market_data: pd.DataFrame):
        """
        Vectorized update of open positions using the day's market data.
        market_data: DataFrame [date, root, expiration, strike, right, bid, ask, ...]
        """
        if self.open_positions.empty:
            return

        # Prepare Join Keys
        join_cols = ["root", "expiration", "strike", "right"]
        
        # We assume market_data has unique rows for these keys per timestamp. 
        # If market_data is the whole day chunk, we might need to filter or aggregate?
        # For this prototype, assume market_data is the "Close" snapshot of the day.
        
        # Left Join Open Positions with Market Data
        # Suffixes: _pos (original), _mkt (new)
        merged = pd.merge(
            self.open_positions, 
            market_data[join_cols + ["bid", "ask", "date"]], 
            on=join_cols, 
            how="left",
            suffixes=("", "_new")
        )
        
        # Update Prices (Mark to Market: Mid Price or Bid for Longs)
        # For Long Calls, we exit at Bid.
        # Handle NaN (if data missing for that contract today) - Forward Fill or keep old price
        merged["current_price"] = merged["bid"].combine_first(merged["current_price"])
        
        # Update P&L
        # PnL = (Current - Entry) * Qty * 100 (options multiplier)
        merged["pnl"] = (merged["current_price"] - merged["entry_price"]) * merged["quantity"] * 100
        merged["pnl_pct"] = (merged["current_price"] - merged["entry_price"]) / merged["entry_price"]
        
        # Update High Water Mark for Trailing Stop
        if "max_pnl_pct" not in merged.columns:
             merged["max_pnl_pct"] = -1.0
             
        merged["max_pnl_pct"] = np.maximum(merged["max_pnl_pct"], merged["pnl_pct"])
        
        self.open_positions = merged[self.open_positions.columns] # Keep schema
        # Add temporary calc columns for check_exits if needed, or recalculate there.
        self.open_positions["pnl_pct"] = merged["pnl_pct"]
        self.open_positions["current_date"] = merged["date"]

    def check_exits(self, take_profit: float = 0.5, stop_loss: float = -0.2):
        """
        Vectorized Exit Logic.
        """
        if self.open_positions.empty:
            return

        # 1. Take Profit
        tp_mask = self.open_positions["pnl_pct"] >= take_profit
        
        # 2. Stop Loss
        sl_mask = self.open_positions["pnl_pct"] <= stop_loss
        
        # 3. Expiration (if current_date >= expiration)
        # Convert strings to datetime if needed, or compare strings ISO format
        exp_mask = self.open_positions["current_date"] >= self.open_positions["expiration"]
        
        exit_mask = tp_mask | sl_mask | exp_mask
        
        if exit_mask.any():
            exits = self.open_positions[exit_mask].copy()
            stays = self.open_positions[~exit_mask].copy()
            
            # Record Exits
            exits["exit_reason"] = np.select(
                [tp_mask[exit_mask], sl_mask[exit_mask], exp_mask[exit_mask]], 
                ["TP", "SL", "EXP"], 
                default="MANUAL"
            )
            exits["exit_price"] = exits["current_price"]
            exits["exit_date"] = exits["current_date"]
            
            # Update Cash
            # Cash += Exit Value
            total_exit_value = (exits["exit_price"] * exits["quantity"] * 100).sum()
            self.cash += total_exit_value
            
            self.closed_trades.extend(exits.to_dict(orient="records"))
            self.open_positions = stays

    def select_contracts(self, 
                        predictions: pd.DataFrame, 
                        options_chain: pd.DataFrame, 
                        min_confidence: float = 0.0): # Relaxed for Demo
        try:
            """
            Enters NEW trades.
            """
            # Filter High Confidence Predictions
            high_conf = predictions[predictions['direction_prob'] > min_confidence].copy()
            
            if high_conf.empty:
                return
            
            # 1. Merge Prediction with Chain
            merged = pd.merge(high_conf, options_chain, on='date', how='inner')
            
            # 2. Directional Matching
            merged = merged[merged['right'] == 'C'] 
            
            # 6. DTE Filter (Relaxed or Wide)
            merged['dte'] = (pd.to_datetime(merged['expiration']) - pd.to_datetime(merged['date'])).dt.days
            # merged = merged[(merged['dte'] >= 14) & (merged['dte'] <= 30)]
            
            if merged.empty:
                return

            # Position Sizing
            # Fixed 1 contract for prototype
            quantity = 1
            
            # Entry
            # Cost = Ask * 100 * Qty
            cost = merged['ask'] * 100 * quantity
            
            # Check Capital
            # Only take trades we can afford
            # Vectorized "cumsum" of cost to see where we run out of cash?
            # Simple: Take top 1
            if not merged.empty:
                 trade = merged.iloc[0] # Take best match
                 trade_cost = trade['ask'] * 100 * quantity
                 
                 if self.cash >= trade_cost:
                     self.cash -= trade_cost
                     
                     new_position = {
                         "entry_date": trade['date'],
                         "root": trade['root'],
                         "expiration": trade['expiration'],
                         "strike": trade['strike'],
                         "right": trade['right'],
                         "entry_price": trade['ask'],
                         "quantity": quantity,
                         "current_price": trade['ask'],
                         "pnl": 0.0,
                         "status": "OPEN",
                         "max_pnl_pct": 0.0,
                         "pnl_pct": 0.0,
                         "current_date": trade['date']
                     }
                     
                     # Append to open_positions
                     # Use pd.concat
                     new_df = pd.DataFrame([new_position])
                     # Clean columns to match
                     new_df = new_df[self.open_positions.columns]
                     self.open_positions = pd.concat([self.open_positions, new_df], ignore_index=True)
                     print("DEBUG: Trade executed!")
        except Exception as e:
            import traceback
            err_msg = f"ERROR in select_contracts: {e}\n{traceback.format_exc()}"
            print(err_msg)
            with open("error.log", "w") as f:
                f.write(err_msg)
            raise e
