import numpy as np
import pandas as pd

def get_weights_ffd(d, thres=1e-5):
    """
    Computes weights for fractional differentiation using fixed window.
    
    Args:
        d (float): Order of differentiation (0 <= d <= 1).
        thres (float): Threshold for weight cutoff.
        
    Returns:
        np.array: Weights vector w (reversed).
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w)

def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    """
    Applies Fractional Differentiation with Fixed Window (FFD) to a pandas Series.
    
    Args:
        series (pd.Series): Time series data.
        d (float): Order of differentiation.
        thres (float): Threshold to determine window size (lag cutoff).
        
    Returns:
        pd.Series: Stationarized series. Note: The first few values will be NaN.
    """
    # 1. Compute weights
    w = get_weights_ffd(d, thres)
    width = len(w)
    
    # 2. Apply weights
    if len(series) < width:
        raise ValueError(f"Series length {len(series)} is smaller than FFD window width {width} (d={d}, thres={thres})")

    output = pd.Series(index=series.index, dtype=float)
    output[:] = np.nan
    
    values = series.values
    
    # np.convolve(x, w, mode='valid')
    # If w = [w_0, w_1, ...], convolve implicitly flips w.
    # Result at t (end of window): w_0*x[t] + w_1*x[t-1] + ...
    # This is exactly what we want.
    
    conv_out = np.convolve(values, w, mode='valid')
    
    output.iloc[width-1:] = conv_out
    
    return output

class FractionalDifferencer:
    """
    Class wrapper for applying Fractional Differencing to a DataFrame.
    """
    def __init__(self, d: float = 0.4, thres: float = 1e-4):
        self.d = d
        self.thres = thres
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies frac diff to numeric columns in df.
        """
        out_df = df.copy()
        target_cols = ['open', 'high', 'low', 'close']
        for col in target_cols:
            if col in df.columns:
                out_df[col] = frac_diff_ffd(df[col], self.d, self.thres)
                
        return out_df
