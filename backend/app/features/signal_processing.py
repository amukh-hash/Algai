import numpy as np
import pandas as pd
import pywt
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from numba import jit
import warnings

# --- 1. Teacher: Acausal Smoothing (MODWT/SWT + UKS) ---

def apply_modwt_uks(data: np.ndarray, wavelet='db4', level=3):
    original_len = len(data)
    pad_len = int(np.ceil(original_len / (2**level))) * (2**level) - original_len
    if pad_len > 0:
        data_padded = np.pad(data, (0, pad_len), 'edge')
    else:
        data_padded = data

    coeffs = pywt.swt(data_padded, wavelet, level=level, start_level=0)

    new_coeffs = []
    for (cA, cD) in coeffs:
        sigma = np.median(np.abs(cD)) / 0.6745 + 1e-6
        limit = sigma * np.sqrt(2 * np.log(len(cD)))
        cD_thresh = pywt.threshold(cD, limit, mode='soft')
        new_coeffs.append((cA, cD_thresh))

    denoised_signal = pywt.iswt(new_coeffs, wavelet)
    denoised_signal = denoised_signal[:original_len]

    def fx(x, dt): return np.array([x[0] + x[1]*dt, x[1]])
    def hx(x): return np.array([x[0]])

    dt = 1.0
    points = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2., kappa=0)

    ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=dt, fx=fx, hx=hx, points=points)
    ukf.x = np.array([denoised_signal[0], 0.])
    ukf.P *= 10.0
    ukf.R = np.std(np.diff(denoised_signal))**2 + 1e-6
    ukf.Q = np.eye(2) * 0.01

    mu, cov = ukf.batch_filter(denoised_signal)
    M, P, K = ukf.rts_smoother(mu, cov)

    return M[:, 0]

# --- 2. Student: Causal Filtering (Sliding Wavelet + UKF) ---

def apply_sliding_wavelet_ukf(data_window: np.ndarray, ukf_object=None):
    try:
        coeffs = pywt.wavedec(data_window, 'db4', level=2, mode='symmetric')
        cD = coeffs[-1]
        sigma = np.median(np.abs(cD)) / 0.6745 + 1e-6
        limit = sigma * np.sqrt(2 * np.log(len(data_window)))
        new_coeffs = [coeffs[0]] + [pywt.threshold(c, limit, mode='soft') for c in coeffs[1:]]
        rec = pywt.waverec(new_coeffs, 'db4', mode='symmetric')

        if len(rec) >= len(data_window):
            observation = rec[len(data_window)-1]
        else:
            observation = rec[-1]
    except Exception as e:
        observation = data_window[-1]

    if ukf_object is None:
        def fx(x, dt): return np.array([x[0] + x[1]*dt, x[1]])
        def hx(x): return np.array([x[0]])
        points = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2., kappa=0)
        ukf_object = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=1.0, fx=fx, hx=hx, points=points)
        ukf_object.x = np.array([observation, 0.])
        ukf_object.P *= 1.0
        ukf_object.R = 0.1
        ukf_object.Q = np.eye(2) * 0.01

    ukf_object.predict()
    ukf_object.update(observation)

    return ukf_object.x[0], ukf_object

# --- 3. Trend Scanning (Numba) ---

@jit(nopython=True)
def trend_scanning_labels(prices, window_min=10, window_max=100):
    n = len(prices)
    labels = np.zeros(n)
    prices_val = prices.astype(np.float64)

    for t in range(n - window_max):
        max_t_stat = 0.0
        for l in range(window_min, window_max):
            end_idx = t + l
            if end_idx >= n: break
            y = prices_val[t : end_idx]
            len_y = len(y)
            x = np.arange(len_y)
            x_mean = (len_y - 1) / 2.0
            y_mean = np.mean(y)
            cov_xy = np.sum((x - x_mean) * (y - y_mean))
            var_x = np.sum((x - x_mean)**2)
            if var_x == 0: continue
            slope = cov_xy / var_x
            y_pred = y_mean + slope * (x - x_mean)
            rss = np.sum((y - y_pred)**2)
            if rss == 0: t_stat = 0.0
            elif len_y <= 2: t_stat = 0.0
            else:
                sigma_sq = rss / (len_y - 2)
                se_slope = np.sqrt(sigma_sq / var_x)
                t_stat = slope / se_slope if se_slope != 0 else 0.0
            if np.abs(t_stat) > np.abs(max_t_stat):
                max_t_stat = t_stat
        labels[t] = max_t_stat
    return labels

# --- 4. Triple Barrier Method ---

def triple_barrier_labels(prices: pd.Series, volatility: pd.Series, vertical_barrier_window=20, sl_tp_ratio=2.0):
    timestamps = prices.index
    vals = prices.values
    vols = volatility.values
    n = len(vals)
    out_labels = np.zeros(n, dtype=int)

    for t in range(n):
        current_price = vals[t]
        vol = vols[t]
        if np.isnan(vol) or vol == 0: continue
        k = 1.0
        upper = current_price * (1 + k * vol)
        lower = current_price * (1 - k * vol / sl_tp_ratio)
        horizon = min(t + vertical_barrier_window, n)
        label = 0
        for future_t in range(t + 1, horizon):
            p = vals[future_t]
            if p >= upper:
                label = 1
                break
            if p <= lower:
                label = -1
                break
        out_labels[t] = label

    return pd.Series(out_labels, index=timestamps)
