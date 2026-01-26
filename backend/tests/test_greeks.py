import numpy as np
import pytest
from app.math.greeks import black_scholes_vectorized

def test_black_scholes_call():
    # Known values: S=100, K=100, T=1, r=0.05, sigma=0.2
    # C should be approx 10.45
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    
    res = black_scholes_vectorized(S, K, T, r, sigma, 'C')
    price = res['price']
    delta = res['delta']
    
    assert np.isclose(price, 10.45, atol=0.01)
    # Delta of ATM call approx 0.63 (N(d1))
    assert 0.5 < delta < 0.7

def test_black_scholes_put():
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    
    res = black_scholes_vectorized(S, K, T, r, sigma, 'P')
    price = res['price']
    
    # Put-Call Parity: C - P = S - K*exp(-rT)
    # 10.45 - P = 100 - 100*exp(-0.05) = 100 - 95.12 = 4.88
    # P = 10.45 - 4.88 = 5.57
    assert np.isclose(price, 5.57, atol=0.01)

def test_vectorized_inputs():
    S = np.array([100, 100])
    K = np.array([100, 110])
    T = 1
    r = 0.05
    sigma = 0.2
    
    res = black_scholes_vectorized(S, K, T, r, sigma, 'C')
    
    assert len(res['price']) == 2
    assert res['price'][0] > res['price'][1] # Higher strike -> Lower Call Price
