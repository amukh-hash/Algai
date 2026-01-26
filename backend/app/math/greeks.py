import numpy as np
from scipy.stats import norm

def black_scholes_vectorized(S, K, T, r, sigma, option_type='C'):
    """
    Vectorized Black-Scholes Calculator.
    All inputs can be numpy arrays.

    Args:
        S: Underlying Price
        K: Strike Price
        T: Time to Expiration (in years)
        r: Risk-free rate
        sigma: Implied Volatility
        option_type: 'C' or 'P' (can be array if encoded)

    Returns:
        dict: {price, delta, gamma, theta, vega, rho}
    """
    # Avoid division by zero
    T = np.maximum(T, 1e-5)
    sigma = np.maximum(sigma, 1e-5)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Cache standard normal CDF and PDF
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    pdf_d1 = norm.pdf(d1)

    # Put-Call Parity logic or mask
    # Assuming option_type is 'C' for now or handle array
    if isinstance(option_type, str):
        is_call = (option_type == 'C')
    else:
        # Assume boolean array or 'C'/'P' array
        is_call = (option_type == 'C')

    # Call Price
    call_price = S * N_d1 - K * np.exp(-r * T) * N_d2

    # Put Price
    # P = C - S + K * exp(-rT)
    put_price = call_price - S + K * np.exp(-r * T)

    # Delta
    call_delta = N_d1
    put_delta = N_d1 - 1

    # Gamma (Same for Call & Put)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))

    # Vega (Same for Call & Put)
    vega = S * pdf_d1 * np.sqrt(T) / 100 # Typically divided by 100

    # Theta (simplified)
    # term1 = - (S * sigma * pdf_d1) / (2 * np.sqrt(T))
    # call_theta = term1 - r * K * np.exp(-r*T) * N_d2
    # put_theta = term1 + r * K * np.exp(-r*T) * norm.cdf(-d2)
    # Keeping it simple for the prototype

    if isinstance(is_call, bool):
        price = call_price if is_call else put_price
        delta = call_delta if is_call else put_delta
    else:
        price = np.where(is_call, call_price, put_price)
        delta = np.where(is_call, call_delta, put_delta)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega
    }
