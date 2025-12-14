# market.py
import numpy as np
from scipy.stats import norm

DT = 1 / 252
T_STEPS = 2000
N_PATHS = 200

N_NOISE = 60
N_FUND = 20
N_MOM = 20

S0 = 100.0
FUNDAMENTAL_VALUE = 100.0
RISK_FREE = 0.01
K_STRIKE = 100.0
T_MATURITY = 30 / 252
SIGMA0 = 0.15

PRICE_IMPACT = 0.0008
EXOG_LOGNOISE_SIGMA = 0.0012
MAX_DLOGS = 0.02

LIQ_SHOCK_PROB = 0.003
LIQ_SHOCK_MULT = 6.0

MOM_ALPHA = 0.12
MOM_LOOKBACK = 15
MOM_MAX_ORDER = 0.8

FLASH_CRASH_ABS = -0.035
CRASH_Z = -3.0
CRASH_Z_WIN = 50

ALPHA = 0.05

TOTAL_OPTION_NOTIONAL = 6000.0

def bs_delta_gamma_call(S, K, r, sigma, T):
    S = max(float(S), 1e-12)
    sigma = max(float(sigma), 1e-12)
    T = max(float(T), 1e-12)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return float(delta), float(gamma)


def estimate_hist_vol(log_returns, win=50, fallback=SIGMA0):
    r = np.asarray(log_returns, dtype=float)
    if len(r) < win + 1:
        return float(fallback)
    return float(np.std(r[-win:], ddof=1) * np.sqrt(252.0))


def realized_volatility(log_returns):
    r = np.asarray(log_returns, dtype=float)
    if len(r) < 2:
        return np.nan
    return float(np.std(r, ddof=1) * np.sqrt(252.0))


def max_drawdown(prices):
    p = np.asarray(prices, dtype=float)
    peak = p[0]
    mdd = 0.0
    for x in p:
        peak = max(peak, x)
        mdd = min(mdd, (x - peak) / peak)
    return float(mdd)


def crash_count_abs(log_returns, thresh=FLASH_CRASH_ABS):
    r = np.asarray(log_returns, dtype=float)
    return int(np.sum(r < float(thresh)))


def crash_count_zscore(log_returns, win=CRASH_Z_WIN, z=CRASH_Z, eps=1e-12):
    r = np.asarray(log_returns, dtype=float)
    if len(r) < win + 1:
        return 0
    cnt = 0
    for t in range(win, len(r)):
        s = np.std(r[t - win : t], ddof=1)
        if s > eps and r[t] < z * s:
            cnt += 1
    return int(cnt)


def crash_flags_zscore(log_returns, win=CRASH_Z_WIN, z=CRASH_Z, eps=1e-12):
    r = np.asarray(log_returns, dtype=float)
    flags = np.zeros(len(r), dtype=bool)
    if len(r) < win + 1:
        return flags
    for t in range(win, len(r)):
        s = np.std(r[t - win : t], ddof=1)
        if s > eps and r[t] < z * s:
            flags[t] = True
    return flags
