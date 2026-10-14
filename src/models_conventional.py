"""Conventional forecasting models. Each returns (point, lower80, upper80) in USD/gal for h = 1..12.
All models are fit on log prices; forecasts are converted back with exp()."""
import numpy as np
import pandas as pd
from statsforecast.models import AutoARIMA, AutoETS
from common import H, fourier, future_index

Z = [80]


def naive(hist, origin):
    y = np.log(hist["ca_gas"].values)
    p = np.full(H, np.exp(y[-1]))
    lo, hi = [], []
    for h in range(1, H + 1):
        ch = y[h:] - y[:-h]                      # empirical h-step log changes, history only
        q10, q90 = np.quantile(ch, [0.1, 0.9])
        lo.append(np.exp(y[-1] + q10)); hi.append(np.exp(y[-1] + q90))
    return p, np.array(lo), np.array(hi)


def seasonal_naive(hist, origin):
    y = np.log(hist["ca_gas"].values)
    p = np.exp(y[-52 + np.arange(1, H + 1) - 1])   # value 52 weeks before each target week
    e = y[52:] - y[:-52]
    q10, q90 = np.quantile(e, [0.1, 0.9])
    return p, p * np.exp(q10), p * np.exp(q90)


def ets(hist, origin):
    y = np.log(hist["ca_gas"].values)
    f = AutoETS(model="ZZN", damped=None).forecast(y=y, h=H, level=Z)   # trend type & damping by AICc
    return np.exp(f["mean"]), np.exp(f["lo-80"]), np.exp(f["hi-80"])


def arima_fourier(hist, origin, K=3):
    y = np.log(hist["ca_gas"].values)
    X = fourier(hist.index, K).values
    Xf = fourier(future_index(origin), K).values
    f = AutoARIMA(seasonal=False).forecast(y=y, h=H, X=X, X_future=Xf, level=Z)
    return np.exp(f["mean"]), np.exp(f["lo-80"]), np.exp(f["hi-80"])


CRUDE_LAGS = 4
EVENT_COLS = ["ev_covid", "closures_cum"]


def arimax_design(hist, origin, K=3, brent_future_dlog=None, first_lag=0):
    """Exogenous matrix for history and the 12 future weeks.
    brent_dlog at row t is already the previous week's crude change (known at t).
    Future crude: random walk (dlog = 0) unless a scenario path is supplied."""
    db = hist["dlog_brent"].fillna(0.0).values
    fut = np.zeros(H) if brent_future_dlog is None else np.asarray(brent_future_dlog, float)
    allb = np.concatenate([db, fut])
    lags = np.column_stack([np.concatenate([np.zeros(k), allb[:len(allb) - k]])
                            for k in range(first_lag, first_lag + CRUDE_LAGS)])
    fidx = future_index(origin)
    four = pd.concat([fourier(hist.index, K), fourier(fidx, K)]).values
    ev_hist = hist[EVENT_COLS].values
    ev_fut = np.repeat(ev_hist[-1:], H, axis=0)        # events persist at their current state
    ev = np.vstack([ev_hist, ev_fut])
    X = np.column_stack([lags, four, ev])
    n = len(hist)
    return X[:n], X[n:]


def arimax(hist, origin):
    y = np.log(hist["ca_gas"].values)
    X, Xf = arimax_design(hist, origin)
    f = AutoARIMA(seasonal=False).forecast(y=y, h=H, X=X, X_future=Xf, level=Z)
    return np.exp(f["mean"]), np.exp(f["lo-80"]), np.exp(f["hi-80"])


def arimax_lags1to4(hist, origin):
    """Robustness check: same as arimax but crude lags 1-4, i.e. without the most recent known week."""
    y = np.log(hist["ca_gas"].values)
    X, Xf = arimax_design(hist, origin, first_lag=1)
    f = AutoARIMA(seasonal=False).forecast(y=y, h=H, X=X, X_future=Xf, level=Z)
    return np.exp(f["mean"]), np.exp(f["lo-80"]), np.exp(f["hi-80"])


MODELS = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "ets": ets,
    "arima_fourier": arima_fourier,
    "arimax": arimax,
    "arimax_lags1to4": arimax_lags1to4,
}
