"""Shared helpers: data loading, rolling-origin backtest, forecast file format, scoring."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FC_DIR = ROOT / "forecasts"
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "results" / "figures"
for d in (FC_DIR, RES_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

TEST_START = pd.Timestamp("2025-07-01")
H = 12
SCORED_H = (4, 12)
FC_COLS = ["model", "origin_date", "horizon", "target_date", "forecast", "lower80", "upper80"]


def load_weekly():
    df = pd.read_csv(DATA / "weekly_data.csv", parse_dates=["date"]).set_index("date")
    return df


def origins(df):
    """Every Monday from the last training week to the second-to-last week:
    the first 1-step forecast lands on the first test week."""
    idx = df.index
    first = idx[idx < TEST_START][-1]
    return idx[(idx >= first) & (idx < idx[-1])]


def make_fc(model, origin, point, lo=None, hi=None):
    h = np.arange(1, len(point) + 1)
    return pd.DataFrame({
        "model": model,
        "origin_date": origin,
        "horizon": h,
        "target_date": origin + pd.to_timedelta(7 * h, unit="D"),
        "forecast": np.asarray(point, float),
        "lower80": np.nan if lo is None else np.asarray(lo, float),
        "upper80": np.nan if hi is None else np.asarray(hi, float),
    })


def rolling_backtest(df, model_name, fit_predict, origin_list=None, verbose=True):
    """fit_predict(history_df, origin) -> (point, lo, hi) arrays of length H in USD/gal.
    history_df contains only rows dated <= origin (no future information)."""
    out = []
    ol = origins(df) if origin_list is None else origin_list
    for i, o in enumerate(ol):
        hist = df.loc[:o]
        p, lo, hi = fit_predict(hist, o)
        out.append(make_fc(model_name, o, p, lo, hi))
        if verbose and (i % 10 == 0 or i == len(ol) - 1):
            print(f"{model_name}: {i + 1}/{len(ol)} origins", flush=True)
    fc = pd.concat(out, ignore_index=True)
    fc.to_csv(FC_DIR / f"{model_name}.csv", index=False, date_format="%Y-%m-%d")
    return fc


def load_forecasts(names=None):
    files = sorted(FC_DIR.glob("*.csv"))
    fcs = [pd.read_csv(f, parse_dates=["origin_date", "target_date"]) for f in files
           if names is None or f.stem in names]
    return pd.concat(fcs, ignore_index=True)


def attach_actuals(fc, df):
    act = df["ca_gas"].rename("actual")
    fc = fc.merge(act, left_on="target_date", right_index=True, how="left")
    fc["error"] = fc["actual"] - fc["forecast"]
    return fc.dropna(subset=["actual"])


def metrics(g):
    e = g["error"]
    cov = ((g["actual"] >= g["lower80"]) & (g["actual"] <= g["upper80"])).mean() if g["lower80"].notna().any() else np.nan
    return pd.Series({
        "n": len(g),
        "MAE": e.abs().mean(),
        "RMSE": np.sqrt((e ** 2).mean()),
        "MAPE": (e.abs() / g["actual"]).mean() * 100,
        "bias": e.mean(),
        "cover80": cov,
    })


def dm_test(e1, e2, h, power=2):
    """Diebold-Mariano with Harvey-Leybourne-Newbold small-sample correction.
    Long-run variance: Newey-West (Bartlett weights) with h-1 lags, because h-step forecasts overlap.
    Bartlett weights keep the variance positive; the unweighted (rectangular) version can collapse
    towards zero at h = 12 with ~55 forecasts and produce absurd statistics.
    Negative stat => model 1 more accurate. Two-sided p-value from t(n-1)."""
    e1, e2 = np.asarray(e1), np.asarray(e2)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))
    dm = dbar / np.sqrt(var / n)
    k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * k
    p = 2 * stats.t.sf(np.abs(stat), df=n - 1)
    return stat, p


def fourier(index, K=3, period=52.1775):
    t = (index.dayofyear.values + 365.25 * (index.year.values - 2000)) / 7.0
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * t / period)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(cols, index=index)


def future_index(origin, h=H):
    return pd.DatetimeIndex([origin + pd.Timedelta(days=7 * i) for i in range(1, h + 1)])
