import sys
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbtools import NB, SETUP
n = NB()
n.md("""
# 03 · Backtest framework and conventional models

**Design (identical for every model, including the AI models in notebooks 05–06)**
- Rolling origin, expanding window: at each origin (every Monday from 30 Jun 2025 to 28 Sep 2026, 66 origins) fit on all data up to and including that Monday, forecast 1–12 weeks ahead.
- Scored horizons: 4 and 12 weeks. Targets after the last observed week are not scored, so h = 12 has fewer scored forecasts than h = 4.
- Log prices for fitting; forecasts and 80% intervals converted back to USD/gal before scoring.
- No future information: ARIMAX uses only crude prices known at the origin; future crude follows a random walk (no change). Event dummies are held at their state on the origin date.

Framework code: `src/common.py` (backtest loop, forecast file format, metrics, Diebold–Mariano). Model code: `src/models_conventional.py`.
""")
n.code(SETUP + """
from models_conventional import MODELS
ol = origins(df)
print(len(ol), "origins:", ol[0].date(), "→", ol[-1].date())""")

n.md("""
## Models
| model | what it does |
|---|---|
| `naive` | last observed price (random walk); intervals from historical h-week log changes |
| `seasonal_naive` | price 52 weeks before the target week |
| `ets` | exponential smoothing on log price; error/trend type and damping chosen by AICc |
| `arima_fourier` | auto-selected ARIMA on log price with 3 Fourier pairs for the 52.18-week season |
| `arimax` | `arima_fourier` + Brent weekly log changes (lags 0–3, where lag 0 = previous week's average) + COVID dummy + count of permanent CA refinery closures |
| `arimax_lags1to4` | robustness check: same as `arimax` but crude lags 1–4, dropping the most recent crude week known at the origin |
""")
n.code("""
import time
fcs = {}
for name, f in MODELS.items():
    t = time.time()
    fcs[name] = rolling_backtest(df, name, f, ol, verbose=False)
    print(f"{name:16s} done in {time.time() - t:6.1f}s")""")

n.md("## Quick accuracy check (full comparison with AI models is in notebook 07)")
n.code("""
fc = attach_actuals(pd.concat(fcs.values()), df)
tab = fc[fc.horizon.isin(SCORED_H)].groupby(["horizon", "model"]).apply(metrics).round(3)
naive_mae = tab.xs("naive", level="model")["MAE"]
tab["skill_vs_naive"] = (tab["MAE"] / tab.index.get_level_values("horizon").map(naive_mae)).round(3)
tab""")

n.md("## What did the models see at a few origins?")
n.code("""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, o in zip(axes, [ol[0], ol[len(ol) // 3], ol[2 * len(ol) // 3]]):
    hist = df.ca_gas.loc[o - pd.Timedelta(weeks=26):o + pd.Timedelta(weeks=12)]
    ax.plot(hist.index, hist.values, color=plots.INK, lw=1.2, label="actual")
    for m in ["naive", "ets", "arimax"]:
        g = fcs[m][fcs[m].origin_date == o]
        ax.plot(g.target_date, g.forecast, color=plots.MODEL_COLORS[m], label=plots.MODEL_LABELS[m])
    g = fcs["arimax"][fcs["arimax"].origin_date == o]
    ax.fill_between(g.target_date, g.lower80, g.upper80, color=plots.MODEL_COLORS["arimax"], alpha=0.12, lw=0)
    ax.axvline(o, color=plots.INK2, ls=":", lw=0.8); ax.set_title(f"origin {o.date()}", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
axes[0].set_ylabel("USD per gallon"); axes[0].legend(fontsize=8, loc="upper left")
fig""")
n.write(sys.argv[1])
