import sys
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbtools import NB, SETUP
n = NB()
n.md("""
# 07 · Evaluation

1. Accuracy of every model on all 66 origins (MAE, RMSE, MAPE, 80% coverage, skill vs naive)
2. Diebold–Mariano tests (HLN-corrected, Newey–West variance with h−1 lags) vs naive and vs the best conventional model
3. Forecast combination: best conventional + best AI model
4. Robustness: ARIMAX without the most recent crude week
""")
n.code(SETUP)
n.md("## 1. Accuracy, all 66 origins")
n.code("""
fc = attach_actuals(load_forecasts(), df)
full_models = [m for m in fc.model.unique() if not m.startswith("llm_")]
FULL = fc[fc.model.isin(full_models)]
def table(d):
    t = d[d.horizon.isin(SCORED_H)].groupby(["horizon", "model"]).apply(metrics)
    t["skill_vs_naive"] = t["MAE"] / t.index.get_level_values("horizon").map(t["MAE"].xs("naive", level="model"))
    return t
tab = table(FULL)
m4 = tab.xs(4, level="horizon")
conv = [m for m in m4.index if m in ("ets", "arima_fourier", "arimax")]
ai = [m for m in m4.index if m in ("chronos_bolt", "moirai2")]
best_conv, best_ai = m4.loc[conv, "MAE"].idxmin(), m4.loc[ai, "MAE"].idxmin()
print("best conventional:", best_conv, "| best AI (full coverage):", best_ai)""")
n.md("## 2. Forecast combination")
n.code("""
combo = (FULL[FULL.model.isin([best_conv, best_ai])]
         .groupby(["origin_date", "horizon", "target_date"], as_index=False)[["forecast", "lower80", "upper80"]].mean())
combo.insert(0, "model", "combo")
combo[FC_COLS].to_csv(FC_DIR / "combo.csv", index=False, date_format="%Y-%m-%d")
fc = attach_actuals(load_forecasts(), df)
FULL = fc[~fc.model.str.startswith("llm_")]
tab = table(FULL).round(3).sort_values(["horizon", "MAE"])
tab.to_csv(RES_DIR / "metrics.csv"); tab""")
n.md("## 3. Diebold–Mariano tests")
n.code("""
rows = []
for h in SCORED_H:
    e = fc[fc.horizon == h].pivot(index="origin_date", columns="model", values="error")
    for m in e.columns:
        for bench in ["naive", best_conv]:
            if m != bench:
                pair = e[[m, bench]].dropna()
                if len(pair) > h + 2:
                    s, p = dm_test(pair[m].values, pair[bench].values, h)
                    rows.append({"horizon": h, "model": m, "vs": bench, "n": len(pair), "DM": s, "p": p})
dm = pd.DataFrame(rows).round(3); dm.to_csv(RES_DIR / "dm_tests.csv", index=False)
dm[dm.p < 0.10]""")
n.code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
order = ["naive", "seasonal_naive", "ets", "arima_fourier", "arimax", "arimax_lags1to4", "chronos_bolt", "moirai2", "combo"]
for ax, h in zip(axes, SCORED_H):
    d = tab.xs(h, level="horizon").reindex([m for m in order if m in tab.xs(h, level="horizon").index])
    d = d.drop(index="seasonal_naive")
    cols = [plots.MODEL_COLORS.get(m, "#b5b4ae") for m in d.index]
    ax.barh([plots.MODEL_LABELS.get(m, m) for m in d.index], d["MAE"], color=cols, height=0.6)
    ax.axvline(d.loc["naive", "MAE"], color=plots.INK2, ls=":", lw=1)
    for y, v in enumerate(d["MAE"]): ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=8, color=plots.INK2)
    ax.set_title(f"{h} weeks ahead: MAE (USD/gal)", fontsize=11); ax.invert_yaxis()
plots.save(fig, FIG_DIR / "fig09_mae.png"); fig""")
n.md("## 4. Robustness: does the most recent crude week matter?")
n.code("""
r = tab.loc[(slice(None), ["arima_fourier", "arimax", "arimax_lags1to4"]), ["MAE", "RMSE", "cover80", "skill_vs_naive"]]
r""")
n.md("""
Dropping the most recent crude week known at the origin (lags 1–4 instead of 0–3) removes ARIMAX's advantage over ARIMA: the latest week carries the strongest crude signal (cross-correlation 0.34, notebook 02).

**Diebold–Mariano variance.** Newey–West (Bartlett) weights with h−1 lags are used. With unweighted autocovariances, the long-run variance can collapse toward zero at h = 12 with ~55 overlapping forecasts (e.g. Moirai 2.0 vs naive would give DM ≈ 10.8), so Bartlett weights are the safer choice.
""")
n.write(sys.argv[1])
