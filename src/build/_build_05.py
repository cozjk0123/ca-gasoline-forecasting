import sys
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbtools import NB, SETUP
n = NB()
n.md("""
# 05 · Time-series foundation models (zero-shot)

Two pretrained forecasting models run **without any retraining** inside the same rolling-origin loop as the conventional models:

- **Chronos-Bolt Small** (Amazon) — a T5-style model trained on ~100B time-series observations; returns quantiles directly.
- **Moirai 2.0 R Small** (Salesforce) — a decoder-only quantile forecaster. Used in place of Google TimesFM: the TimesFM checkpoint (~800 MB) was too large to move into the project environment; Moirai 2.0 is a comparable zero-shot foundation model.

Each sees the last **512 weekly prices** at the origin (≈10 years), nothing else — no crude, no events, no dates. Median = point forecast; 10%/90% quantiles = 80% interval.

**Look-ahead check:** Chronos-Bolt was released in Nov 2024, before the test period. Moirai 2.0 was released in Aug 2025, about five weeks into the test period (repo last updated Jan 2026). Its pretraining corpus is built from public benchmark collections, so it is unlikely to contain this EIA series' test-period values, but its first few test weeks carry a small leakage risk, which we flag in the report. Weights were downloaded from Hugging Face into `models/` (files listed in the README).
""")
n.code(SETUP + """
import time
from models_foundation import MODELS, MODEL_INFO
pd.DataFrame(MODEL_INFO).T""")
n.code("""
ol = origins(df)
fcs = {}
for name, f in MODELS.items():
    t = time.time()
    fcs[name] = rolling_backtest(df, name, f, ol, verbose=False)
    print(f"{name:14s} done in {time.time() - t:6.1f}s")""")
n.code("""
fc = attach_actuals(pd.concat(list(fcs.values()) + [pd.read_csv(FC_DIR / "naive.csv", parse_dates=["origin_date", "target_date"])]), df)
tab = fc[fc.horizon.isin(SCORED_H)].groupby(["horizon", "model"]).apply(metrics).round(3)
tab""")
n.write(sys.argv[1])
