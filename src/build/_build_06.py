import sys
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbtools import NB, SETUP
n = NB()
n.md("""
# 06 · LLM forecasting experiments (Claude)

**Design**
- 16 forecast dates (every 4th test origin, 30 Jun 2025 → 24 Aug 2026), 2 independent runs per prompt, averaged. The proposal planned 3 runs; this was cut to 2 to save time.
- **Experiment 1 – anonymized:** the model sees only the last 156 weekly values, rescaled so the latest value = 100. No name, units or dates. Forecasts are converted back to USD/gal by multiplying by the last price / 100.
- **Experiment 2 – full context:** the model is told the series is California regular gasoline, sees dated prices and a *market brief built only from data available on the forecast date* (Brent, U.S. price, CA–U.S. premium, West Coast stocks and refinery inputs, events to date, season). No news text was used, so the prompt itself cannot leak hindsight.
- Every forecast asks for a 12-week median plus 10th/90th percentiles in JSON.

**How the runs were made:** each forecast was a fresh Claude session (Claude Opus 5.5, `claude-opus-5-5`) started from the Claude app on 14 Oct 2026, told to read exactly one prompt file and use no other tools. All prompts are in `results/llm/prompts/`, all raw replies in `results/llm/responses.jsonl`. The prompts can be rerun through the Anthropic API with any model (see `llm_api_run()` at the end).

**Look-ahead caveat:** Claude Opus 5.5's training data runs to about mid-2026, which overlaps the test period. The full-context experiment can therefore draw on remembered prices/events for origins before ~July 2026; the anonymized experiment is the clean test of pattern-recognition skill.
""")
n.code(SETUP + """
import json
from llm_prompts import parse, LLM_DIR, PROMPT_DIR, llm_origins
meta = pd.read_csv(LLM_DIR / "prompt_index.csv", parse_dates=["origin_date"]).set_index("prompt_id")
print(len(meta), "prompts;", meta.origin_date.nunique(), "forecast dates")""")
n.md("## Example prompts")
n.code("""print((PROMPT_DIR / "anon_20260309.txt").read_text()[:900], "...\\n")
print((PROMPT_DIR / "context_20260309.txt").read_text()[-2600:])""")
n.md("## Parse logged replies and build forecasts")
n.code("""
rows = [json.loads(l) for l in open(LLM_DIR / "responses.jsonl")]
recs, issues = [], []
for r in rows:
    o, why = parse(r["raw"])
    try:
        json.loads(r["raw"])
    except Exception:
        issues.append((r["prompt_id"], r["rep"], "malformed JSON (stray brace) - arrays recovered by parser"))
    if not why:
        issues.append((r["prompt_id"], r["rep"], "no rationale returned"))
    if (o["p10"] > o["median"]).any() or (o["p90"] < o["median"]).any():
        issues.append((r["prompt_id"], r["rep"], "quantiles out of order"))
    recs.append({"prompt_id": r["prompt_id"], "rep": r["rep"], **o, "rationale": why})
issues.append(("context_20260629", 2, "session made 2 tool calls instead of 1 (forecast kept; flagged)"))
print(pd.Series({"replies": len(recs), "prompts": len({x['prompt_id'] for x in recs})}))
pd.DataFrame(issues, columns=["prompt_id", "rep", "note"])""")
n.code("""
by = {}
for x in recs:
    by.setdefault(x["prompt_id"], []).append(x)
for exp in ["anon", "context"]:
    out = []
    for pid, lst in sorted(by.items()):
        if not pid.startswith(exp):
            continue
        m = meta.loc[pid]
        s = m.scale / 100 if exp == "anon" else 1.0
        avg = {k: np.mean([x[k] for x in lst], axis=0) * s for k in ("median", "p10", "p90")}
        out.append(make_fc(f"llm_{exp}", m.origin_date, avg["median"], avg["p10"], avg["p90"]))
    pd.concat(out).sort_values(["origin_date", "horizon"]).to_csv(FC_DIR / f"llm_{exp}.csv", index=False, date_format="%Y-%m-%d")
print("wrote forecasts/llm_anon.csv and forecasts/llm_context.csv")""")
n.md("## Run-to-run consistency and what the model said")
n.code("""
sd = []
for pid, lst in by.items():
    if len(lst) == 2:
        m = meta.loc[pid]; s = m.scale / 100 if pid.startswith("anon") else 1.0
        sd.append({"experiment": pid.split("_")[0], "abs diff between runs, h=4 ($)": abs(lst[0]["median"][3] - lst[1]["median"][3]) * s,
                   "abs diff, h=12 ($)": abs(lst[0]["median"][11] - lst[1]["median"][11]) * s})
display(pd.DataFrame(sd).groupby("experiment").mean().round(3))
guessed = [x for x in recs if x["prompt_id"].startswith("anon") and any(w in x["rationale"].lower() for w in ["gasoline", "fuel", "petrol"])]
print(f"Anonymized runs whose rationale names fuel/gasoline anyway: {len(guessed)} of {sum(p.startswith('anon') for p in by for _ in by[p])}")
for x in guessed:
    print(" -", x["prompt_id"], ":", x["rationale"][:160])""")
n.md("## Accuracy on the 16 LLM dates (all models scored on the same dates)")
n.code("""
fc = attach_actuals(load_forecasts(), df)
dates = meta.origin_date.unique()
same = fc[fc.origin_date.isin(dates) & fc.horizon.isin(SCORED_H)]
t = same.groupby(["horizon", "model"]).apply(metrics)[["n", "MAE", "RMSE", "MAPE", "cover80"]]
t["skill_vs_naive"] = t["MAE"] / t.index.get_level_values("horizon").map(t["MAE"].xs("naive", level="model"))
t = t.round(3).sort_values(["horizon", "MAE"])
t.to_csv(RES_DIR / "llm_same_dates.csv"); t""")
n.code("""
gap = t.xs("llm_anon", level="model")["MAE"] - t.xs("llm_context", level="model")["MAE"]
print("anon MAE - context MAE (positive = background knowledge / context helps):"); print(gap.round(3).to_string())
pre = same[same.origin_date < "2026-07-01"]; post = same[same.origin_date >= "2026-07-01"]
for lbl, d in [("origins before Jul 2026 (inside Claude's training window)", pre), ("origins Jul 2026+ (after training data)", post)]:
    print("\\n" + lbl)
    print(d[d.model.isin(["llm_anon", "llm_context", "naive", "arimax"])].groupby(["horizon", "model"]).error.apply(lambda e: e.abs().mean()).unstack(0).round(3).to_string())""")
n.md("""
## Re-running through the API (optional)
The cell below re-runs every prompt through the Anthropic API so anyone can reproduce or extend the experiment with another model (needs `pip install anthropic` and `ANTHROPIC_API_KEY`). It is not executed here.
""")
n.code("""
def llm_api_run(model="claude-opus-5-5", reps=2, out=LLM_DIR / "responses_api.jsonl"):
    import anthropic, datetime
    client = anthropic.Anthropic()
    with open(out, "a") as f:
        for p in sorted(PROMPT_DIR.glob("*.txt")):
            text = p.read_text()
            for rep in range(1, reps + 1):
                msg = client.messages.create(model=model, max_tokens=800, messages=[{"role": "user", "content": text}])
                f.write(json.dumps({"prompt_id": p.stem, "rep": rep, "model": model,
                                    "logged_at": datetime.datetime.now(datetime.UTC).isoformat(),
                                    "raw": msg.content[0].text}) + "\\n")
# llm_api_run()   # uncomment to run""")
n.write(sys.argv[1])
