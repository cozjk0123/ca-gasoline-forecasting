"""Prompt construction and response parsing for the two LLM experiments.

Experiment 1 (anonymized): last 156 weekly values rescaled to an index (last value = 100). No name, units or dates.
Experiment 2 (full context): series named, dated, plus a market brief built mechanically from data known at the
origin (crude, U.S. price, West Coast stocks/inputs, active events, calendar). No news text is used, so no hindsight
can leak in through the prompt; the model's own training knowledge is the remaining leakage channel.
"""
import json
import re
import numpy as np
import pandas as pd
from common import ROOT, H, origins

LLM_DIR = ROOT / "results" / "llm"
PROMPT_DIR = LLM_DIR / "prompts"
RESP_FILE = LLM_DIR / "responses.jsonl"
for d in (LLM_DIR, PROMPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_CONTEXT = 156
REPS = 3
SYSTEM = ("You are a forecasting model. Use only the information in this message. "
          "Do not use tools, browse, or look anything up. Reply with JSON only.")

JSON_SPEC = (
    'Return exactly this JSON and nothing else:\n'
    '{"median": [12 numbers], "p10": [12 numbers], "p90": [12 numbers], "rationale": "one sentence"}\n'
    'Element i is the forecast for week i ahead (i = 1..12). p10 and p90 are the 10th and 90th percentiles '
    '(an 80% interval). Ensure p10 <= median <= p90 for every week.'
)


def llm_origins(df):
    ol = origins(df)
    return ol[0:64:4]          # every 4th test origin -> 16 forecast dates


def prompt_anon(df, origin):
    y = df.loc[:origin, "ca_gas"].values[-N_CONTEXT:]
    idx = np.round(100 * y / y[-1], 2)
    body = (
        "Below are 156 consecutive weekly observations of a time series, oldest first. "
        "Values are an index scaled so the most recent observation equals 100.\n\n"
        + ", ".join(f"{v:.2f}" for v in idx)
        + "\n\nForecast the next 12 weekly values of this index.\n\n" + JSON_SPEC
    )
    return body, float(y[-1])


def _events_active(origin):
    ev = pd.read_csv(ROOT / "data" / "events.csv", parse_dates=["start", "end"])
    ev = ev[ev.start <= origin].sort_values("start")
    lines = []
    for _, e in ev.iterrows():
        if e.kind == "step":
            lines.append(f"- {e.start:%b %Y}: {e.description} (permanent capacity loss)")
        elif pd.isna(e.end) or e.end >= origin - pd.Timedelta(weeks=26):
            status = "ongoing" if (pd.isna(e.end) or e.end >= origin) else f"ended {min(e.end, origin):%b %Y}"
            lines.append(f"- {e.start:%b %Y}: {e.description} ({status})")
    return lines


def market_brief(df, origin):
    h = df.loc[:origin]
    last = h.iloc[-1]
    b = h["brent"].iloc[-13:]
    yr = h.index.year
    same_week = h[(h.index.isocalendar().week.values == origin.isocalendar().week)]
    st5 = same_week["p5_stocks"].iloc[-6:-1].mean()
    in5 = same_week["p5_inputs"].iloc[-6:-1].mean()
    month = origin.month
    season = "summer-grade (Apr-Oct)" if 4 <= month <= 10 else "winter-grade (Nov-Mar)"
    lines = [
        f"Market brief as of {origin:%A %d %B %Y} (only information available on that date):",
        f"- Brent crude, average of the previous week: ${last.brent:.2f}/bbl; 12 weeks earlier: ${b.iloc[0]:.2f}/bbl; "
        f"12-week range ${b.min():.2f}-${b.max():.2f}.",
        f"- Brent weekly averages, last 13 weeks (oldest first): " + ", ".join(f"{v:.1f}" for v in b.values) + ".",
        f"- U.S. average regular gasoline this week: ${last.us_gas:.3f}/gal; California premium over U.S.: ${last.ca_us_spread:.2f}/gal "
        f"(52-week average premium ${h.ca_us_spread.iloc[-52:].mean():.2f}).",
        f"- West Coast (PADD 5) gasoline stocks, latest released week: {last.p5_stocks:,.0f} thousand bbl "
        f"({100 * (last.p5_stocks / st5 - 1):+.1f}% vs. same week, prior 5-year average).",
        f"- West Coast refinery gross inputs, latest released week: {last.p5_inputs:,.0f} thousand bbl/day "
        f"({100 * (last.p5_inputs / in5 - 1):+.1f}% vs. same week, prior 5-year average).",
        f"- Season: California is currently in the {season} gasoline season. Summer-grade season runs April-October; "
        "the state gasoline excise tax adjusts each July 1.",
        "- California refinery and market events to date:",
    ] + _events_active(origin)
    return "\n".join(lines)


def prompt_context(df, origin):
    s = df.loc[:origin, "ca_gas"].iloc[-N_CONTEXT:]
    series = "\n".join(f"{d:%Y-%m-%d},{v:.3f}" for d, v in s.items())
    body = (
        "Series: California regular gasoline, all formulations, average retail price in U.S. dollars per gallon "
        "(U.S. Energy Information Administration weekly survey, collected each Monday).\n"
        f"Forecast origin: {origin:%Y-%m-%d}. The last 156 weekly observations (date,price) are:\n\n{series}\n\n"
        + market_brief(df, origin)
        + f"\n\nForecast the California regular gasoline price (USD/gal) for each of the next 12 weeks "
        f"(the Mondays from {origin + pd.Timedelta(days=7):%Y-%m-%d} to {origin + pd.Timedelta(days=84):%Y-%m-%d}).\n\n"
        + JSON_SPEC
    )
    return body, 1.0


def build_all(df):
    rows = []
    for o in llm_origins(df):
        for exp, fn in (("anon", prompt_anon), ("context", prompt_context)):
            body, scale = fn(df, o)
            pid = f"{exp}_{o:%Y%m%d}"
            (PROMPT_DIR / f"{pid}.txt").write_text(SYSTEM + "\n\n" + body)
            rows.append({"prompt_id": pid, "experiment": exp, "origin_date": o, "scale": scale})
    meta = pd.DataFrame(rows)
    meta.to_csv(LLM_DIR / "prompt_index.csv", index=False)
    return meta


def parse(text):
    """Extract the three arrays even if the JSON is slightly malformed (e.g. a stray brace)."""
    out = {}
    for k in ("median", "p10", "p90"):
        m = re.search(r'"%s"\s*:\s*\[([^\]]*)\]' % k, text)
        out[k] = np.asarray([float(x) for x in m.group(1).split(",")], float)[:H]
    assert all(len(v) == H for v in out.values())
    r = re.search(r'"rationale"\s*:\s*"(.*?)"\s*\}?\s*$', text, re.S)
    return out, (r.group(1) if r else "")
