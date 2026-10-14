"""Append raw LLM responses to results/llm/responses.jsonl.
Usage: python log_resp.py <<'EOF'
prompt_id|rep|<raw response text on one line>
...
EOF"""
import sys, json, datetime
from llm_prompts import RESP_FILE, parse

MODEL = "Claude Opus 5.5 (claude-opus-5-5), fresh session per forecast, run 2026-10-14"
n = 0
with open(RESP_FILE, "a") as f:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        pid, rep, raw = line.split("|", 2)
        parse(raw)   # validate before logging
        f.write(json.dumps({"prompt_id": pid, "rep": int(rep), "model": MODEL,
                            "logged_at": datetime.datetime.utcnow().isoformat(), "raw": raw}) + "\n")
        n += 1
print("logged", n)
