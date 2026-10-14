"""Zero-shot time-series foundation models (no retraining). Weights are loaded from local folders in models/.
Both take the last CONTEXT weekly prices (USD/gal) and return the median and 10%/90% quantiles (80% interval)."""
import os
import warnings
import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.filterwarnings("ignore")
from common import ROOT, H

CONTEXT = 512
torch.set_num_threads(max(1, os.cpu_count() or 1))

_cache = {}


def _chronos():
    if "bolt" not in _cache:
        from chronos import BaseChronosPipeline
        _cache["bolt"] = BaseChronosPipeline.from_pretrained(str(ROOT / "models/chronos-bolt-small"),
                                                             device_map="cpu", torch_dtype=torch.float32)
    return _cache["bolt"]


def _moirai2():
    if "moirai2" not in _cache:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        _cache["moirai2"] = Moirai2Forecast(module=Moirai2Module.from_pretrained(str(ROOT / "models/moirai-2.0-R-small")),
                                            prediction_length=H, context_length=CONTEXT, target_dim=1,
                                            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    return _cache["moirai2"]


def chronos_bolt(hist, origin):
    y = torch.tensor(hist["ca_gas"].values[-CONTEXT:], dtype=torch.float32)
    q, _ = _chronos().predict_quantiles(y, prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
    q = q[0].numpy()
    return q[:, 1], q[:, 0], q[:, 2]


def moirai2(hist, origin):
    y = hist["ca_gas"].values[-CONTEXT:]
    x = torch.tensor(y, dtype=torch.float32).reshape(1, -1, 1)
    with torch.no_grad():
        out = _moirai2()(past_target=x, past_observed_target=torch.ones_like(x, dtype=torch.bool),
                         past_is_pad=torch.zeros(1, len(y), dtype=torch.bool))
    q = out[0].numpy()          # [9 quantiles (0.1..0.9), H]
    return q[4], q[0], q[8]


MODELS = {"chronos_bolt": chronos_bolt, "moirai2": moirai2}

MODEL_INFO = {
    "chronos_bolt": {"name": "Chronos-Bolt Small (Amazon)", "params": "48M", "weights": "amazon/chronos-bolt-small",
                     "released": "Nov 2024", "context": CONTEXT},
    "moirai2": {"name": "Moirai 2.0 R Small (Salesforce)", "params": "11M", "weights": "Salesforce/moirai-2.0-R-small",
                "released": "Aug 2025", "context": CONTEXT},
}
