"""FastAPI inference service. Run: uvicorn lobrep.api:app
Endpoints: GET /health (is a trained model loaded?) and POST /predict (send one 50x8 feature window,
get back P(down/flat/up) and a log-volatility forecast).
Loads artifacts/model.pt (from scripts/export_model.py) if present, else an UNTRAINED demo model whose
outputs are meaningless (/health reports trained=false). The 8 features must be the same as
lobrep.features.lob_features produces; the stored mean/std standardise them the way training did."""

from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .models import Predictor, build_encoder

N_FEATURES, WINDOW = 8, 50
app = FastAPI(title="lobrep", version="0.1.0")
_model = Predictor(build_encoder("transformer", N_FEATURES, 32))
_stats = None
ckpt = Path("artifacts/model.pt")
if ckpt.exists():
    blob = torch.load(ckpt, map_location="cpu")
    _model.load_state_dict(blob["state"])
    _stats = (blob["mu"].numpy(), blob["sd"].numpy())
_model.eval()


class Req(BaseModel):
    window: list[list[float]]  # (50, 8) engineered feature window


@app.get("/health")
def health():
    return {"status": "ok", "trained": _stats is not None}


@app.post("/predict")
def predict(r: Req):
    x = np.asarray(r.window, dtype=np.float32)
    if x.shape != (WINDOW, N_FEATURES):
        raise HTTPException(
            422, f"expected shape ({WINDOW}, {N_FEATURES}), got {x.shape}"
        )
    if _stats is not None:
        x = (x - _stats[0]) / _stats[1]
    with torch.no_grad():
        lm, lv = _model(torch.from_numpy(x)[None])
    p = torch.softmax(lm, -1)[0].tolist()
    return {
        "p_down": p[0],
        "p_flat": p[1],
        "p_up": p[2],
        "log_vol_forecast": float(lv[0]),
    }
