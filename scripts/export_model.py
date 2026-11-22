"""Train a transformer (SSL-pretrained) on simulated data and export it for the API."""

from pathlib import Path
import torch
from lobrep.data import simulate_lob
from lobrep.features import Scaler, make_windows, time_split
from lobrep.models import build_encoder
from lobrep.train import finetune, pretrain, set_seed

set_seed(0)
w, _ = make_windows(simulate_lob(30_000, seed=0))
tr, va, _ = time_split(w)
sc = Scaler().fit(tr.X)
enc = build_encoder("transformer", tr.X.shape[-1], 32)
pretrain(enc, sc(tr.X), epochs=8)
m = finetune(enc, sc(tr.X), tr.y_move, tr.y_vol, sc(va.X), (va.y_move, va.y_vol))
Path("artifacts").mkdir(exist_ok=True)
torch.save(
    dict(state=m.state_dict(), mu=torch.from_numpy(sc.mu), sd=torch.from_numpy(sc.sd)),
    "artifacts/model.pt",
)
print("saved artifacts/model.pt")
