"""Benchmark on FI-2010 (download separately, see README). Chronological split of the train file;
pretrain SSL on train windows, fine-tune on official labels, report acc / macro-F1 / MCC.
IMPORTANT: use the un-normalised "DecPre" files (e.g. Train_Dst_NoAuction_DecPre_CF_7.txt). Our features take
logs/ratios of prices, which are invalid on z-scored prices.
Usage: python scripts/run_fi2010.py --file data/Train_Dst_NoAuction_DecPre_CF_7.txt --horizon-idx 0"""
import argparse
import numpy as np
from lobrep.data import load_fi2010
from lobrep.features import Scaler, lob_features
from lobrep.models import build_encoder
from lobrep.train import evaluate, finetune, pretrain, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True)
ap.add_argument("--horizon-idx", type=int, default=0, help="0..4 -> k=1,2,3,5,10")
ap.add_argument("--encoder", default="transformer")
ap.add_argument("--window", type=int, default=50)
ap.add_argument("--frac", type=float, default=1.0)
a = ap.parse_args()
set_seed(0)
d, y = load_fi2010(a.file, a.horizon_idx)
F = lob_features(d.raw)
ends = np.arange(a.window - 1, len(F), 2)
X = np.stack([F[e - a.window + 1:e + 1] for e in ends]); y = y[ends]
n = len(X); i1, i2 = int(.6 * n), int(.8 * n)
sc = Scaler().fit(X[:i1])
Xtr, Xva, Xte = sc(X[:i1 - 20]), sc(X[i1:i2 - 20]), sc(X[i2:])
ytr, yva, yte = y[:i1 - 20], y[i1:i2 - 20], y[i2:]
zero = lambda k: np.zeros(k, dtype=np.float32)   # FI-2010 task is movement only
for init in ("scratch", "ssl"):
    enc = build_encoder(a.encoder, Xtr.shape[-1], 64)
    if init == "ssl":
        pretrain(enc, Xtr, epochs=10)
    k = int(len(Xtr) * a.frac)
    m = finetune(enc, Xtr[:k], ytr[:k], zero(k), Xva, (yva, zero(len(yva))), epochs=15, vol_w=0.0)
    r = evaluate(m, Xte, yte, zero(len(yte)))
    print(f"{a.encoder}/{init}  acc={r['acc']:.4f}  f1={r['f1']:.4f}  mcc={r['mcc']:.4f}")
