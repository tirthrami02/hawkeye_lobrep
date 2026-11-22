"""Main study: does SSL pretraining help? (encoders x label fraction x init x seeds)

EXPERIMENT DESIGN (read this first)
  For each of 3 random seeds:
    1. Simulate instrument A (source) and instrument B (1.8x more volatile, different regime
       persistence = distribution-shifted test set).
    2. Window A, split chronologically into train / val / test (with embargo).
    3. For each encoder (LSTM, CNN, Transformer):
         a. SSL-pretrain on the UNLABELLED train windows only.
         b. Cluster test embeddings (SSL vs random-init vs raw stats) -> regime quality.
         c. For label fractions 100%, 10%, 2% and init in {scratch, ssl}: fine-tune, then evaluate
            on the in-distribution test set AND the shifted instrument B.
  Outputs go to results/: raw CSVs, mean/std summary, regime summary, label_efficiency.png.
Usage: python scripts/run_experiments.py [--quick] [--seeds N] [--mlflow]"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd

from lobrep.data import simulate_lob
from lobrep.features import Scaler, make_windows, time_split
from lobrep.models import build_encoder
from lobrep.regimes import cluster_eval
from lobrep.train import embed, evaluate, finetune, pretrain, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--steps", type=int, default=30_000)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--dim", type=int, default=32)
ap.add_argument("--out", default="results")
ap.add_argument("--mlflow", action="store_true")
args = ap.parse_args()
if args.quick:
    args.steps, args.seeds = 8_000, 1
EP_PRE, EP_FT = (3, 4) if args.quick else (8, 12)
out = Path(args.out); out.mkdir(exist_ok=True)

mlf = None
if args.mlflow:
    import mlflow as mlf
    mlf.set_experiment("lob-ssl")

rows, cluster_rows = [], []
t0 = time.time()
for seed in range(args.seeds):
    set_seed(seed)
    A = simulate_lob(args.steps, seed=seed)                          # source instrument
    B = simulate_lob(args.steps // 2, seed=100 + seed, vol_scale=1.8, stay_prob=0.995)  # shifted
    wA, thr = make_windows(A)
    tr, va, te = time_split(wA)
    wB, _ = make_windows(B, thresh=thr)                              # same label threshold => real shift
    sc = Scaler().fit(tr.X)
    Xtr, Xva, Xte, XB = sc(tr.X), sc(va.X), sc(te.X), sc(wB.X)
    vmu, vsd = tr.y_vol.mean(), tr.y_vol.std() + 1e-8           # standardise vol target (train stats only)
    for w_ in (tr, va, te, wB):
        w_.y_vol = ((w_.y_vol - vmu) / vsd).astype(np.float32)
    F = Xtr.shape[-1]
    for enc_name in ["lstm", "cnn", "transformer"]:
        # SSL pretrain once per (encoder, seed) on UNLABELLED train windows only
        set_seed(seed)
        enc_ssl = build_encoder(enc_name, F, args.dim)
        hist = pretrain(enc_ssl, Xtr, epochs=EP_PRE)
        # Save pretrained weights so every fine-tuning run below starts from the SAME checkpoint.
        ssl_state = {k: v.clone() for k, v in enc_ssl.state_dict().items()}
        # regime clustering quality: SSL vs random-init vs raw-feature statistics
        set_seed(seed)
        enc_rand = build_encoder(enc_name, F, args.dim)
        raw_stats = np.concatenate([Xte.mean(1), Xte.std(1)], 1)
        for tag, Z in [("ssl", embed(enc_ssl, Xte)), ("random_init", embed(enc_rand, Xte)),
                       ("raw_stats", raw_stats)]:
            cluster_rows.append(dict(seed=seed, encoder=enc_name, repr=tag, **cluster_eval(Z, te.regime, seed=seed)))
        # Label-efficiency sweep: use only the FIRST `frac` of the (chronological) training labels.
        for frac in [1.0, 0.1, 0.02]:
            n = max(32, int(len(Xtr) * frac))
            for init in ["scratch", "ssl"]:
                set_seed(seed)  # same seed => scratch and ssl differ ONLY in initial weights
                enc = build_encoder(enc_name, F, args.dim)
                if init == "ssl":
                    enc.load_state_dict(ssl_state)  # start from pretrained weights instead of random
                m = finetune(enc, Xtr[:n], tr.y_move[:n], tr.y_vol[:n], Xva, (va.y_move, va.y_vol),
                             epochs=EP_FT)
                for split, X, w in [("in_dist", Xte, te), ("shifted", XB, wB)]:
                    r = evaluate(m, X, w.y_move, w.y_vol)
                    rows.append(dict(seed=seed, encoder=enc_name, label_frac=frac, init=init, split=split, **r))
                    if mlf:
                        with mlf.start_run():
                            mlf.log_params(dict(seed=seed, encoder=enc_name, frac=frac, init=init, split=split))
                            mlf.log_metrics(r)
        print(f"seed {seed} {enc_name} done  [{time.time()-t0:.0f}s]  ssl loss {hist[-1]}", flush=True)

df = pd.DataFrame(rows); cd = pd.DataFrame(cluster_rows)
df.to_csv(out / "supervised_results.csv", index=False); cd.to_csv(out / "regime_clustering.csv", index=False)
summ = df.groupby(["split", "encoder", "label_frac", "init"])[["acc", "f1", "mcc", "mae", "rmse"]].agg(["mean", "std"]).round(4)
summ.to_csv(out / "summary.csv")
csumm = cd.groupby(["encoder", "repr"]).mean(numeric_only=True).drop(columns="seed").round(3)
csumm.to_csv(out / "regime_summary.csv")
print(summ.to_string()); print(csumm.to_string())

# plots
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, split in zip(axes, ["in_dist", "shifted"]):
    for (enc, init), g in df[df.split == split].groupby(["encoder", "init"]):
        s = g.groupby("label_frac").mcc.mean()
        ax.plot(s.index, s.values, marker="o", ls="-" if init == "ssl" else "--", label=f"{enc}/{init}")
    ax.set_xscale("log"); ax.set_title(split); ax.set_xlabel("fraction of labels used"); ax.grid(alpha=.3)
axes[0].set_ylabel("MCC (movement)"); axes[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(out / "label_efficiency.png", dpi=150)
