"""Training loops: SSL pretraining, supervised fine-tuning (with early stopping), prediction, metrics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from .models import Predictor, SSLModel


def set_seed(s: int):
    """Seed numpy and torch so a run is reproducible."""
    np.random.seed(s)
    torch.manual_seed(s)


def _batches(n, bs, shuffle=True):
    """Yield index arrays of size `bs` covering range(n), optionally shuffled."""
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def pretrain(encoder, X: np.ndarray, epochs=8, bs=128, lr=2e-3, lam=0.5):
    """SELF-SUPERVISED pretraining. Uses only the input windows X (no labels at all).
    Modifies `encoder` in place. Returns per-epoch [masked_loss, contrastive_loss] for logging."""
    f = X.shape[-1]
    model = SSLModel(encoder, f, lam=lam)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.from_numpy(X)
    hist = []
    model.train()
    for ep in range(epochs):
        tot = []
        for b in _batches(len(Xt), bs):
            if len(b) < 8:  # skip tiny last batch (contrastive loss needs several negatives)
                continue
            loss, lm, lc = model(Xt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot.append((lm, lc))
        hist.append(np.mean(tot, 0).tolist())
    return hist


def finetune(encoder, Xtr, ymove, yvol, Xva=None, yva=None, epochs=12, bs=128, lr=1e-3,
             freeze_encoder=False, vol_w=1.0):
    """SUPERVISED training of encoder + heads. Loss = cross-entropy(movement) + vol_w * MSE(volatility).

    If a validation set is given, we keep the weights from the epoch with the best validation score
    (early stopping; score = -MCC + RMSE, lower is better). The validation set comes AFTER train in
    time, so this does not peek at the test period.
    Pass a randomly initialised encoder for the 'scratch' baseline, or a pretrained one for 'ssl'.
    """
    model = Predictor(encoder)
    params = list(model.move.parameters()) + list(model.vol.parameters()) if freeze_encoder \
        else model.parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    Xt, ym, yv = torch.from_numpy(Xtr), torch.from_numpy(ymove), torch.from_numpy(yvol)
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        if freeze_encoder:
            model.enc.eval()
        for b in _batches(len(Xt), bs):
            lm, lv = model(Xt[b])
            loss = Fn.cross_entropy(lm, ym[b]) + vol_w * Fn.mse_loss(lv, yv[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if Xva is not None:
            v = evaluate(model, Xva, yva[0], yva[1])
            score = -v["mcc"] + v["rmse"]
            if score < best:
                best, best_state = score, {k: t.clone() for k, t in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, X, bs=512):
    """Return (predicted movement class, predicted log-vol) for all windows in X."""
    model.eval()
    Xt = torch.from_numpy(X)
    lm, lv = [], []
    for i in range(0, len(Xt), bs):
        a, b = model(Xt[i:i + bs])
        lm.append(a); lv.append(b)
    return torch.cat(lm).argmax(1).numpy(), torch.cat(lv).numpy()


@torch.no_grad()
def embed(encoder, X, bs=512):
    """Encode windows into embedding vectors (N, d); used for regime clustering."""
    encoder.eval()
    Xt = torch.from_numpy(X)
    return torch.cat([encoder(Xt[i:i + bs]) for i in range(0, len(Xt), bs)]).numpy()


def evaluate(model, X, ymove, yvol):
    """Metrics. Movement: accuracy, macro-F1 (averages the 3 classes equally), and MCC
    (Matthews correlation: 0 = chance, 1 = perfect; robust to class imbalance).
    Volatility: MAE and RMSE in standardised units."""
    pm, pv = predict(model, X)
    return dict(acc=accuracy_score(ymove, pm), f1=f1_score(ymove, pm, average="macro"),
                mcc=matthews_corrcoef(ymove, pm),
                mae=float(np.abs(pv - yvol).mean()), rmse=float(np.sqrt(((pv - yvol) ** 2).mean())))
