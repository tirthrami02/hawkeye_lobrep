"""Feature engineering, windowing, and leakage-safe train/val/test splitting.

PIPELINE: raw book (T,40) -> per-step features (T,8) -> sliding windows (N,50,8) + targets.
Each window is one training example: "given the last 50 book snapshots, what happens next?"
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .data import LEVELS, LOBData


def lob_features(raw: np.ndarray, depth: int = 5) -> np.ndarray:
    """Turn raw 40-column books into 8 microstructure features per time step.

    Columns of the output, in order:
      0 log-return of the mid-price (price change since previous step)
      1 relative spread = (best ask - best bid) / mid   (liquidity cost; widens when volatile)
      2 order-book imbalance (OBI) over top 1 level   } OBI = (bid vol - ask vol)/(bid + ask vol).
      3 OBI over top 3 levels                          } Ranges -1..+1; positive = more buyers
      4 OBI over top `depth` (5) levels                } waiting, which tends to precede price rises.
      5 microprice offset: volume-weighted mid minus mid, relative. Leans toward the thicker side.
      6 log of total volume in top 5 levels (market depth)
      7 depth-weighted imbalance across all 10 levels (nearer levels count more: weight 1/level)
    """
    ap, av = raw[:, 0::4], raw[:, 1::4]  # ask price / volume, shape (T,10)
    bp, bv = raw[:, 2::4], raw[:, 3::4]  # bid price / volume
    mid = (ap[:, 0] + bp[:, 0]) / 2
    ret = np.diff(np.log(mid), prepend=np.log(mid[0]))  # first step gets 0 return
    spread = (ap[:, 0] - bp[:, 0]) / mid
    cols = [ret, spread]
    for d in (1, 3, depth):
        b, a = bv[:, :d].sum(1), av[:, :d].sum(1)
        cols.append((b - a) / (b + a + 1e-9))  # 1e-9 avoids divide-by-zero
    micro = (ap[:, 0] * bv[:, 0] + bp[:, 0] * av[:, 0]) / (bv[:, 0] + av[:, 0] + 1e-9)
    cols.append((micro - mid) / mid)
    cols.append(np.log1p(bv[:, :depth].sum(1) + av[:, :depth].sum(1)))
    w = 1.0 / np.arange(1, LEVELS + 1)  # level weights 1, 1/2, 1/3, ...
    cols.append(((bv - av) @ w) / ((bv + av) @ w + 1e-9))
    return np.stack(cols, 1).astype(np.float32)


@dataclass
class Windows:
    """A set of training examples.
    X: (N, W, F) feature windows | y_move: 0=down,1=flat,2=up | y_vol: log realised vol
    regime: ground-truth regime at each window's last step (simulator only) or None."""
    X: np.ndarray
    y_move: np.ndarray
    y_vol: np.ndarray
    regime: np.ndarray | None


def make_windows(data: LOBData, window: int = 50, horizon: int = 10, stride: int = 5,
                 flat_q: float = 0.33, thresh: float | None = None):
    """Slice the series into overlapping windows and attach FUTURE-looking targets.

    For a window ending at step e:
      * movement label: sign of log(mid[e+horizon]/mid[e]), with a dead-zone `thresh`
        (moves smaller than thresh count as "flat"). thresh is the `flat_q` quantile of |move|
        on the data given, so ~33% are flat. When evaluating a *different* dataset we pass the
        TRAIN thresh back in, so the shift in volatility is real rather than normalised away.
      * volatility target: log of sqrt(sum of squared returns over the next `horizon` steps)
        (realised volatility; log makes the distribution closer to symmetric).
    The window itself only contains steps <= e, so inputs never see the future.
    Returns (Windows, thresh).
    """
    F = lob_features(data.raw)
    mid = (data.raw[:, 0] + data.raw[:, 2]) / 2
    T = len(mid)
    ends = np.arange(window - 1, T - horizon, stride)  # last valid window end leaves room for horizon
    fut = np.log(mid[ends + horizon]) - np.log(mid[ends])
    if thresh is None:
        thresh = float(np.quantile(np.abs(fut), flat_q))
    y_move = np.where(fut > thresh, 2, np.where(fut < -thresh, 0, 1)).astype(np.int64)
    r = F[:, 0]
    y_vol = np.array([np.log(np.sqrt((r[e + 1:e + 1 + horizon] ** 2).sum()) + 1e-8) for e in ends],
                     dtype=np.float32)
    X = np.stack([F[e - window + 1:e + 1] for e in ends])
    reg = data.regimes[ends] if data.regimes is not None else None
    return Windows(X, y_move, y_vol, reg), thresh


def time_split(w: Windows, train=0.6, val=0.2, gap: int = 20):
    """Chronological split: first 60% train, next 20% validation, last 20% test.

    NEVER shuffle time series: a random split would put windows from the same moment in both
    train and test, leaking information. We also leave an EMBARGO of `gap` windows at the end of
    train and validation, because overlapping windows and forward-looking labels near a boundary
    would otherwise share information across it.
    """
    n = len(w.X)
    a, b = int(n * train), int(n * (train + val))
    sl = lambda lo, hi: Windows(w.X[lo:hi], w.y_move[lo:hi], w.y_vol[lo:hi],
                                None if w.regime is None else w.regime[lo:hi])
    return sl(0, a - gap), sl(a, b - gap), sl(b, n)


class Scaler:
    """Standardise each feature to mean 0 / std 1. Fit on TRAIN ONLY, then applied to val/test,
    so no test statistics leak into training."""
    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-8
        return self

    def __call__(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)
