import numpy as np
import torch

from lobrep.data import simulate_lob
from lobrep.features import Scaler, make_windows, time_split
from lobrep.models import ENCODERS, Predictor, SSLModel, build_encoder


def small():
    w, thr = make_windows(simulate_lob(3000, seed=1))
    return w, thr


def test_shapes_and_labels():
    w, _ = small()
    assert w.X.shape[1:] == (50, 8)
    assert set(np.unique(w.y_move)) <= {0, 1, 2}
    assert np.isfinite(w.X).all() and np.isfinite(w.y_vol).all()


def test_time_split_is_chronological_with_embargo():
    w, _ = small()
    tr, va, te = time_split(w, gap=20)
    n = len(w.X)
    assert len(tr.X) + len(va.X) + len(te.X) == n - 40
    # last train window must start before first val window with an embargo gap
    assert not np.array_equal(tr.X[-1], va.X[0])


def test_scaler_fit_on_train_only():
    w, _ = small()
    tr, _, te = time_split(w)
    sc = Scaler().fit(tr.X)
    assert np.allclose(sc(tr.X).reshape(-1, 8).mean(0), 0, atol=1e-3)


def test_encoders_forward_and_ssl_backward():
    x = torch.randn(16, 50, 8)
    for name in ENCODERS:
        enc = build_encoder(name, 8, 16)
        assert enc(x).shape == (16, 16)
        lm, lv = Predictor(enc)(x)
        assert lm.shape == (16, 3) and lv.shape == (16,)
        loss, _, _ = SSLModel(enc, 8)(x)
        loss.backward()


def test_no_future_leak_in_features():
    """Perturbing prices strictly after a window's end must not change that window."""
    d = simulate_lob(500, seed=2)
    w1, _ = make_windows(d)
    d.raw[300:, :] *= 1.5
    w2, _ = make_windows(d, thresh=0.0)
    k = (299 - 49) // 5
    assert np.allclose(w1.X[k], w2.X[k])
