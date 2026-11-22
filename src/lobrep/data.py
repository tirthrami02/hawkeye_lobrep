"""Data layer: a regime-switching limit-order-book (LOB) simulator and a FI-2010 loader.

BACKGROUND
----------
A limit order book lists, for one stock, the resting buy orders (bids) and sell orders (asks)
at each price level. "Level 1" is the best price on each side; we keep 10 levels. The mid-price is
the average of best bid and best ask. Quants study this data because the *shape* of the book
(e.g. much more volume on the bid side) carries information about where price goes next.

CANONICAL FORMAT used everywhere in this repo
---------------------------------------------
A numpy array `raw` of shape (T, 40): T time steps, and for each of the 10 levels four numbers in
this order: ask_price, ask_volume, bid_price, bid_volume. This matches FI-2010's column layout,
so real data and simulated data go through exactly the same code.

WHY A SIMULATOR?
----------------
Real LOB data is licensed/large. The simulator lets the whole study run on a laptop CPU and, crucially,
gives us GROUND-TRUTH regime labels, so we can measure whether clustering really finds the regimes.
The cost: conclusions are about the simulator, not real markets (stated in the README).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

LEVELS = 10  # number of price levels per side

# The four hidden market regimes the simulator switches between.
#   drift          : average price push per step (in units of 0.2 ticks)
#   vol            : size of random price noise
#   spread         : typical gap between best ask and best bid, in ticks
#   depth          : typical order volume at each level (thin books in volatile regimes)
#   imb_rho        : persistence (AR(1) coefficient) of the order-book imbalance signal
REGIMES = {
    0: dict(name="calm-range", drift=0.0, vol=0.4, spread=1.0, depth=300.0, imb_rho=0.80),
    1: dict(name="uptrend", drift=0.25, vol=0.7, spread=1.2, depth=250.0, imb_rho=0.93),
    2: dict(name="downtrend", drift=-0.25, vol=0.7, spread=1.2, depth=250.0, imb_rho=0.93),
    3: dict(name="high-vol", drift=0.0, vol=2.0, spread=2.5, depth=120.0, imb_rho=0.60),
}


@dataclass
class LOBData:
    """Container: raw book array, optional ground-truth regime per step, and a name."""
    raw: np.ndarray            # (T, 40)
    regimes: np.ndarray | None  # (T,) true regime ids (simulator only; None for real data)
    name: str = "lob"


def simulate_lob(n_steps: int = 60_000, seed: int = 0, vol_scale: float = 1.0,
                 stay_prob: float = 0.998, tick: float = 0.01, p0: float = 100.0) -> LOBData:
    """Simulate a limit order book whose behaviour depends on a hidden Markov regime.

    How it works, step by step:
      1. Regime chain: at each step the regime stays the same with probability `stay_prob`,
         otherwise jumps to a random regime. (So regimes last ~500 steps on average.)
      2. Imbalance signal `obi`: an AR(1) process, i.e. obi[t] = rho*obi[t-1] + noise. It is
         persistent, mimicking how real order-book pressure lingers.
      3. Price: each step's return = regime drift + a push proportional to the PREVIOUS step's
         imbalance + Gaussian noise. Using the previous step makes imbalance a genuine *leading*
         indicator (it is observable before the price move it causes), which is what real
         microstructure research finds and what makes prediction possible but noisy.
      4. Book: volume on the bid side is scaled up by exp(+obi/2) and the ask side by exp(-obi/2),
         so imbalance is visible in the data. Deeper levels hold more volume.

    `vol_scale` multiplies price noise; setting it >1 builds a DIFFERENT "instrument" used as a
    distribution-shifted test set.
    """
    rng = np.random.default_rng(seed)  # seeded => fully reproducible
    k = len(REGIMES)
    # Step 1: generate the hidden regime path.
    reg = np.zeros(n_steps, dtype=np.int64)
    for t in range(1, n_steps):
        reg[t] = reg[t - 1] if rng.random() < stay_prob else rng.integers(k)
    mid = np.empty(n_steps)
    mid[0] = p0
    obi = np.zeros(n_steps)
    raw = np.zeros((n_steps, 4 * LEVELS))
    for t in range(n_steps):
        r = REGIMES[reg[t]]
        prev = obi[t - 1] if t else 0.0
        # Step 2: AR(1) imbalance; sqrt(1-rho^2) keeps its variance constant across regimes.
        obi[t] = r["imb_rho"] * prev + np.sqrt(1 - r["imb_rho"] ** 2) * rng.standard_normal() * 0.5
        # Step 3: price update uses obi[t-1] (past), never the current/future value.
        if t:
            ret = (r["drift"] * 0.2 + 0.6 * np.tanh(obi[t - 1]) * r["vol"] * 0.5
                   + r["vol"] * vol_scale * rng.standard_normal()) * tick
            mid[t] = max(mid[t - 1] + ret, tick * 10)  # floor keeps prices positive
        spread = max(1.0, r["spread"] + 0.3 * rng.standard_normal()) * tick
        # Step 4: fill the 10 levels on each side.
        for l in range(LEVELS):
            base = r["depth"] * (1 + 0.15 * l)               # deeper levels hold more volume
            bid_v = base * np.exp(0.5 * obi[t]) * rng.lognormal(0, 0.15)
            ask_v = base * np.exp(-0.5 * obi[t]) * rng.lognormal(0, 0.15)
            ask_p = mid[t] + spread / 2 + l * tick           # asks step up from the mid
            bid_p = mid[t] - spread / 2 - l * tick           # bids step down from the mid
            raw[t, 4 * l:4 * l + 4] = (ask_p, ask_v, bid_p, bid_v)
    return LOBData(raw=raw, regimes=reg, name=f"sim(seed={seed},vol={vol_scale})")


def load_fi2010(path: str, horizon_idx: int = 0) -> tuple[LOBData, np.ndarray]:
    """Load a FI-2010 text file (e.g. Train_Dst_NoAuction_DecPre_CF_7.txt).

    FI-2010 is a public benchmark of Nasdaq Nordic order books (5 Finnish stocks, 10 days).
    File layout: 149 rows x N columns, time runs along the columns. Rows 0-39 are the raw 10-level
    LOB; the last 5 rows are labels for prediction horizons k = 1, 2, 3, 5, 10 events
    (value 1=up, 2=stationary, 3=down). We transpose to (N, 40) and shift labels to {0,1,2}.
    NOTE: use the un-normalised 'DecPre' files; the z-scored files break log/ratio features.
    """
    arr = np.loadtxt(path)
    raw = arr[:40].T
    labels = arr[-5 + horizon_idx].astype(int) - 1
    return LOBData(raw=raw, regimes=None, name="FI-2010"), labels
