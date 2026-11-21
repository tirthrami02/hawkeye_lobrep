# Hawkeye_lobrep: Does Self-Supervised Pretraining Help Limit-Order-Book Models Under Distribution Shift?

A small, fully reproducible research codebase that asks:

> **Does self-supervised pretraining (masked time-series modelling + contrastive learning) improve
> short-horizon price-movement prediction, volatility forecasting and market-regime discovery from
> limit-order-book (LOB) data, especially with few labels or under distribution shift?**

Everything runs on a **CPU** (the full study takes about 25 minutes). Every number below was produced by
`scripts/run_experiments.py` and is stored in [`results/`](results/).

## Headline finding (honest, and a negative result)

In this controlled setup, **SSL pretraining did *not* reliably beat training from scratch.**

| Setting (mean over 3 seeds × 3 encoders) | Scratch | SSL-pretrained |
|---|---|---|
| In-distribution, 100% labels, movement MCC | 0.132 | 0.145 |
| In-distribution, 10% labels, movement MCC | **0.119** | 0.059 |
| In-distribution, 10% labels, volatility RMSE (std. units) | 0.834 | **0.751** |
| Shifted instrument, 100% labels, movement MCC | 0.108 | 0.108 |
| Shifted instrument, 10% labels, volatility RMSE | 1.348 | 1.340 |

- Movement: no consistent gain; a slight edge at full labels, a loss at 10% and 2% labels.
- Volatility: SSL helps modestly in the low-label regime (about 10% lower RMSE at 10% labels), and not at full labels.
- Regimes: K-Means on SSL embeddings scored ARI 0.29 to 0.42, about the same as **raw rolling statistics (0.39)** and
  random-initialised encoders (0.29 to 0.41). HDBSCAN on SSL embeddings gave the highest ARI for the transformer
  (0.48 vs 0.45 for random init), a gap well within seed noise.
- Encoders: LSTM, CNN and Transformer land within noise of each other on this data.

Full tables: `results/summary.csv`, `results/regime_summary.csv`, plot: `results/label_efficiency.png`.

### Real-data check: FI-2010 (Nasdaq Nordic stocks)

[FI-2010](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) is a public benchmark of real
limit-order-book data (5 Finnish stocks, 10 trading days). Same Transformer, same code path, one run:

| Model (horizon k=1, 3-class mid-price movement) | Accuracy | Macro-F1 | MCC |
|---|---|---|---|
| Transformer, trained from scratch | 0.8529 | 0.8117 | 0.7140 |
| Transformer, SSL-pretrained then fine-tuned | 0.8545 | 0.8121 | 0.7154 |

**Reading:** both models learn strong signal (MCC ~0.71, where 0 is chance), but pretraining adds essentially
nothing (ΔMCC = +0.0014). With a single seed that gap is within noise, so this agrees with the simulator
study: no reliable benefit from SSL pretraining in this setup.

**Caveats:** (1) This is a chronological 60/20/20 split of the *training* file (`Train_Dst_NoAuction_DecPre_CF_7.txt`),
**not the official FI-2010 test split**, so it is not directly comparable to published numbers.
(2) Only the shortest horizon (k=1, the easiest) was run; published work also reports k=5 and k=10.
(3) One model, one seed. (4) No majority-class baseline was computed, so accuracy alone is not very informative;
MCC and F1 are the more reliable figures. Command: `python scripts/run_fi2010.py --file <path to DecPre training file>`.

**Why this is still a useful result.** Many papers report that SSL "helps" without the right controls. This repo
includes the controls (scratch baseline, random-init embeddings, raw-feature clustering, label-fraction sweep,
shifted test instrument, multiple seeds), and under them the gain largely disappears. Plausible reasons are in
*Limitations*.

## What's in the box

```
src/lobrep/
  data.py       regime-switching LOB simulator (4 hidden regimes) + FI-2010 loader
  features.py   8 microstructure features, windowing, embargoed chronological split, train-only scaler
  models.py     LSTM / dilated-CNN / Transformer encoders, SSL objectives, multi-task heads
  train.py      SSL pretraining, fine-tuning with early stopping, metrics (Acc, macro-F1, MCC, MAE, RMSE)
  regimes.py    K-Means + HDBSCAN regime discovery; Silhouette / ARI / NMI
  api.py        FastAPI inference service
scripts/        run_experiments.py (main study), run_fi2010.py, export_model.py
cpp/            streaming order-book-imbalance feature kernel (latency path sketch)
tests/          shape checks, chronological-split check, future-leakage test
```

### Method

1. **Data.** A simulator with 4 hidden regimes (calm range, uptrend, downtrend, high-vol) in which
   order-book imbalance is a persistent signal that leads mid-price drift, producing a noisy but learnable
   signal. A second **"instrument B"** is generated with 1.8× volatility and different regime persistence,
   and is used purely as a distribution-shifted test set. Its label threshold is *reused from the source
   instrument*, so the shift is real.
2. **Features.** Log-return, relative spread, order-book imbalance at 1/3/5 levels, microprice offset,
   log depth, depth-weighted imbalance. Windows of 50 ticks.
3. **Targets.** 3-class mid-price movement over the next 10 ticks; log realised volatility of the next 10 ticks.
4. **Leakage control.** Chronological 60/20/20 split with a 20-window embargo between segments, scaler and
   thresholds fit on train only, SSL pretrains on unlabelled *train* windows only, and a unit test that perturbs
   post-window prices and asserts the window is unchanged.
5. **SSL.** Masked span reconstruction (30% of steps, span 3) + NT-Xent contrastive loss between two jittered /
   rescaled views, weighted 0.5 / 0.5.
6. **Fine-tuning.** Multi-task (cross-entropy + MSE), early stopping on validation, with 100%, 10% and 2% of labels.
7. **Regimes.** Cluster test-set embeddings (K-Means k=4, HDBSCAN) and compare to ground-truth regimes.

## Reproduce

Requires Python 3.10+. No GPU needed.

**1. Environment and tests** (macOS/Linux shown; on Windows activate with `.venv\Scripts\activate`):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[serve,dev]"
pytest -q                                           # 5 tests
```

**2. Main study on simulated data** (no downloads needed):

```bash
python scripts/run_experiments.py                   # full study, ~25 min CPU; writes results/
python scripts/run_experiments.py --quick           # ~1 min smoke run (undertrained; not for conclusions)
python scripts/run_experiments.py --mlflow          # optional: pip install mlflow, then `mlflow ui`
```

**3. Real-data check on FI-2010** (optional, needs a manual download):

1. Create a `data/` folder in the repo root: `mkdir data`
2. Download `BenchmarkDatasets.zip` (about 1.8 GB, free, CC0) from the
   [FI-2010 dataset page](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649)
   using your browser (the host needs a download token, so scripted download does not work).
3. Unzip it inside `data/` so this file exists:
   `data/BenchmarkDatasets/NoAuction/3.NoAuction_DecPre/NoAuction_DecPre_Training/Train_Dst_NoAuction_DecPre_CF_7.txt`
4. Run (takes tens of minutes to a couple of hours on a laptop CPU; it prints nothing until each of its two
   stages finishes, and a harmless `enable_nested_tensor` warning is expected):

```bash
python scripts/run_fi2010.py --file data/BenchmarkDatasets/NoAuction/3.NoAuction_DecPre/NoAuction_DecPre_Training/Train_Dst_NoAuction_DecPre_CF_7.txt
```

Use the **DecPre** (un-normalised) files, not ZScore or MinMax; the features take logs and ratios of prices.
`data/` is git-ignored, so the dataset is never committed.

**4. Serve a model (optional demo API).** The service runs only on your own computer; nothing is exposed to the internet.

```bash
python scripts/export_model.py                      # trains a small demo model, saves artifacts/model.pt (~1 min)
python -m uvicorn lobrep.api:app --port 8000        # starts the server (use `python -m` so the venv's uvicorn is used)
```

Then open these addresses in a browser while the server is running:

| Address | What it shows |
|---|---|
| http://127.0.0.1:8000/health | `{"status":"ok","trained":true}` means the saved model loaded. (`/` itself shows "Not Found"; that is normal, there is no page there.) |
| http://127.0.0.1:8000/docs | Interactive page to try `POST /predict` (click "Try it out"). |

`POST /predict` takes `{"window": [[...8 features...] x 50]}` and returns `p_down`, `p_flat`, `p_up` and `log_vol_forecast`.
`127.0.0.1` means "this computer" and `8000` is the port the server listens on. The demo model is trained on
simulated data, so its outputs only show that the service works, not real market predictions.
Stop the server with `Ctrl+C` in the terminal where it is running.

## Status of each component

| Component | Status |
|---|---|
| Simulator study, models, SSL, clustering, tests | Run and verified (5 passing tests; results in `results/`) |
| FastAPI service | Written and loads; endpoint not load-tested |
| `scripts/run_fi2010.py` (FI-2010 benchmark) | Run on a MacBook (CPU), horizon k=1, Transformer only, one seed. Results above. Not run on the official test split, other horizons, LSTM/CNN, or multiple seeds |
| `cpp/lob_features.cpp` | Compiled (Zig C++, `-O3`) and run: 5M synthetic books in ~0.70 s ≈ **140 ns/book on one core**, *including* the RNG that generates 20 volumes per book, so the pure feature cost is lower. Not compared against the Python implementation, and not checked for numerical parity with `features.py` |

## Limitations (please read)

- **Mostly synthetic data.** The main study (all three encoders, label fractions, distribution shift, regimes) uses the simulator; only a small one-seed check ran on real data (FI-2010). Regime discovery was evaluated only on the simulator, because real data has no ground-truth regimes. In a simulator whose signal (OBI) is
  directly observable in the input features, there is little left for pretraining to discover, which may explain
  the null result. Real LOB data has richer structure, so the FI-2010 experiments are the real test.
- **Small scale.** 3 seeds, 32-dim encoders, ~8 pretraining epochs, no hyperparameter search, no significance
  tests. Differences of ~0.02 MCC are noise.
- Predictive MCC of ~0.13 to 0.16 reflects the simulator's built-in noise level, not a claim about tradable edge.
- No transaction costs or backtest; this is a representation-learning study, not a trading strategy.

## Roadmap

1. Extend FI-2010: official train/test split, horizons k=5 and k=10, LSTM and CNN, multiple seeds, a majority-class baseline, and comparison against published DeepLOB-style baselines.
2. Stronger SSL (longer pretraining, larger models, GPU), and ablate masked vs contrastive terms.
3. Benchmark the C++ kernel against the Python features and verify numerical parity; export the model via ONNX.
4. Paired bootstrap significance tests across seeds.

## License

MIT
