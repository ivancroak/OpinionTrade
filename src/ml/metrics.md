# Fill-Probability Model — Evaluation

`src/ml/` trains a scikit-learn classifier that predicts the probability an order
placed at the current book gets **filled**, from engineered orderbook-microstructure
features. This file records the model definition and the metrics it produces.
**No accuracy figure is hard-coded anywhere in the project** — every number below is
reproduced live by the training CLI (`python -m src.ml.train`).

## Pipeline

`SimpleImputer(median) → StandardScaler → GradientBoostingClassifier`

- Target: `filled` (1 = order filled, 0 = not). `predict_proba` → fill probability.
- Features: spread, relative spread, depth imbalance, bid-side liquidity concentration
  (Herfindahl index), volume/liquidity efficiency, log total liquidity, log 24h volume,
  book-depth levels, best bid, bonus-points flag.
- Falls back to a deterministic, fixed-reference heuristic when there is too little
  data to train (no silent failure, no fabricated scores).

## Dataset

`data/orderbook_snapshots.csv` — the bundled snapshot corpus.

| | |
|---|---|
| Snapshots | 900 |
| Fill rate | 46.9% |
| Regenerate | `python -m src.ml.dataset` |

For live use, append real snapshots from the trading loop with `append_snapshot()` and
retrain on the accumulated CSV — the schema and loader are identical.

## Results — 5-fold stratified cross-validation

**ROC-AUC: 0.709** (trained gradient-boosted estimator, not the heuristic fallback).

| Feature | Gain importance |
|---|---|
| volume_liquidity_efficiency | 0.296 |
| relative_spread | 0.154 |
| log_volume_24h | 0.120 |
| depth_imbalance | 0.118 |
| liquidity_concentration | 0.100 |

The top drivers are exactly the microstructure signals the execution policy reasons
about — efficiency, spread, depth imbalance and concentration.

## Reproduce

```bash
pip install -e ".[ml]"
python -m src.ml.train                                      # prints the metrics above
python -m src.ml.train --save src/ml/artifacts/fill_model.joblib
```

The persisted model `src/ml/artifacts/fill_model.joblib` is the artifact produced by
the last command, loadable with `FillProbabilityModel.load(...)`.
