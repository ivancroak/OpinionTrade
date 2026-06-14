"""Training-corpus generation, loading and live snapshot logging.

The fill-probability model trains on a corpus of orderbook snapshots. This
module provides three things:

1. :func:`generate_training_frame` -- a *reproducible, seeded* synthetic corpus
   of orderbook snapshots whose ``filled`` label is a noisy function of the
   microstructure features, so the model has a genuine (non-trivial) signal to
   learn. It is clearly synthetic and exists so the pipeline is runnable
   out-of-the-box; replace it with your own logged snapshots for production use.
2. :func:`load_default_training_frame` -- loads the committed CSV corpus from
   ``data/orderbook_snapshots.csv`` (regenerating in-memory if absent).
3. :func:`append_snapshot` -- appends a *real* live snapshot row to a CSV so the
   bot can accumulate its own historical orderbook snapshots over time.

Running ``python -m src.ml.dataset`` (re)writes the committed CSV deterministically.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import RAW_SNAPSHOT_COLUMNS, market_snapshot_row

logger = logging.getLogger(__name__)

#: Columns persisted for each stored snapshot row.
SNAPSHOT_COLUMNS: list[str] = [
    "market_id",
    "side",
    "title",
    *RAW_SNAPSHOT_COLUMNS,
    "filled",
]

#: Default on-disk location of the committed corpus (repo_root/data/...).
DEFAULT_DATA_PATH: Path = Path(__file__).resolve().parents[2] / "data" / "orderbook_snapshots.csv"

_DEFAULT_N_SAMPLES = 900
_DEFAULT_SEED = 7


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_training_frame(
    n_samples: int = _DEFAULT_N_SAMPLES, *, seed: int = _DEFAULT_SEED
) -> pd.DataFrame:
    """Generate a deterministic synthetic corpus of labelled orderbook snapshots.

    The label is drawn from a logistic model of the engineered features (tighter
    spreads, higher volume/liquidity efficiency, more top-of-book concentration
    and bid-heavy depth imbalance => higher fill probability), with Bernoulli
    noise so the relationship is learnable but not perfectly separable.
    """
    rng = np.random.default_rng(seed)

    best_bid = rng.uniform(0.05, 0.95, n_samples)
    # Spread in tick units (tick = 0.001), heavier near the touch.
    spread = np.clip(rng.exponential(0.008, n_samples), 0.001, 0.08)
    total_liquidity = rng.lognormal(mean=8.5, sigma=1.1, size=n_samples)
    volume_24h = rng.lognormal(mean=9.0, sigma=1.3, size=n_samples)
    depth_imbalance = np.clip(rng.normal(0.0, 0.45, n_samples), -1.0, 1.0)
    liquidity_concentration = np.clip(rng.beta(2.0, 2.0, n_samples), 0.02, 1.0)
    book_depth_levels = rng.integers(1, 25, n_samples).astype(float)
    has_bonus_points = rng.binomial(1, 0.4, n_samples).astype(float)

    # Derived signals approximating features.build_feature_frame (efficiency uses
    # raw division here -- synthetic total_liquidity is always > 0).
    mid = best_bid + spread / 2.0
    relative_spread = spread / mid
    efficiency = volume_24h / total_liquidity

    latent = (
        1.15 * np.log1p(efficiency)
        - 14.0 * relative_spread
        + 1.30 * (liquidity_concentration - 0.5)
        + 0.85 * depth_imbalance
        + 0.20 * np.log1p(total_liquidity)
        + 0.35 * has_bonus_points
    )
    # Centre and scale the latent score so probabilities span a useful range.
    latent = (latent - latent.mean()) / (latent.std() + 1e-9)
    fill_prob = _sigmoid(1.25 * latent)
    filled = rng.binomial(1, fill_prob).astype(int)

    sides = np.where(rng.binomial(1, 0.5, n_samples) == 1, "YES", "NO")
    frame = pd.DataFrame(
        {
            "market_id": np.arange(1, n_samples + 1),
            "side": sides,
            "title": [f"synthetic-market-{i}" for i in range(1, n_samples + 1)],
            "best_bid": best_bid,
            "spread": spread,
            "total_liquidity": total_liquidity,
            "volume_24h": volume_24h,
            "depth_imbalance": depth_imbalance,
            "liquidity_concentration": liquidity_concentration,
            "book_depth_levels": book_depth_levels,
            "has_bonus_points": has_bonus_points,
            "filled": filled,
        }
    )
    return frame.loc[:, SNAPSHOT_COLUMNS]


def write_csv(frame: pd.DataFrame, path: str | Path = DEFAULT_DATA_PATH) -> Path:
    """Write a snapshot corpus to CSV, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_frame(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load a snapshot corpus from CSV."""
    return pd.read_csv(Path(path))


def load_default_training_frame() -> pd.DataFrame:
    """Load the committed corpus, regenerating it in-memory if missing."""
    if DEFAULT_DATA_PATH.exists():
        return load_frame(DEFAULT_DATA_PATH)
    logger.warning("corpus %s not found; generating synthetic data in-memory", DEFAULT_DATA_PATH)
    return generate_training_frame()


def append_snapshot(
    path: str | Path,
    market: Any,
    side: str,
    orderbook: Any,
    *,
    filled: int | None = None,
) -> None:
    """Append one live orderbook snapshot to a CSV (header written on first use).

    This is the production data-collection hook: call it from the trading loop
    once an order's outcome is known to grow a real historical corpus that
    :func:`load_frame` / the training CLI can consume.
    """
    row = market_snapshot_row(market, side, orderbook)
    row["filled"] = filled
    record = {col: row.get(col) for col in SNAPSHOT_COLUMNS}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SNAPSHOT_COLUMNS)
        # tell() == 0 only when the file was just created empty -- race-free
        # within a process, so concurrent appends never duplicate the header.
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(record)


def main() -> None:
    """Regenerate the committed corpus deterministically."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    frame = generate_training_frame()
    out = write_csv(frame)
    fill_rate = float(frame["filled"].mean())
    logger.info("wrote %d snapshots -> %s (fill rate %.1f%%)", len(frame), out, 100 * fill_rate)


if __name__ == "__main__":
    main()
