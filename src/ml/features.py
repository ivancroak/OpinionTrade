"""Orderbook-microstructure feature engineering (pandas / NumPy).

This module turns raw Opinion.trade orderbook snapshots into a numeric feature
matrix suitable for a scikit-learn model. The engineered features are exactly the
ones the execution policy reasons about:

* ``spread``                       -- best ask minus best bid
* ``relative_spread``              -- spread normalised by the mid price
* ``depth_imbalance``              -- (bid_liq - ask_liq) / (bid_liq + ask_liq)
* ``liquidity_concentration``      -- Herfindahl index of bid-level liquidity
* ``volume_liquidity_efficiency``  -- 24h volume / resting orderbook liquidity

plus a few supporting scale features (log liquidity, log volume, book depth,
best bid level, bonus-points flag).

Everything here is pure, side-effect free and dependency-light: it only needs
``pandas``/``numpy`` and duck-typed orderbook objects exposing ``bids``/``asks``
lists of levels with ``price``/``amount`` attributes (see
``src.opinion_client.Orderbook``). That keeps the feature layer unit-testable
without the trading SDK or a network connection.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

# --- Column contracts -------------------------------------------------------

#: Raw per-snapshot columns as stored on disk / produced from a live orderbook.
RAW_SNAPSHOT_COLUMNS: list[str] = [
    "best_bid",
    "spread",
    "total_liquidity",
    "volume_24h",
    "depth_imbalance",
    "liquidity_concentration",
    "book_depth_levels",
    "has_bonus_points",
]

#: Engineered columns fed to the model, in a fixed order.
FEATURE_COLUMNS: list[str] = [
    "spread",
    "relative_spread",
    "depth_imbalance",
    "liquidity_concentration",
    "volume_liquidity_efficiency",
    "log_total_liquidity",
    "log_volume_24h",
    "book_depth_levels",
    "best_bid",
    "has_bonus_points",
]

#: Non-feature columns carried through for identification / supervision.
IDENTIFIER_COLUMNS: list[str] = ["market_id", "side", "title"]
LABEL_COLUMN: str = "filled"

# Sides and their Opinion.trade ``symbol_type`` codes (mirrors the existing
# efficiency ranker: YES -> 0, NO -> 1).
_SIDE_SYMBOL_TYPE: dict[str, int] = {"YES": 0, "NO": 1}


def _to_float(value: Any) -> float:
    """Best-effort conversion to ``float`` (Decimal/str/None safe)."""
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def orderbook_features(orderbook: Any) -> dict[str, float]:
    """Compute snapshot-level microstructure statistics from one orderbook.

    The orderbook may be ``None`` or empty on one or both sides; missing
    quantities are returned as ``NaN`` so that "unknown" is never silently
    confused with a real zero.
    """
    bids = list(getattr(orderbook, "bids", []) or [])
    asks = list(getattr(orderbook, "asks", []) or [])

    bid_prices = np.array([_to_float(lvl.price) for lvl in bids], dtype=float)
    bid_amounts = np.array([_to_float(lvl.amount) for lvl in bids], dtype=float)
    ask_prices = np.array([_to_float(lvl.price) for lvl in asks], dtype=float)
    ask_amounts = np.array([_to_float(lvl.amount) for lvl in asks], dtype=float)

    best_bid = float(bid_prices[0]) if bid_prices.size else math.nan
    best_ask = float(ask_prices[0]) if ask_prices.size else math.nan

    spread = best_ask - best_bid if bid_prices.size and ask_prices.size else math.nan

    # USD value resting on each side (price * shares per level).
    bid_liq = float(np.sum(bid_prices * bid_amounts)) if bid_prices.size else 0.0
    ask_liq = float(np.sum(ask_prices * ask_amounts)) if ask_prices.size else 0.0
    total_liq = bid_liq + ask_liq

    depth_imbalance = (bid_liq - ask_liq) / total_liq if total_liq > 0 else math.nan

    # Herfindahl-Hirschman index of bid-side liquidity: 1.0 means all depth sits
    # on a single level, 1/N means perfectly uniform across N levels. Per-level
    # liquidity is clipped to >= 0 so the index stays a valid concentration
    # measure in [0, 1] even if malformed data carries a non-positive price.
    level_liq = np.clip(bid_prices * bid_amounts, 0.0, None)
    bid_liq_positive = float(level_liq.sum())
    if bid_liq_positive > 0:
        shares = level_liq / bid_liq_positive
        liquidity_concentration = float(np.sum(np.square(shares)))
    else:
        liquidity_concentration = math.nan

    return {
        "best_bid": best_bid,
        "spread": spread,
        "total_liquidity": total_liq,
        "depth_imbalance": depth_imbalance,
        "liquidity_concentration": liquidity_concentration,
        "book_depth_levels": float(bid_prices.size),
    }


def market_snapshot_row(market: Any, side: str, orderbook: Any) -> dict[str, Any]:
    """Build one raw snapshot row from a market + its orderbook for a side."""
    feats = orderbook_features(orderbook)
    return {
        "market_id": getattr(market, "market_id", None),
        "side": side,
        "title": getattr(market, "title", ""),
        "volume_24h": _to_float(getattr(market, "volume_24h", math.nan)),
        "has_bonus_points": float(bool(getattr(market, "has_bonus_points", False))),
        **feats,
    }


def build_feature_frame(
    rows: Iterable[Mapping[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Engineer the model feature matrix from raw snapshot rows.

    Accepts either an iterable of raw-snapshot mappings or a ``DataFrame`` with
    (at least) the :data:`RAW_SNAPSHOT_COLUMNS`. Returns a frame carrying the
    identifier columns (when present), all :data:`FEATURE_COLUMNS`, and the
    :data:`LABEL_COLUMN` when it was supplied.

    Expects *raw* snapshot rows: feeding an already-engineered frame back in
    yields ``NaN`` derived columns (the raw inputs are no longer present).
    """
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))

    # Guarantee every raw input column exists so downstream math never KeyErrors.
    for col in RAW_SNAPSHOT_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan

    for col in RAW_SNAPSHOT_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    best_bid = frame["best_bid"].to_numpy(dtype=float)
    spread = frame["spread"].to_numpy(dtype=float)
    total_liq = frame["total_liquidity"].to_numpy(dtype=float)
    volume = frame["volume_24h"].to_numpy(dtype=float)

    # mid = best_bid + spread/2 (best_ask == best_bid + spread).
    mid = best_bid + spread / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_spread = np.where(mid > 0, spread / mid, np.nan)
        efficiency = np.where(total_liq > 0, volume / total_liq, np.nan)

    frame["relative_spread"] = relative_spread
    frame["volume_liquidity_efficiency"] = efficiency
    frame["log_total_liquidity"] = np.log1p(np.clip(total_liq, 0.0, None))
    frame["log_volume_24h"] = np.log1p(np.clip(volume, 0.0, None))
    frame["has_bonus_points"] = frame["has_bonus_points"].fillna(0.0).astype(float)

    # Tame non-finite values produced by degenerate books.
    frame = frame.replace([np.inf, -np.inf], np.nan)

    keep = [c for c in IDENTIFIER_COLUMNS if c in frame.columns]
    keep += FEATURE_COLUMNS
    if LABEL_COLUMN in frame.columns:
        keep.append(LABEL_COLUMN)
    return frame.loc[:, keep]


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Extract the ordered :data:`FEATURE_COLUMNS` as a float ``ndarray``."""
    return frame.reindex(columns=FEATURE_COLUMNS).to_numpy(dtype=float)


def extract_features(
    client: Any,
    markets: Iterable[Any],
    *,
    sides: tuple[str, ...] = ("YES", "NO"),
) -> pd.DataFrame:
    """Fetch live orderbooks and engineer features, one row per (market, side).

    ``client`` only needs a ``get_orderbook(token_id, question_id, symbol_type)``
    method (duck-typed against :class:`src.opinion_client.OpinionClient`). Rows
    whose orderbook could not be fetched are skipped.
    """
    rows: list[dict[str, Any]] = []
    for market in markets:
        question_id = getattr(market, "question_id", "")
        for side in sides:
            token_attr = "yes_token_id" if side == "YES" else "no_token_id"
            token_id = getattr(market, token_attr, None)
            if not token_id:
                continue
            symbol_type = _SIDE_SYMBOL_TYPE.get(side, 0)
            orderbook = client.get_orderbook(token_id, question_id, symbol_type)
            if orderbook is None:
                continue
            rows.append(market_snapshot_row(market, side, orderbook))

    if not rows:
        return build_feature_frame([])
    return build_feature_frame(rows)
