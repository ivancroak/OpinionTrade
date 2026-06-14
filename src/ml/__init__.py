"""Optional machine-learning layer for OpinionTrade.

A self-contained, additive package that learns an execution policy from
orderbook-microstructure features instead of a static heuristic:

* :mod:`~src.ml.features`  -- pandas/NumPy feature engineering from orderbook snapshots
* :mod:`~src.ml.dataset`   -- reproducible training corpus + live snapshot logging
* :mod:`~src.ml.model`     -- scikit-learn fill-probability model (+ heuristic fallback)
* :mod:`~src.ml.ranker`    -- ranks markets/sides by predicted fill probability

Importing this package requires the optional ML dependencies (``pandas``,
``numpy``, ``scikit-learn``, ``joblib``). The core trading bot runs without them;
callers that want the ML path should guard the import::

    try:
        from src.ml import rank_markets_by_fill_probability
        _ML_AVAILABLE = True
    except ImportError:
        _ML_AVAILABLE = False
"""

from __future__ import annotations

from .dataset import (
    DEFAULT_DATA_PATH,
    append_snapshot,
    generate_training_frame,
    load_default_training_frame,
    load_frame,
)
from .features import (
    FEATURE_COLUMNS,
    RAW_SNAPSHOT_COLUMNS,
    build_feature_frame,
    extract_features,
    feature_matrix,
    orderbook_features,
)
from .model import MIN_TRAIN_SAMPLES, FillProbabilityModel, RankedMarketScore
from .ranker import (
    default_model,
    format_ranking,
    rank_markets_by_fill_probability,
)

__all__ = [
    "DEFAULT_DATA_PATH",
    "FEATURE_COLUMNS",
    "MIN_TRAIN_SAMPLES",
    "RAW_SNAPSHOT_COLUMNS",
    "FillProbabilityModel",
    "RankedMarketScore",
    "append_snapshot",
    "build_feature_frame",
    "default_model",
    "extract_features",
    "feature_matrix",
    "format_ranking",
    "generate_training_frame",
    "load_default_training_frame",
    "load_frame",
    "orderbook_features",
    "rank_markets_by_fill_probability",
]
