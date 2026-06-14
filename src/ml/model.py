"""Fill-probability model (scikit-learn).

:class:`FillProbabilityModel` wraps a scikit-learn pipeline
(``SimpleImputer -> StandardScaler -> GradientBoostingClassifier``) that learns
to predict the probability that a pegged limit order placed on a given market /
side will be filled, from the engineered orderbook-microstructure features in
:mod:`src.ml.features`.

Design notes (honest by construction):

* The estimator is **trained at runtime** on whatever orderbook-snapshot corpus
  is available -- it is *not* a pre-trained black box, and it publishes **no**
  headline accuracy numbers. Run :mod:`src.ml.train` to print a real
  cross-validated ROC-AUC for the data on hand.
* When there is too little labelled data to train responsibly
  (:data:`MIN_TRAIN_SAMPLES`), or when the optional ML dependencies are absent,
  the model degrades gracefully to a **deterministic heuristic** over the same
  features so that ranking still works.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, LABEL_COLUMN, feature_matrix

logger = logging.getLogger(__name__)

#: Minimum labelled rows (and per-class support) required before we trust a
#: trained estimator over the heuristic fallback.
MIN_TRAIN_SAMPLES: int = 40


@dataclass(frozen=True)
class RankedMarketScore:
    """A scored (market, side) candidate ordered by predicted fill probability."""

    market_id: Any
    side: str
    title: str
    fill_probability: float
    used_model: bool


# Reference feature mean/std for the training-free heuristic, derived once from
# the shipped orderbook-snapshot corpus (data/orderbook_snapshots.csv). Using
# *fixed* references rather than batch statistics keeps the heuristic stable and
# comparable across calls -- and well-defined even when scoring a single candidate.
_HEURISTIC_REFERENCE: dict[str, tuple[float, float]] = {
    "volume_liquidity_efficiency": (6.51196, 23.6411),
    "relative_spread": (0.0247865, 0.0342395),
    "liquidity_concentration": (0.485032, 0.230939),
    "depth_imbalance": (-0.00577636, 0.440691),
    "log_total_liquidity": (8.53229, 1.0924),
}

# Signed weights: tighter spreads, higher volume/liquidity efficiency, more
# top-of-book concentration and bid-heavy imbalance all raise the probability.
_HEURISTIC_WEIGHTS: dict[str, float] = {
    "volume_liquidity_efficiency": 1.10,
    "relative_spread": -1.00,
    "liquidity_concentration": 0.70,
    "depth_imbalance": 0.55,
    "log_total_liquidity": 0.25,
}


def _heuristic_fill_probability(matrix: np.ndarray) -> np.ndarray:
    """Transparent, training-free fill-probability estimate.

    Standardises the most informative features against fixed reference statistics
    (:data:`_HEURISTIC_REFERENCE`) and squashes a weighted sum through a logistic.
    Deterministic, and well-defined for any batch size including a single row.
    Missing features are treated as the reference mean (neutral contribution).
    """
    cols = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
    data = np.asarray(matrix, dtype=float)
    latent = np.zeros(data.shape[0], dtype=float)

    for name, weight in _HEURISTIC_WEIGHTS.items():
        mean, std = _HEURISTIC_REFERENCE[name]
        column = data[:, cols[name]]
        column = np.where(np.isfinite(column), column, mean)
        z = (column - mean) / std if std else np.zeros(data.shape[0], dtype=float)
        latent = latent + weight * z

    return 1.0 / (1.0 + np.exp(-np.clip(latent, -30.0, 30.0)))


class FillProbabilityModel:
    """Predicts P(order filled) and ranks candidates by it."""

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        *,
        random_state: int = 0,
        min_samples: int = MIN_TRAIN_SAMPLES,
    ) -> None:
        self.feature_columns = (
            list(feature_columns) if feature_columns is not None else list(FEATURE_COLUMNS)
        )
        self.random_state = random_state
        self.min_samples = min_samples
        self._pipeline: Any | None = None

    # --- lifecycle ---------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None

    def _build_pipeline(self) -> Any:
        # Imported lazily so importing this module does not hard-require sklearn
        # to merely construct the object (the heuristic path needs no sklearn).
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(random_state=self.random_state),
                ),
            ]
        )

    def _xy(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if LABEL_COLUMN not in frame.columns:
            raise ValueError(f"training frame must contain a '{LABEL_COLUMN}' column")
        x = feature_matrix(frame)
        y = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(y)
        return x[mask], y[mask].astype(int)

    def fit(self, frame: pd.DataFrame) -> FillProbabilityModel:
        """Fit on an engineered feature frame containing the label column.

        Stays in the heuristic-fallback state (``is_fitted == False``) when there
        is insufficient data or only a single class is present, rather than
        training an untrustworthy estimator.
        """
        x, y = self._xy(frame)
        classes, counts = np.unique(y, return_counts=True)
        if len(y) < self.min_samples or classes.size < 2 or counts.min() < 2:
            logger.info(
                "FillProbabilityModel: %d samples / %d classes -> heuristic fallback",
                len(y),
                classes.size,
            )
            self._pipeline = None
            return self

        pipeline = self._build_pipeline()
        pipeline.fit(x, y)
        self._pipeline = pipeline
        return self

    # --- inference ---------------------------------------------------------

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return P(fill) for each row of an engineered feature frame."""
        proba, _ = self.predict_with_provenance(frame)
        return proba

    def predict_with_provenance(self, frame: pd.DataFrame) -> tuple[np.ndarray, bool]:
        """Return ``(probabilities, used_model)`` for an engineered frame."""
        matrix = feature_matrix(frame)
        if matrix.shape[0] == 0:
            return np.empty(0, dtype=float), self.is_fitted
        if self._pipeline is None:
            return _heuristic_fill_probability(matrix), False
        proba = self._pipeline.predict_proba(matrix)[:, 1]
        return np.asarray(proba, dtype=float), True

    def rank(self, frame: pd.DataFrame) -> list[RankedMarketScore]:
        """Rank rows of an engineered frame by descending fill probability."""
        proba, used_model = self.predict_with_provenance(frame)
        scores = [
            RankedMarketScore(
                market_id=frame["market_id"].iloc[i] if "market_id" in frame.columns else None,
                side=str(frame["side"].iloc[i]) if "side" in frame.columns else "",
                title=str(frame["title"].iloc[i]) if "title" in frame.columns else "",
                fill_probability=float(proba[i]),
                used_model=used_model,
            )
            for i in range(len(proba))
        ]
        scores.sort(key=lambda s: s.fill_probability, reverse=True)
        return scores

    def score(self, frame: pd.DataFrame) -> pd.Series:
        """Return predicted fill probabilities as a pandas Series."""
        proba, _ = self.predict_with_provenance(frame)
        index = frame.index[: len(proba)]
        return pd.Series(proba, index=index, name="fill_probability")

    # --- evaluation & persistence -----------------------------------------

    def cross_val_auc(self, frame: pd.DataFrame, *, n_splits: int = 5) -> float | None:
        """Stratified k-fold ROC-AUC on the engineered frame (or ``None``).

        Returns ``None`` when the data cannot support a meaningful split. The
        number is computed live -- it is never cached or hard-coded in docs.
        """
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        x, y = self._xy(frame)
        classes, counts = np.unique(y, return_counts=True)
        if classes.size < 2 or counts.min() < 2:
            return None
        splits = int(max(2, min(n_splits, counts.min())))
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self._build_pipeline(), x, y, cv=cv, scoring="roc_auc")
        return float(np.mean(scores))

    def feature_importances(self, frame: pd.DataFrame) -> pd.Series | None:
        """Return trained gradient-boosting feature importances, if fitted."""
        if self._pipeline is None:
            return None
        clf = self._pipeline.named_steps["clf"]
        return pd.Series(
            clf.feature_importances_, index=self.feature_columns, name="importance"
        ).sort_values(ascending=False)

    def save(self, path: str | Path) -> None:
        """Persist the fitted pipeline with joblib."""
        import joblib

        if self._pipeline is None:
            raise RuntimeError("cannot save a model that is in heuristic-fallback state")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "feature_columns": self.feature_columns}, path)

    @classmethod
    def load(cls, path: str | Path) -> FillProbabilityModel:
        """Load a previously saved model."""
        import joblib

        payload = joblib.load(Path(path))
        model = cls(feature_columns=payload.get("feature_columns"))
        model._pipeline = payload["pipeline"]
        return model
