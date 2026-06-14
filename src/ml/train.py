"""Train / evaluate the fill-probability model from an orderbook-snapshot corpus.

Usage::

    python -m src.ml.train                      # train on the committed corpus
    python -m src.ml.train --data my.csv        # train on your own snapshots
    python -m src.ml.train --save src/ml/artifacts/fill_model.joblib

The reported ROC-AUC is computed live via stratified cross-validation on the
data provided -- no accuracy figure is hard-coded anywhere in the project.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .dataset import DEFAULT_DATA_PATH, load_frame
from .features import build_feature_frame
from .model import FillProbabilityModel

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fill-probability model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV corpus of orderbook snapshots (default: committed sample).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to persist the fitted model (joblib).",
    )
    parser.add_argument(
        "--random-state", type=int, default=0, help="Random seed for the estimator."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    if not args.data.exists():
        logger.error(
            "corpus not found: %s (run `python -m src.ml.dataset` first)", args.data
        )
        return 1

    raw = load_frame(args.data)
    frame = build_feature_frame(raw)
    n = len(frame)
    fill_rate = (
        float(frame["filled"].mean()) if "filled" in frame.columns else float("nan")
    )

    model = FillProbabilityModel(random_state=args.random_state)
    auc = model.cross_val_auc(frame)
    model.fit(frame)

    logger.info("Fill-probability model")
    logger.info("  corpus            : %s", args.data)
    logger.info("  snapshots         : %d", n)
    logger.info("  fill rate         : %.1f%%", 100 * fill_rate)
    logger.info("  cross-val ROC-AUC : %s", f"{auc:.3f}" if auc is not None else "n/a")
    logger.info(
        "  trained estimator : %s", "yes" if model.is_fitted else "heuristic fallback"
    )

    importances = model.feature_importances(frame)
    if importances is not None:
        logger.info("  top features      :")
        for name, weight in importances.head(5).items():
            logger.info("      %-28s %.3f", name, weight)

    if args.save is not None and model.is_fitted:
        model.save(args.save)
        logger.info("  saved model       : %s", args.save)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
