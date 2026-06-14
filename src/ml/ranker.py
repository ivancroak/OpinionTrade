"""ML-driven market/side ranking by predicted fill probability.

This is the integration seam that lets the learned policy *replace the static
``volume / liquidity`` efficiency heuristic* used elsewhere in market selection.
Given a live client and a list of markets, it engineers orderbook-microstructure
features for each (market, side), scores them with a
:class:`~src.ml.model.FillProbabilityModel`, and returns the candidates ordered
by predicted fill probability.

It is intentionally decoupled from the trading SDK: ``client`` and ``market``
objects are duck-typed, so the seam is unit-testable with lightweight fakes. To
wire it into the interactive menu, call :func:`rank_markets_by_fill_probability`
from ``select_market_interactive`` in ``src.market_selection``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

from .dataset import load_default_training_frame
from .features import build_feature_frame, extract_features
from .model import FillProbabilityModel, RankedMarketScore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..opinion_client import MarketInfo, OpinionClient

logger = logging.getLogger(__name__)


def default_model() -> FillProbabilityModel:
    """Build a model trained on the committed corpus (heuristic if it can't)."""
    model = FillProbabilityModel()
    try:
        corpus = build_feature_frame(load_default_training_frame())
        model.fit(corpus)
    except Exception as exc:  # noqa: BLE001 - never let training break ranking
        logger.warning("fill-probability model training failed; using heuristic: %s", exc)
    return model


def rank_markets_by_fill_probability(
    client: OpinionClient,
    markets: Iterable[MarketInfo],
    *,
    top_n: int = 10,
    sides: tuple[str, ...] = ("YES", "NO"),
    use_model: bool = True,
    model: FillProbabilityModel | None = None,
) -> list[RankedMarketScore]:
    """Rank (market, side) candidates by predicted fill probability.

    Args:
        client: Anything exposing ``get_orderbook(token_id, question_id, symbol_type)``.
        markets: Markets to score (need ``yes_token_id``/``no_token_id`` etc.).
        top_n: Number of top candidates to return.
        sides: Which sides to consider per market.
        use_model: When ``False``, force the deterministic heuristic (no training).
        model: Pre-built/pre-trained model to reuse; built on demand otherwise.

    Returns:
        Up to ``top_n`` :class:`RankedMarketScore` objects, highest probability first.
    """
    features = extract_features(client, markets, sides=sides)
    if features.empty:
        return []

    if model is None:
        model = default_model() if use_model else FillProbabilityModel()

    ranked = model.rank(features)
    return ranked[:top_n]


def format_ranking(scores: list[RankedMarketScore]) -> str:
    """Render a human-readable table of ranked candidates."""
    if not scores:
        return "  No rankable markets found."
    # Invariant: all rows from a single rank() call share the same provenance.
    provenance = "model" if scores[0].used_model else "heuristic"
    lines = [f"  Predicted fill probability ({provenance}):"]
    for rank, score in enumerate(scores, start=1):
        lines.append(
            f"  {rank:>2}. [{score.side:<3}] P(fill)={score.fill_probability:6.1%}  {score.title}"
        )
    return "\n".join(lines)
