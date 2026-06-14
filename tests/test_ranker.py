"""Tests for the ML market/side ranking seam."""

from __future__ import annotations

import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from src.ml.model import FillProbabilityModel, RankedMarketScore  # noqa: E402
from src.ml.ranker import (  # noqa: E402
    default_model,
    format_ranking,
    rank_markets_by_fill_probability,
)


def _market_universe(make_market, make_orderbook):
    markets = [
        make_market(
            market_id=i, title=f"m{i}", yes_token_id=f"yes{i}", no_token_id=f"no{i}"
        )
        for i in range(1, 6)
    ]
    books = {}
    for i in range(1, 6):
        # Vary spread/liquidity so candidates are genuinely differentiable.
        books[f"yes{i}"] = make_orderbook(
            [(f"0.{40 + i}", f"{100 * i}"), ("0.30", "20")],
            [(f"0.{45 + i}", f"{80 * i}")],
        )
        books[f"no{i}"] = make_orderbook(
            [("0.50", f"{50 * i}")], [("0.56", f"{50 * i}")]
        )
    return markets, books


def test_rank_returns_sorted_scores(make_client, make_market, make_orderbook):
    markets, books = _market_universe(make_market, make_orderbook)
    client = make_client(books)
    scores = rank_markets_by_fill_probability(client, markets, top_n=10)
    assert scores, "expected non-empty ranking"
    assert all(isinstance(s, RankedMarketScore) for s in scores)
    probs = [s.fill_probability for s in scores]
    assert probs == sorted(probs, reverse=True)


def test_rank_respects_top_n(make_client, make_market, make_orderbook):
    markets, books = _market_universe(make_market, make_orderbook)
    client = make_client(books)
    scores = rank_markets_by_fill_probability(client, markets, top_n=3)
    assert len(scores) == 3


def test_use_model_false_is_heuristic(make_client, make_market, make_orderbook):
    markets, books = _market_universe(make_market, make_orderbook)
    client = make_client(books)
    scores = rank_markets_by_fill_probability(client, markets, use_model=False)
    assert all(s.used_model is False for s in scores)


def test_reused_model_is_applied(make_client, make_market, make_orderbook):
    markets, books = _market_universe(make_market, make_orderbook)
    client = make_client(books)
    model = default_model()
    scores = rank_markets_by_fill_probability(client, markets, model=model)
    assert scores
    assert all(s.used_model is model.is_fitted for s in scores)


def test_empty_when_no_books(make_client, make_market):
    market = make_market(market_id=1, yes_token_id="a", no_token_id="b")
    client = make_client({"a": None, "b": None})
    assert rank_markets_by_fill_probability(client, [market]) == []


def test_default_model_trains_on_corpus():
    assert default_model().is_fitted is True


def test_format_ranking_renders(make_client, make_market, make_orderbook):
    markets, books = _market_universe(make_market, make_orderbook)
    client = make_client(books)
    scores = rank_markets_by_fill_probability(client, markets, top_n=3)
    rendered = format_ranking(scores)
    assert "P(fill)" in rendered
    assert "No rankable markets" in format_ranking([])


def test_isolated_model_predicts(make_client, make_market, make_orderbook):
    markets, books = _market_universe(make_market, make_orderbook)
    client = make_client(books)
    scores = rank_markets_by_fill_probability(
        client, markets, model=FillProbabilityModel(), use_model=True
    )
    assert scores
