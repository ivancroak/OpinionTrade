"""Shared lightweight fakes for the ML test-suite.

These duck-typed stand-ins mirror the public shape of
``src.opinion_client.Orderbook`` / ``MarketInfo`` / ``OpinionClient`` without
pulling in the trading SDK or any network dependency, so the feature, model and
ranker layers can be tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import pytest


@dataclass
class FakeLevel:
    price: Decimal
    amount: Decimal


@dataclass
class FakeOrderbook:
    bids: list[FakeLevel] = field(default_factory=list)
    asks: list[FakeLevel] = field(default_factory=list)


@dataclass
class FakeMarket:
    market_id: int
    title: str = "fake-market"
    volume_24h: float = 10_000.0
    total_volume: float = 50_000.0
    has_bonus_points: bool = False
    yes_token_id: str = "yes-token"
    no_token_id: str = "no-token"
    question_id: str = "q1"


class FakeClient:
    """Returns canned orderbooks keyed by token id."""

    def __init__(self, books: dict[str, FakeOrderbook | None]) -> None:
        self._books = books
        self.calls: list[tuple[str, str, int]] = []

    def get_orderbook(
        self, token_id: str, question_id: str = "", symbol_type: int = 0
    ) -> FakeOrderbook | None:
        self.calls.append((token_id, question_id, symbol_type))
        return self._books.get(token_id)


def _book(bids: list[tuple[str, str]], asks: list[tuple[str, str]]) -> FakeOrderbook:
    return FakeOrderbook(
        bids=[FakeLevel(Decimal(p), Decimal(a)) for p, a in bids],
        asks=[FakeLevel(Decimal(p), Decimal(a)) for p, a in asks],
    )


@pytest.fixture
def make_orderbook():
    """Factory: ``make_orderbook(bids, asks)`` from ``(price, amount)`` string tuples."""
    return _book


@pytest.fixture
def make_market():
    def _make(market_id: int = 1, **kwargs) -> FakeMarket:
        return FakeMarket(market_id=market_id, **kwargs)

    return _make


@pytest.fixture
def make_client():
    def _make(books: dict[str, FakeOrderbook | None]) -> FakeClient:
        return FakeClient(books)

    return _make
