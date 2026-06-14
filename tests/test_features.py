"""Tests for orderbook-microstructure feature engineering."""

from __future__ import annotations

import math

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from src.ml import features  # noqa: E402
from src.ml.features import (  # noqa: E402
    FEATURE_COLUMNS,
    build_feature_frame,
    extract_features,
    feature_matrix,
    orderbook_features,
)


def test_orderbook_features_basic(make_orderbook):
    book = make_orderbook(
        bids=[("0.40", "100"), ("0.39", "50")],
        asks=[("0.42", "80"), ("0.43", "60")],
    )
    f = orderbook_features(book)

    assert f["best_bid"] == pytest.approx(0.40)
    assert f["spread"] == pytest.approx(0.02)
    assert f["book_depth_levels"] == 2.0
    # bid liq = 0.40*100 + 0.39*50 = 59.5 ; ask liq = 0.42*80 + 0.43*60 = 59.4
    assert f["total_liquidity"] == pytest.approx(59.5 + 59.4)
    assert -1.0 <= f["depth_imbalance"] <= 1.0
    assert 0.0 < f["liquidity_concentration"] <= 1.0


def test_orderbook_features_empty_is_nan():
    f = orderbook_features(None)
    assert math.isnan(f["spread"])
    assert math.isnan(f["depth_imbalance"])
    assert math.isnan(f["liquidity_concentration"])
    assert f["book_depth_levels"] == 0.0
    assert f["total_liquidity"] == 0.0


def test_concentration_single_level_is_one(make_orderbook):
    book = make_orderbook(bids=[("0.50", "100")], asks=[("0.51", "100")])
    assert orderbook_features(book)["liquidity_concentration"] == pytest.approx(1.0)


def test_concentration_uniform_levels(make_orderbook):
    # Four levels of equal USD value -> HHI = 4 * (1/4)^2 = 0.25.
    book = make_orderbook(
        bids=[("0.50", "100"), ("0.50", "100"), ("0.50", "100"), ("0.50", "100")],
        asks=[("0.51", "100")],
    )
    assert orderbook_features(book)["liquidity_concentration"] == pytest.approx(0.25)


def test_depth_imbalance_sign(make_orderbook):
    bid_heavy = make_orderbook(bids=[("0.50", "1000")], asks=[("0.51", "10")])
    assert orderbook_features(bid_heavy)["depth_imbalance"] > 0
    ask_heavy = make_orderbook(bids=[("0.50", "10")], asks=[("0.51", "1000")])
    assert orderbook_features(ask_heavy)["depth_imbalance"] < 0


def test_build_feature_frame_has_all_feature_columns():
    rows = [
        {
            "market_id": 1,
            "side": "YES",
            "title": "m1",
            "best_bid": 0.4,
            "spread": 0.02,
            "total_liquidity": 1000.0,
            "volume_24h": 5000.0,
            "depth_imbalance": 0.1,
            "liquidity_concentration": 0.5,
            "book_depth_levels": 6.0,
            "has_bonus_points": 1.0,
            "filled": 1,
        }
    ]
    frame = build_feature_frame(rows)
    for col in FEATURE_COLUMNS:
        assert col in frame.columns
    assert "filled" in frame.columns
    assert {"market_id", "side", "title"} <= set(frame.columns)


def test_build_feature_frame_derived_values():
    rows = [
        {
            "best_bid": 0.40,
            "spread": 0.02,
            "total_liquidity": 1000.0,
            "volume_24h": 5000.0,
            "depth_imbalance": 0.0,
            "liquidity_concentration": 0.5,
            "book_depth_levels": 6.0,
            "has_bonus_points": 0.0,
        }
    ]
    frame = build_feature_frame(rows)
    # efficiency = volume / liquidity = 5000 / 1000 = 5.
    assert frame["volume_liquidity_efficiency"].iloc[0] == pytest.approx(5.0)
    # relative_spread = spread / (best_bid + spread/2) = 0.02 / 0.41.
    assert frame["relative_spread"].iloc[0] == pytest.approx(0.02 / 0.41)
    assert frame["log_total_liquidity"].iloc[0] == pytest.approx(np.log1p(1000.0))


def test_build_feature_frame_handles_missing_columns():
    frame = build_feature_frame([{"best_bid": 0.5}])
    assert len(frame) == 1
    # Missing raw inputs become NaN rather than raising.
    assert math.isnan(frame["volume_liquidity_efficiency"].iloc[0])


def test_build_feature_frame_empty():
    frame = build_feature_frame([])
    assert list(frame.columns) == FEATURE_COLUMNS
    assert frame.empty


def test_feature_matrix_shape_and_order():
    rows = [
        {
            "best_bid": 0.4,
            "spread": 0.02,
            "total_liquidity": 1000.0,
            "volume_24h": 5000.0,
            "depth_imbalance": 0.0,
            "liquidity_concentration": 0.5,
            "book_depth_levels": 6.0,
            "has_bonus_points": 0.0,
        }
    ]
    matrix = feature_matrix(build_feature_frame(rows))
    assert matrix.shape == (1, len(FEATURE_COLUMNS))
    assert matrix.dtype == float


def test_extract_features_one_row_per_side(make_client, make_market, make_orderbook):
    market = make_market(market_id=7, yes_token_id="yes7", no_token_id="no7")
    client = make_client(
        {
            "yes7": make_orderbook([("0.40", "100")], [("0.42", "100")]),
            "no7": make_orderbook([("0.55", "100")], [("0.58", "100")]),
        }
    )
    frame = extract_features(client, [market])
    assert len(frame) == 2
    assert set(frame["side"]) == {"YES", "NO"}


def test_extract_features_skips_missing_books(make_client, make_market, make_orderbook):
    market = make_market(market_id=8, yes_token_id="yes8", no_token_id="no8")
    client = make_client(
        {"yes8": make_orderbook([("0.40", "100")], [("0.42", "100")]), "no8": None}
    )
    frame = extract_features(client, [market])
    assert len(frame) == 1
    assert frame["side"].iloc[0] == "YES"


def test_extract_features_empty_returns_empty_frame(make_client, make_market):
    market = make_market(market_id=9, yes_token_id="x", no_token_id="y")
    client = make_client({"x": None, "y": None})
    assert extract_features(client, [market]).empty


def test_module_exposes_column_contracts():
    assert "spread" in features.FEATURE_COLUMNS
    assert "depth_imbalance" in features.FEATURE_COLUMNS
    assert "liquidity_concentration" in features.FEATURE_COLUMNS
    assert "volume_liquidity_efficiency" in features.FEATURE_COLUMNS
