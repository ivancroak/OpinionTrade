"""Tests for the snapshot corpus generation, IO and live logging hook."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("numpy")

from src.ml.dataset import (  # noqa: E402
    SNAPSHOT_COLUMNS,
    append_snapshot,
    generate_training_frame,
    load_frame,
    write_csv,
)


def test_generate_training_frame_is_deterministic():
    a = generate_training_frame(n_samples=120, seed=11)
    b = generate_training_frame(n_samples=120, seed=11)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == SNAPSHOT_COLUMNS
    assert set(a["filled"].unique()) <= {0, 1}


def test_generate_has_both_classes():
    frame = generate_training_frame(n_samples=400, seed=5)
    assert frame["filled"].nunique() == 2


def test_write_and_load_roundtrip(tmp_path):
    frame = generate_training_frame(n_samples=50, seed=1)
    path = write_csv(frame, tmp_path / "snaps.csv")
    loaded = load_frame(path)
    assert len(loaded) == 50
    assert list(loaded.columns) == SNAPSHOT_COLUMNS


def test_append_snapshot_roundtrip(tmp_path, make_market, make_orderbook):
    path = tmp_path / "live.csv"
    market = make_market(market_id=42, title="m42")
    book = make_orderbook([("0.40", "100"), ("0.39", "40")], [("0.42", "80")])
    append_snapshot(path, market, "YES", book, filled=1)
    append_snapshot(path, market, "NO", book, filled=0)

    df = load_frame(path)
    assert len(df) == 2
    assert list(df.columns) == SNAPSHOT_COLUMNS
    assert list(df["filled"]) == [1, 0]
    assert list(df["side"]) == ["YES", "NO"]
    assert df["market_id"].tolist() == [42, 42]


def test_append_snapshot_writes_single_header(tmp_path, make_market, make_orderbook):
    path = tmp_path / "live.csv"
    market = make_market(market_id=1)
    book = make_orderbook([("0.50", "100")], [("0.52", "100")])
    for _ in range(3):
        append_snapshot(path, market, "YES", book, filled=1)
    text = path.read_text()
    assert text.count("market_id,side,title") == 1
    assert len(load_frame(path)) == 3
