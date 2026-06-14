"""Tests for the scikit-learn fill-probability model."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from src.ml.dataset import generate_training_frame  # noqa: E402
from src.ml.features import build_feature_frame  # noqa: E402
from src.ml.model import (  # noqa: E402
    MIN_TRAIN_SAMPLES,
    FillProbabilityModel,
    RankedMarketScore,
)


@pytest.fixture
def corpus() -> pd.DataFrame:
    """A learnable engineered corpus with both classes well represented."""
    return build_feature_frame(generate_training_frame(n_samples=600, seed=3))


def test_heuristic_predict_in_range(corpus):
    model = FillProbabilityModel()
    proba, used_model = model.predict_with_provenance(corpus)
    assert used_model is False
    assert proba.shape[0] == len(corpus)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_fit_trains_estimator(corpus):
    model = FillProbabilityModel().fit(corpus)
    assert model.is_fitted is True
    proba, used_model = model.predict_with_provenance(corpus)
    assert used_model is True
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_min_samples_guard_keeps_fallback(corpus):
    tiny = corpus.head(MIN_TRAIN_SAMPLES - 1)
    model = FillProbabilityModel().fit(tiny)
    assert model.is_fitted is False


def test_single_class_keeps_fallback(corpus):
    frame = corpus.copy()
    frame["filled"] = 1  # degenerate: only one class
    model = FillProbabilityModel().fit(frame)
    assert model.is_fitted is False


def test_cross_val_auc_beats_chance(corpus):
    auc = FillProbabilityModel(random_state=0).cross_val_auc(corpus)
    assert auc is not None
    assert auc > 0.55  # comfortably above chance; robust to minor sklearn drift


def test_trained_model_beats_heuristic(corpus):
    from sklearn.metrics import roc_auc_score

    from src.ml.features import feature_matrix
    from src.ml.model import _heuristic_fill_probability

    y = corpus["filled"].to_numpy()
    model = FillProbabilityModel().fit(corpus)
    model_auc = roc_auc_score(y, model.predict_proba(corpus))
    heuristic_auc = roc_auc_score(y, _heuristic_fill_probability(feature_matrix(corpus)))
    # Structural invariant (version-robust): learning beats the hand-tuned
    # heuristic in-sample, regardless of the absolute AUC.
    assert model_auc >= heuristic_auc


def test_rank_orders_descending(corpus):
    ranked = FillProbabilityModel().fit(corpus).rank(corpus)
    assert len(ranked) == len(corpus)
    assert all(isinstance(r, RankedMarketScore) for r in ranked)
    probs = [r.fill_probability for r in ranked]
    assert probs == sorted(probs, reverse=True)


def test_score_returns_series(corpus):
    series = FillProbabilityModel().fit(corpus).score(corpus)
    assert isinstance(series, pd.Series)
    assert len(series) == len(corpus)


def test_save_load_roundtrip(tmp_path, corpus):
    model = FillProbabilityModel().fit(corpus)
    path = tmp_path / "fill_model.joblib"
    model.save(path)
    loaded = FillProbabilityModel.load(path)
    assert loaded.is_fitted is True
    np.testing.assert_allclose(model.predict_proba(corpus), loaded.predict_proba(corpus))


def test_save_unfitted_raises(corpus):
    model = FillProbabilityModel()
    with pytest.raises(RuntimeError):
        model.save("/tmp/should-not-write.joblib")


def test_feature_importances_presence(corpus):
    model = FillProbabilityModel()
    assert model.feature_importances(corpus) is None
    model.fit(corpus)
    importances = model.feature_importances(corpus)
    assert importances is not None
    assert importances.sum() == pytest.approx(1.0, abs=1e-6)


def test_empty_frame_predicts_empty():
    model = FillProbabilityModel()
    empty = build_feature_frame([])
    proba, _ = model.predict_with_provenance(empty)
    assert proba.shape[0] == 0
