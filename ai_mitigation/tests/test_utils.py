import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_mitigation.demo_utils import convert_demo_dicts_to_df
from ai_mitigation.models import MPOClassifier, MPORegressor
from ai_mitigation.utils import (
    _cohens_d,
    calc_auc_with_ci,
    calc_r_with_ci,
    get_cross_validated_scores,
    is_mpo_model,
)

from .test_mpo_models import gen_biased_dataset


@pytest.mark.parametrize(
    "model,binarize",
    [
        [Pipeline([("scaler", StandardScaler()), ("model", MPORegressor())]), False],
        [Pipeline([("scaler", StandardScaler()), ("model", MPOClassifier())]), True],
        [MPORegressor(), False],
        [MPOClassifier(), True],
    ],
)
def test_cross_validation_with_demo(model, binarize):
    X, y, demo_dicts = gen_biased_dataset(missing_perf=True, missing_demo=True, n_rows=240)
    df = convert_demo_dicts_to_df(demo_dicts)
    df = df.rename(columns={"Cat1": "demo_Cat1"})

    if binarize:
        mask = np.isfinite(y)
        y[mask] = y[mask] > 0
    df["y"] = y
    df["other_column"] = np.linspace(0, 1, len(df))

    get_cross_validated_scores(X, df.y, model, demo=df, n_splits=2)


@pytest.mark.parametrize(
    "model,binarize",
    [
        [Pipeline([("scaler", StandardScaler()), ("model", Ridge())]), False],
        [Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]), True],
        [Ridge(), False],
        [LogisticRegression(), True],
    ],
)
def test_cross_validation_without_demo(model, binarize):
    X, y, _demo_dicts = gen_biased_dataset(missing_perf=True, missing_demo=True, n_rows=240)
    if binarize:
        mask = np.isfinite(y)
        y[mask] = y[mask] > 0
    get_cross_validated_scores(X, y, model, n_splits=2)


@pytest.mark.parametrize(
    "model,is_mpo",
    [
        [Pipeline([("model", Ridge())]), False],
        [Pipeline([("model", LogisticRegression())]), False],
        [Ridge(), False],
        [MPORegressor(), True],
        [Pipeline([("model", MPOClassifier())]), True],
    ],
)
def test_is_mpo_model(model, is_mpo):
    assert is_mpo_model(model) == is_mpo


def test_auc_with_ci():
    auc, CI = calc_auc_with_ci([0, 0, 1], [0.1, 0.2, 0.3])
    assert auc == 1
    assert CI == [1, 1]

    auc, CI = calc_auc_with_ci([0] * 3 + [1] * 6 + [0] * 3, range(12))
    assert auc == 0.5
    assert np.allclose(CI, [0.12203661777408908, 0.877963382225911])


def test_r_with_ci():
    r, CI = calc_r_with_ci([0, 1, 2, 3], [10, 20, 30, 40])
    assert r == 1
    assert CI == [1, 1]

    r, CI = calc_r_with_ci([0, 1, 2], [10, 20, 30])
    assert np.isclose(r, 1)
    assert CI == [-1, 1]

    r, CI = calc_r_with_ci([0, 1, 2, 3], [10, 20, 10, 20])
    assert np.isclose(r, 0.447213595499958)
    assert np.allclose(CI, [-0.9012339349268834, 0.984955683958303])


def test_cohens_d():
    d, se = _cohens_d([1], [2, 3, 4, 5])
    assert np.isnan(d)
    assert np.isnan(se)

    d, se = _cohens_d([1, 2, 3, 4], [2, 3, 4, 5])
    assert np.isclose(d, -0.7745966692414834)
    assert np.isclose(se, 0.8563488385776753)
