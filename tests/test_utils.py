import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_mitigation.models import MPOClassifier, MPORegressor
from ai_mitigation.utils import get_cross_validated_scores, is_mpo_model
from ai_mitigation.demo_utils import convert_demo_dicts_to_df

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
