import numpy as np
import pytest
from scipy.sparse import csc_matrix
from sklearn.base import is_classifier
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from ai_mitigation.models import MPOClassifier, MPORegressor
from ai_mitigation.mpo_mitigation import (
    MultiPenaltyMitigator,
    MultiPenaltyMitigatorCV,
    MultiPenaltyMitigatorTrainTest,
    _get_base_model,
)
from ai_mitigation.utils import is_mpo_model

from .example_data import gen_example_data


@pytest.mark.parametrize(
    "input_model",
    [
        Pipeline([("model", Ridge())]),
        Pipeline([("model", LogisticRegression())]),
        Ridge(),
        MPORegressor(),
        Pipeline([("model", MPOClassifier())]),
    ],
)
def test_get_base_model_returns_valid_model(input_model):
    mpo_model = _get_base_model(input_model)
    assert is_mpo_model(mpo_model)
    assert is_classifier(input_model) == is_classifier(mpo_model)


@pytest.mark.parametrize(
    "input_model",
    [
        Pipeline([("model", Lasso())]),
        Pipeline([("model", LogisticRegression(penalty="elasticnet"))]),
        Lasso(),
        LogisticRegression(penalty="elasticnet"),
    ],
)
def test_get_base_model_invalid_model(input_model):
    with pytest.raises(RuntimeError):
        _get_base_model(input_model)


def test_mpo_mitigator():
    data = gen_example_data()
    mitigator = MultiPenaltyMitigator(Ridge(), betas=[0, 1])
    mitigator.run(X=data["X"], y=data["y"], demo=data["demo"])
    results = mitigator.calc_results([("Male", "Female")])
    results = mitigator.calc_results([("Male", "Female")])
    results = mitigator.calc_results_with_ci([("Male", "Female")])


def test_mpo_mitigator_classifier():
    data = gen_example_data()
    mitigator = MultiPenaltyMitigator(LogisticRegression(), betas=[0, 1])
    y = data["y"] > np.mean(data["y"])
    X = csc_matrix(data["X"])
    mitigator.run(X=X, y=y, demo=data["demo"])
    results = mitigator.calc_results([("Male", "Female")])
    results = mitigator.calc_results([("Male", "Female")])
    results = mitigator.calc_results_with_ci([("Male", "Female")])


def test_mpo_mitigator_cv():
    data = gen_example_data()
    mitigator = MultiPenaltyMitigatorCV(Ridge(), betas=[0, 1])
    mitigator.run(X=data["X"], y=data["y"], demo=data["demo"])
    results = mitigator.calc_results([("Male", "Female")])
    results = mitigator.calc_results_with_ci([("Male", "Female")])


def test_mpo_mitigator_train_test():
    data = gen_example_data()
    mitigator = MultiPenaltyMitigatorTrainTest(Ridge(), betas=[0, 1])
    mitigator.run(
        X_train=data["X"],
        y_train=data["y"],
        demo_train=data["demo"],
        X_test=data["X_test"],
        y_test=data["y_test"],
        demo_test=data["demo_test"],
    )
    results = mitigator.calc_results([("Male", "Female")])
    results = mitigator.calc_results_with_ci([("Male", "Female")])
