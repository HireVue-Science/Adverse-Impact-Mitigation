import numpy as np
from scipy.sparse import csc_matrix
from sklearn.linear_model import LogisticRegression, Ridge

from ai_mitigation.predictor_removal_mitigation import (
    PredictorRemovalMitigator,
    PredictorRemovalMitigatorCV,
    PredictorRemovalMitigatorTrainTest,
)

from .example_data import gen_example_data


def test_pr_mitigator():
    data = gen_example_data()
    mitigator = PredictorRemovalMitigator(Ridge(), predictors_per_step=1, param_search=True)
    mitigator.run(X=data["X"], y=data["y"], demo=data["demo"])

    mitigator.calc_results([("Male", "Female")])
    mitigator.calc_results_with_ci([("Male", "Female")])
    mitigator.predict(data["X"], 1)


def test_pr_mitigator__nonstandard_options():
    data = gen_example_data()
    mitigator = PredictorRemovalMitigator(
        LogisticRegression(),
        predictors_per_step=1,
        verbose=3,
        rank_method="single_group",
    )
    X = csc_matrix(data["X"])
    y = data["y"] > np.mean(data["y"])
    mitigator.run(X=X, y=y, demo=data["demo"])

    mitigator.calc_results([("Male", "Female")])
    mitigator.calc_results_with_ci([("Male", "Female")])
    mitigator.predict(data["X"], 1)


def test_pr_mitigator_cv():
    data = gen_example_data()
    mitigator = PredictorRemovalMitigatorCV(Ridge(), predictors_per_step=1, verbose=0)
    mitigator.run(X=data["X"], y=data["y"], demo=data["demo"])

    mitigator.calc_results([("Male", "Female")])
    mitigator.calc_results_with_ci([("Male", "Female")])


def test_pr_mitigator_train_test():
    data = gen_example_data()
    mitigator = PredictorRemovalMitigatorTrainTest(Ridge(), predictors_per_step=1)
    mitigator.run(
        X_train=data["X"],
        y_train=data["y"],
        demo_train=data["demo"],
        X_test=data["X_test"],
        y_test=data["y_test"],
        demo_test=data["demo_test"],
    )

    mitigator.calc_results([("Male", "Female")])
    mitigator.calc_results_with_ci([("Male", "Female")])
    mitigator.predict(data["X_test"], 1)
