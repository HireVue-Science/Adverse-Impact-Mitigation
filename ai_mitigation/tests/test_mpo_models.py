import numpy as np
import pytest
from scipy.sparse import csc_matrix
from sklearn.base import is_classifier, is_regressor
from sklearn.linear_model import LogisticRegression, Ridge

from ai_mitigation.demo_utils import _create_mask_pairs_from_demo
from ai_mitigation.models import (
    MPOClassifier,
    MPORegressor,
    _calc_costs_log,
    _calc_costs_ridge,
    _calc_grads_log,
    _calc_grads_ridge,
    _cost_bias,
    _cost_likelihood,
    _cost_ols,
    _get_Xy_labeled,
)


def gen_biased_dataset(missing_perf=False, missing_demo=False, n_rows=1000):

    rng = np.random.RandomState(0)

    A = np.array([1.0] * (n_rows // 2) + [0.0] * (n_rows // 2))
    demo_dicts = {"Cat1": {"NotA": A == 0.0, "A": A == 1.0}}
    if missing_demo:
        # Delete half of the demographics
        demo_dicts["Cat1"]["A"][::2] = False
        demo_dicts["Cat1"]["NotA"][::2] = False

    x1 = rng.randn(n_rows)
    x2 = rng.randn(n_rows)
    x3 = rng.randn(n_rows)

    y = x1 + x2 + x3 + 1.1 * rng.randn(n_rows) + 0.3 + 2 * A
    if missing_perf:
        y[::2] = np.nan

    # x1 is biased against A, x3 is biased towards A
    x1 -= 0.5 * A
    x3 += 0.5 * A
    X = np.vstack([x1, x2, x3]).T

    return X, y, demo_dicts


@pytest.mark.parametrize(
    "fit_intercept,C",
    [
        [True, 1.0],
        [True, 0.01],
        [False, 1.0],
        [False, 0.01],
    ],
)
def test_logistic_regression_to_sklearn(fit_intercept, C):
    """if beta == 0, make sure we get the sklearn solution"""

    X, y, demo_dicts = gen_biased_dataset()
    y = y > 0

    sklearn_model = LogisticRegression(
        penalty="l2", C=C, fit_intercept=fit_intercept, solver="lbfgs"
    )

    mpo_model = MPOClassifier(C=C, fit_intercept=fit_intercept, beta=0.0)

    sklearn_model.fit(X, y)
    mpo_model.fit(X, y, demo_dicts)

    assert np.allclose(sklearn_model.coef_, mpo_model.coef_, rtol=1e-03)
    assert np.isclose(sklearn_model.intercept_, mpo_model.intercept_, rtol=1e-03)

    assert np.allclose(sklearn_model.predict_proba(X), mpo_model.predict_proba(X), rtol=1e-03)
    assert np.allclose(
        sklearn_model.predict_log_proba(X), mpo_model.predict_log_proba(X), rtol=1e-03
    )
    assert np.array_equal(sklearn_model.predict(X), mpo_model.predict(X))


@pytest.mark.parametrize(
    "fit_intercept,alpha",
    [
        [True, 1],
        [True, 500],
        [False, 1],
        [False, 500],
    ],
)
def test_ridge_regression_to_sklearn(fit_intercept, alpha):
    """if beta == 0, make sure we get the sklearn solution"""

    X, y, demo_dicts = gen_biased_dataset()

    sklearn_model = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver="cholesky")

    mpo_model = MPORegressor(alpha=alpha, fit_intercept=fit_intercept, beta=0.0)

    sklearn_model.fit(X, y)
    mpo_model.fit(X, y, demo_dicts)

    assert np.allclose(sklearn_model.coef_, mpo_model.coef_)
    assert np.isclose(sklearn_model.intercept_, mpo_model.intercept_)


@pytest.mark.parametrize(
    "fit_intercept,alpha",
    [
        [True, 0],
        [True, 500],
        [False, 0],
        [False, 500],
    ],
)
def test_regressor_sparse(fit_intercept, alpha):
    X, y, demo_dicts = gen_biased_dataset()
    Xsp = csc_matrix(X)

    model1 = MPORegressor(alpha=alpha, fit_intercept=fit_intercept, beta=1.0)
    model2 = MPORegressor(alpha=alpha, fit_intercept=fit_intercept, beta=1.0)
    model1.fit(X, y, demo_dicts)
    model2.fit(Xsp, y, demo_dicts)

    assert np.allclose(model1.coef_, model2.coef_)
    assert np.isclose(model1.intercept_, model2.intercept_)


def test_regressor_decreases_group_differences():
    """if beta == 0, make sure we get the sklearn solution"""

    X, y, demo_dicts = gen_biased_dataset()
    mask1 = demo_dicts["Cat1"]["A"]
    y = y > 0

    group_differences = []
    pro_coefs = []
    neg_coefs = []

    for beta in (0.0, 2.0, 5.0):

        model = MPOClassifier(C=0.01, fit_intercept=True, beta=beta)

        model.fit(X, y, demo_dicts)
        ypred = model.predict_log_proba(X)

        group_differences.append(np.abs(np.mean(ypred[mask1]) - np.mean(ypred[~mask1])))
        pro_coefs.append(model.coef_[2])
        neg_coefs.append(model.coef_[0])

    assert group_differences[0] > group_differences[1] > group_differences[2]
    assert pro_coefs[0] > pro_coefs[1] > pro_coefs[2]
    assert neg_coefs[0] < neg_coefs[1] < neg_coefs[2]


def test_ridge_regression_decreases_group_differences():
    """if beta == 0, make sure we get the sklearn solution"""

    X, y, demo_dicts = gen_biased_dataset()
    mask1 = demo_dicts["Cat1"]["A"]

    group_differences = []
    pro_coefs = []
    neg_coefs = []

    for beta in (0.0, 2.0, 5.0):

        model = MPORegressor(alpha=10, fit_intercept=True, beta=beta)

        model.fit(X, y, demo_dicts)
        ypred = model.predict(X)

        group_differences.append(np.abs(np.mean(ypred[mask1]) - np.mean(ypred[~mask1])))
        pro_coefs.append(model.coef_[2])
        neg_coefs.append(model.coef_[0])

    assert group_differences[0] > group_differences[1] > group_differences[2]
    assert pro_coefs[0] > pro_coefs[1] > pro_coefs[2]
    assert neg_coefs[0] < neg_coefs[1] < neg_coefs[2]


def test_masked_cost_functions():

    X, y, demo_dicts_full = gen_biased_dataset(missing_perf=False, missing_demo=False)
    _, y_with_nans, demo_dicts_sub = gen_biased_dataset(missing_perf=True, missing_demo=True)

    mask_pairs_full = _create_mask_pairs_from_demo(demo_dicts_full)
    mask_pairs_sub = _create_mask_pairs_from_demo(demo_dicts_sub)

    w0 = np.array([-0.5, 1.0, 2.5])
    w1 = np.array([1.0, 1.0, 1.0])

    for dense in (True, False):
        if not dense:
            csc_matrix(X)

        Xlab, ylab = _get_Xy_labeled(X, y_with_nans)

        # performance masking
        ols_cost_nomask = _cost_ols(X, y, w0)
        ols_cost_submask = _cost_ols(Xlab, ylab, w0)

        assert 0.4 * ols_cost_nomask < ols_cost_submask < 0.6 * ols_cost_nomask

        likelihood_cost_nomask = _cost_likelihood(X, y, w0)
        likelihood_cost_submask = _cost_likelihood(Xlab, ylab, w0)

        assert 0.4 * likelihood_cost_nomask < likelihood_cost_submask < 0.6 * likelihood_cost_nomask

        # demographic masking
        bias_cost_fullmask = _cost_bias(X, w0, mask_pairs_full)
        bias_cost_submask = _cost_bias(X, w0, mask_pairs_sub)
        assert 0.4 * bias_cost_fullmask < bias_cost_submask < 0.6 * bias_cost_fullmask

        # bias is greater for worse coefs
        assert _cost_bias(X, w0, mask_pairs_full) > _cost_bias(X, w1, mask_pairs_full)
        assert _cost_bias(X, w0, mask_pairs_sub) > _cost_bias(X, w1, mask_pairs_sub)

        costs = _calc_costs_ridge(X, y_with_nans, w0, True, mask_pairs_sub, 1, 1)
        for cost in costs:
            assert np.all(np.isfinite(cost))
        costs = _calc_costs_log(X, y_with_nans, w0, True, mask_pairs_sub, 1, 1)
        for cost in costs:
            assert np.all(np.isfinite(cost))

        grad = _calc_grads_ridge(X, y_with_nans, w0, True, mask_pairs_sub, 1, 1)
        assert np.all(np.isfinite(grad))
        grad = _calc_grads_log(X, y_with_nans, w0, True, mask_pairs_sub, 1, 1)
        assert np.all(np.isfinite(grad))


def test_models_are_classifiers_and_regressors():
    model_ridge = MPORegressor()
    model_log = MPOClassifier()

    assert is_regressor(model_ridge)
    assert is_classifier(model_log)
    assert not is_classifier(model_ridge)
    assert not is_regressor(model_log)
