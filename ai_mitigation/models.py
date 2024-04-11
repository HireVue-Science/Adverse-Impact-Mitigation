import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .demo_utils import _create_mask_pairs_from_demo


class MPORegressor(BaseEstimator, RegressorMixin):
    """
    Multipenalty Optimized Ridge Regression

    minimizes ||y - Xw||^2_2 + alpha * ||w||^2_2 + beta * sum_i(group_difference_i)

    Parameters
    ----------

    alpha (float): Strength of the l2 regularization term (same as scikit-learn's Ridge)

    beta (float): Strength of the group differences term. A single group difference
    term is defined as the squared difference between a single demographic group's mean
    score and the overall mean score (for all valid labels)

    For details, see

    Rottman, C., Gardner, C., Liff, J., Mondragon, N., & Zuloaga, L. (2023, April 10). New Strategies for Addressing the Diversity–Validity Dilemma With Big Data. Journal of Applied Psychology. Advance online publication. https://dx.doi.org/10.1037/apl0001084
    """

    def __init__(self, alpha=1.0, beta=1.0, fit_intercept=True, solver="newton_cg"):
        self.alpha = alpha
        self.beta = beta
        self.fit_intercept = fit_intercept
        self.solver = solver

    def fit(self, X, y, demo):
        """
        Fit Multipenalty Optimized Ridge.

        Parameters
        ----------

        X: training matrix (n_samples, n_features)

        y: target data (n_samples)

        demo: demographics data. Can be a DataFrame with demographic columns starting with
        'demo_' and demographic values (i.e. demo_Gender: ['Male', 'Female', '', 'Male', ...].
        It also can  be a demographic dictionary of boolean masks, i.e.
           {'Gender':
             {
               'Male': [True, False, False, True, ...],
               'Female': [False, True, False, False, ...]
             }
           }
        """

        mask_pairs = _create_mask_pairs_from_demo(demo)
        assert mask_pairs, "No demographics found"
        y = np.array(y, "float")
        assert X.shape[0] == len(y)
        for mask_pair in mask_pairs:
            assert X.shape[0] == len(mask_pair[0]) == len(mask_pair[1])

        if self.fit_intercept:
            X_fit = _add_const_to_X(X)
        else:
            X_fit = X

        def f_norm(w):
            return sum(
                _calc_costs_ridge(
                    X_fit, y, w, self.fit_intercept, mask_pairs, self.alpha, self.beta
                )
            )

        def f_grad(w):
            return _calc_grads_ridge(
                X_fit, y, w, self.fit_intercept, mask_pairs, self.alpha, self.beta
            )

        w = np.zeros(X_fit.shape[1])
        assert self.solver in {
            "newton_cg",
            "lbfgs",
        }, f"Unknown solver '{self.solver}'. Choose from 'lbfgs' or 'newton_cg'"
        if self.solver == "lbfgs":
            out = minimize(
                f_norm, w, jac=f_grad, method="L-BFGS-B", options={"gtol": 1e-3, "maxiter": 100}
            )
        else:  # self.solver == "newton_cg"
            out = minimize(f_norm, w, jac=f_grad, method="Newton-CG", options={"xtol": 5e-6})
        if self.fit_intercept:
            self.coef_ = out.x[:-1]
            self.intercept_ = out.x[-1]
        else:
            self.coef_ = out.x
            self.intercept_ = 0.0
        self.out_ = out

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


class MPOClassifier(BaseEstimator, ClassifierMixin):
    """
    Multipenalty Optimized L2 logistic regression

    minimizes log likelihood + alpha * ||w||^2_2 + beta * sum_i(group_difference_i)

    Parameters
    ----------

    C (float): Inverse strength of the l2 regularization term (same as scikit-learn's LogisticRegression)

    beta (float): Strength of the group differences term. A single group difference
    term is defined as the squared difference between a single demographic group's mean
    score and the overall mean score (for all valid labels). Mean score here is the linear
    term of the logistic regression (before sending it through the sigmoid)

    For details, see

    Rottman, C., Gardner, C., Liff, J., Mondragon, N., & Zuloaga, L. (2023, April 10). New Strategies for Addressing the Diversity–Validity Dilemma With Big Data. Journal of Applied Psychology. Advance online publication. https://dx.doi.org/10.1037/apl0001084
    """

    def __init__(self, C=1, beta=1, fit_intercept=True, max_iter=100, tol=1e-4):
        self.C = C
        self.beta = beta
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y, demo):
        """
        Fit Multipenalty Optimized l2-Logistic Regression.

        Parameters
        ----------

        X: training matrix (n_samples, n_features)

        y: target data (n_samples)

        demo: demographics data. Can be a DataFrame with demographic columns starting with
        'demo_' and demographic values (i.e. demo_Gender: ['Male', 'Female', '', 'Male', ...].
        It also can  be a demographic dictionary of boolean masks, i.e.
           {'Gender':
             {
               'Male': [True, False, False, True, ...],
               'Female': [False, True, False, False, ...]
             }
           }
        """
        mask_pairs = _create_mask_pairs_from_demo(demo)
        assert mask_pairs, "No demographics found"
        y = np.array(y, "float")
        perf_mask = np.isfinite(y)
        assert np.array_equal(np.unique(y[perf_mask]), [0, 1]), np.unique(y[perf_mask])
        y[perf_mask] = y[perf_mask] * 2 - 1  # optimization requires y \in [-1, 1]
        assert np.array_equal(np.unique(y[perf_mask]), [-1, 1]), np.unique(y[perf_mask])

        if self.fit_intercept:
            X_fit = _add_const_to_X(X)
        else:
            X_fit = X

        def f_norm(w):
            return sum(
                _calc_costs_log(X_fit, y, w, self.fit_intercept, mask_pairs, self.C, self.beta)
            )

        def f_grad(w):
            return _calc_grads_log(X_fit, y, w, self.fit_intercept, mask_pairs, self.C, self.beta)

        w = np.zeros(X_fit.shape[1])
        out = minimize(
            f_norm,
            w,
            jac=f_grad,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )
        if self.fit_intercept:
            self.coef_ = out.x[:-1]
            self.intercept_ = out.x[-1]
        else:
            self.coef_ = out.x
            self.intercept_ = 0.0
        self.out_ = out

    def predict_raw(self, X):
        """returns the linear model output (i.e. pre-sigmoid). This is a 1-D log prability
        of the positive class"""
        return X @ self.coef_ + self.intercept_

    def predict_log_proba(self, X):
        """returns the linear model output (i.e. pre-sigmoid). This is a 1-D log prability
        of the positive class"""
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        """returns 0.0-1.0 probability (1-d array) of the positive class"""
        pos_prob = _sigmoid(self.predict_raw(X))
        return np.array([1 - pos_prob, pos_prob]).T

    def predict(self, X):
        """returns {0.0, 1.0} binary predictions"""
        return 1.0 * (self.predict_raw(X) > 0)


def _add_const_to_X(X):
    """adds a final column of ones to X, for when fit_intercept is True. This isn't exactly memory
    efficient, but is a convenient implementation that works for any solver"""
    one_vec = np.ones((X.shape[0], 1))
    if sparse.issparse(X):
        X_fit = sparse.hstack([X, one_vec])
    else:
        X_fit = np.hstack([X, one_vec])
    return X_fit


def _calc_costs_ridge(X, y, w, fit_intercept, mask_pairs, alpha, beta):
    """full cost of MPO ridge"""
    return [
        _cost_ols(X, y, w),
        alpha * _cost_reg(w, fit_intercept),
        beta * _cost_bias(X, w, mask_pairs),
    ]


def _calc_costs_log(X, y, w, fit_intercept, mask_pairs, C, beta):
    """full cost of MPO l2 logistic regression"""
    return [
        _cost_likelihood(X, y, w),
        0.5 / C * _cost_reg(w, fit_intercept),
        0.5 * beta * _cost_bias(X, w, mask_pairs),
    ]


def _calc_grads_ridge(X, y, w, fit_intercept, mask_pairs, alpha, beta):
    """Full gradient for MPO Ridge"""
    return (
        _grad_ols(X, y, w)
        + alpha * _grad_reg(w, fit_intercept)
        + beta * _grad_bias(X, w, mask_pairs)
    )


def _calc_grads_log(X, y, w, fit_intercept, mask_pairs, C, beta):
    """Full gradient for MPO l2 Logistic Regression"""
    return (
        _grad_likelihood(X, y, w)
        + 1.0 / C * _grad_reg(w, fit_intercept)
        + beta * _grad_bias(X, w, mask_pairs)
    )


def _cost_ols(X, y, w):
    """returns SSE cost for linear regression"""
    y = np.asarray(y, "float")
    mask = np.isfinite(y)
    if not np.all(mask):
        mask = np.asarray(mask, dtype="bool")
        if sparse.issparse(X):
            X = X.tocsr()
        X = X[mask, :]
        y = y[mask]
    return np.sum(np.asarray((X @ w - y)) ** 2)


def _grad_likelihood(X, y, w):
    """gradient of the likelihood term"""
    y = np.asarray(y, "float")
    mask = np.isfinite(y)
    if not np.all(mask):
        mask = np.asarray(mask, dtype="bool")
        if sparse.issparse(X):
            X = X.tocsr()
        X = X[mask, :]
        y = y[mask]
    w = np.asarray(w, "float")
    return -X.T @ (y * (_sigmoid(-y * (X @ w))))


def _cost_reg(w, fit_intercept):
    """returns cost of regularization term"""
    w = np.asarray(w)
    if fit_intercept:
        return w[:-1] @ w[:-1]
    return w @ w


def _cost_bias(X, w, mask_pairs):
    """mask pairs is a list of [(demo_mask, cat_mask), ...]
    where demo_mask is the people who are a specific demographic (i.e. female)
    and cat_mask is people with a valid demographic in that category (i.e. Gender)

    "bias" is shorthand for the sum of group differences
    """
    w = np.asarray(w)
    cost = 0.0
    for demo_mask, cat_mask in mask_pairs:
        demo_mask = np.asarray(demo_mask, bool)
        cat_mask = np.asarray(cat_mask, bool)
        z = (1.0 / np.sum(demo_mask) * demo_mask - 1.0 / np.sum(cat_mask) * cat_mask) @ X
        cost += np.sum(cat_mask) * (np.asarray(z @ w) ** 2).squeeze()
    return cost


def _grad_ols(X, y, w):
    """gradient of the OLS term"""
    y = np.asarray(y, "float")
    mask = np.isfinite(y)
    if not np.all(mask):
        mask = np.asarray(mask, dtype="bool")
        if sparse.issparse(X):
            X = X.tocsr()
        X = X[mask, :]
        y = y[mask]
    w = np.asarray(w, "float")
    return X.T @ (X @ w - y)


def _grad_reg(w, fit_intercept):
    """gradient of the regularization term"""
    w = np.asarray(w, "float")
    if fit_intercept:
        return np.append(w[:-1], [0.0])
    return w


def _grad_bias(X, w, mask_pairs):
    """gradient of the bias cost function"""
    w = np.asarray(w)
    grad = np.zeros(w.shape)
    for demo_mask, cat_mask in mask_pairs:
        z = (1.0 / np.sum(demo_mask) * demo_mask - 1.0 / np.sum(cat_mask) * cat_mask) @ X
        grad += np.sum(cat_mask) * z * (z @ w)
    return grad


def _sigmoid(v):
    """sigmoid for logistic regression, returns number between -1 and 1"""
    v = np.asarray(v, dtype="float")
    idx = v > 0
    out = np.empty(v.size, dtype="float")
    out[idx] = 1.0 / (1.0 + np.exp(-v)[idx])
    exp_v = np.exp(v[~idx])
    out[~idx] = exp_v / (1.0 + exp_v)
    return out


def _log_sigmoid(v):
    v = np.asarray(v, dtype="float")
    idx = v > 0
    out = np.empty(v.size, dtype="float")
    out[idx] = v[idx] - np.log(1 + np.exp(v[idx]))
    out[~idx] = -np.log(1 + np.exp(-v[~idx]))
    return out


def _cost_likelihood(X, y, w):
    """returns log likelihood cost for logistic regression"""
    mask = np.isfinite(y)
    if not np.all(mask):
        if sparse.issparse(X):
            X = X.tocsr()
        mask = np.asarray(mask, dtype="bool")
        X = X[mask, :]
        y = y[mask]
    return -np.sum(_log_sigmoid(y * (X @ w)))
