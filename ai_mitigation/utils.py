import itertools

import numpy as np
import scipy
from scipy.sparse import csc_matrix, issparse
from scipy.stats import norm, pearsonr
from sklearn.base import is_classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from .demo_utils import calc_valid_groups, convert_demo_dicts_to_df, convert_series_to_demo_dict
from .models import MPOClassifier, MPORegressor


class CohensDResults:
    def __init__(self):
        self.d = None
        self.d_ci = None


def get_cross_validated_scores(
    X, y, model, demo=None, n_splits=5, shuffle=True, random_state=42, verbose=False
):
    """gets out of sample scores for either regular sklearn models or MPO models"""
    if isinstance(demo, dict):
        demo = convert_demo_dicts_to_df(demo, list(range(len(y))))
    if is_classifier(model) and np.all(np.isfinite(y)):
        k_fold = StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state)
    else:
        k_fold = KFold(n_splits, shuffle=shuffle, random_state=random_state)
    y_pred = np.zeros_like(y, dtype=float)

    iterator = tqdm if verbose else list
    # Run through each fold and score.
    for train_index, test_index in iterator(list(k_fold.split(X, y))):
        x_train, x_test = X[train_index, :], X[test_index, :]
        if issparse(x_train):
            x_train = csc_matrix(x_train)
            x_test = csc_matrix(x_test)
        y_train = y[train_index]
        if is_mpo_model(model):
            assert demo is not None, f"{model} cannot fit without demo"
            demo_train = demo.iloc[train_index, :]
            if isinstance(model, Pipeline):
                model.fit(x_train, y_train, model__demo=demo_train)
            else:
                model.fit(x_train, y_train, demo_train)
        else:
            assert demo is None
            fit_model_with_nans(model, x_train, y_train)
        y_pred[test_index] = get_y_pred(model, x_test)

    return y_pred


def get_y_pred(model, X):
    """Get "raw scores" from a model. For a regressor, this just uses .predict.
    For a logistic regression, this is just the linear term (aX+b), before being
    sent through the sigmoid function
    """
    if hasattr(model, "predict_raw"):
        y_pred = model.predict_raw(X)
    elif is_classifier(model):
        log_probs = model.predict_log_proba(X)
        log_probs[np.isinf(log_probs)] = -1e300
        y_pred = log_probs[:, 1] - log_probs[:, 0]
    else:
        y_pred = model.predict(X)
    return y_pred


def get_model_metric(model):
    if is_classifier(model):
        return "AUC"
    return "R"


def calc_auc_with_ci(y, y_pred, alpha=0.05):
    """returns [AUC, confidence_interval] where the confidence interval is the
    two-sided confidence interval [CI_alpha/2, CI_(1-alpha/2)].
    see Hanley/MacNeil 1982."""
    auc = roc_auc_score(y, y_pred)
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    n1 = sum(y)
    n2 = len(y) - sum(y)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2)) / (n1 * n2)
    se = np.sqrt(var)
    se_mult = scipy.stats.t.ppf(1 - alpha / 2.0, df=len(y))
    CI = [auc - se_mult * se, auc + se_mult * se]
    CI[0] = max(0.0, CI[0])  # threshold at [0, 1]
    CI[1] = min(1.0, CI[1])
    return auc, CI


def calc_r_with_ci(y, y_pred, alpha=0.05):
    """returns [R, confidence_interval] where the confidence interval is the
    two-sided confidence interval [CI_alpha/2, CI_(1-alpha/2)]."""
    r = pearsonr(y, y_pred)[0]
    n = len(y)
    if n <= 3:
        return r, [-1, 1]
    if r == 1:
        return 1, [1, 1]

    z = np.log((1 + r) / (1 - r)) / 2.0
    se = 1.0 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2.0)  # 2-tailed z critical value

    conf_low = z - z_crit * se
    conf_high = z + z_crit * se

    def _z_to_r(z):
        e = np.exp(2 * z)
        return (e - 1) / (e + 1)

    CI = [_z_to_r(conf_low), _z_to_r(conf_high)]
    return r, CI


def calc_model_score(y, y_pred, metric, alpha=0.05):
    """calculates the masked model score. For classifiers, calculates
    the AUC. For regressors, calculates the Pearson's R"""
    assert metric in {"R", "AUC"}, f"unknown metric '{metric}': valid metrics are 'R' and 'AUC'"
    real_mask = np.isfinite(y)
    y = np.asarray(y)[real_mask]
    y_pred = np.asarray(y_pred)[real_mask]
    if metric == "AUC":
        return calc_auc_with_ci(y, y_pred, alpha)
    return calc_r_with_ci(y, y_pred, alpha)


def calc_cohens_d_pairwise(scores, demo):
    """calculates the one vs. one cohen's Ds for all pairs"""
    demo_dict = convert_series_to_demo_dict(demo)

    info = CohensDResults()
    info.d = {}
    info.d_ci = {}

    for group1, group2 in itertools.combinations(demo_dict.keys(), 2):
        d, se = _cohens_d(scores[demo_dict[group1]], scores[demo_dict[group2]])

        info.d[(group1, group2)] = d
        info.d_ci[(group1, group2)] = [d - se * 1.96, d + se * 1.96]
        info.d[(group2, group1)] = -d
        info.d_ci[(group2, group1)] = [-d - se * 1.96, -d + se * 1.96]

    return info


def _cohens_d(x1, x2):
    """returns Cohen's D and standard error between two lists of scores"""
    n1 = len(x1)
    n2 = len(x2)
    if n1 <= 1 or n2 <= 1:
        return np.nan, np.nan
    s1 = np.std(x1, ddof=1)
    s2 = np.std(x2, ddof=1)
    dof = n1 + n2 - 2
    s = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / dof)
    d = (np.mean(x1) - np.mean(x2)) / s
    se = np.sqrt(((n1 + n2) / (n1 * n2) + d**2 / (2 * dof)) * ((n1 + n2) / dof))
    return d, se


def fit_model_with_nans(model, X, y):
    real_mask = np.isfinite(y)
    assert np.sum(real_mask)
    model.fit(X[real_mask, :], y[real_mask])


def calc_cohens_d_reference(scores, demo, reference_group=None, ref_min_size=20, ref_min_frac=0.02):
    """calculates the one vs. reference cohen's D for all demo groups.  if
    reference group isn't defined, chooses the highest scoring
    reference group
    """
    demo_dict = convert_series_to_demo_dict(demo)
    info = CohensDResults()
    info.d = {}
    info.d_ci = {}

    if reference_group is None:
        valid_ref_groups = calc_valid_groups(demo_dict, ref_min_size, ref_min_frac)
        if not valid_ref_groups:
            valid_ref_groups = demo_dict.keys()  # just use max scoring group if none are valid
        all_means = {group: np.mean(scores[demo_dict[group]]) for group in valid_ref_groups}
        reference_group = max(all_means, key=all_means.get)  # argmax
    info.reference_group = reference_group

    assert reference_group in demo_dict

    for comparison_group in demo_dict.keys():
        d, se = _cohens_d(scores[demo_dict[comparison_group]], scores[demo_dict[reference_group]])

        info.d[comparison_group] = d
        info.d_ci[comparison_group] = [d - se * 1.96, d + se * 1.96]

    return info


def is_mpo_model(model):
    if isinstance(model, (MPOClassifier, MPORegressor)):
        return True
    if isinstance(model, Pipeline):
        return is_mpo_model(model.steps[-1][1])
    return False
