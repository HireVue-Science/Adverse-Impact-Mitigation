import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.base import clone, is_classifier
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm.auto import tqdm, trange

from . import utils
from .demo_utils import (
    prune_demo_dicts,
    convert_demo_dicts_to_df,
    convert_df_to_demo_dicts,
)
from .mitigator_results import MitigatorResults


class PredictorRemovalMitigatorCV(MitigatorResults):
    def __init__(self, base_model, max_steps=None, predictors_per_step=20, n_splits=5, verbose=1):
        self.base_model = base_model
        self.max_steps = max_steps
        self.predictors_per_step = predictors_per_step
        self.n_splits = n_splits
        self.verbose = verbose

    def run(self, X, y, demo):
        demo_dicts = prune_demo_dicts(convert_df_to_demo_dicts(demo))
        demo = convert_demo_dicts_to_df(demo_dicts, demo.index)
        max_steps = self.max_steps
        if max_steps is None:
            max_steps = (X.shape[1] - 1) // self.predictors_per_step + 1

        mitigators = [
            PredictorRemovalMitigator(
                base_model=self.base_model,
                max_steps=max_steps,
                predictors_per_step=self.predictors_per_step,
                verbose=self.verbose > 1,
            )
            for _ in range(self.n_splits)
        ]

        if is_classifier(self.base_model):
            k_fold = StratifiedKFold(self.n_splits, shuffle=True, random_state=42)
        else:
            k_fold = KFold(self.n_splits, shuffle=True, random_state=42)

        self.k_fold_splits = list((k_fold.split(X, y)))
        if self.verbose > 0:
            iterator = tqdm
        else:
            iterator = list

        y_pred_full = np.zeros((len(y), max_steps))
        for i, (train_index, test_index) in enumerate(iterator(self.k_fold_splits)):
            mitigator = mitigators[i]
            X_train = X[train_index, :]
            y_train = y[train_index]
            demo_train = demo.iloc[train_index, :]

            mitigator.run(X_train, y_train, demo_train)

            X_test = X[test_index, :]
            for j in range(max_steps):
                # Get predictions at every step
                y_pred_full[test_index, j] = mitigator.predict(X_test, j)

        self.mitigators = mitigators
        self.y_pred_full = y_pred_full
        self.y = y
        self.demo = demo

        columns = [str(i * self.predictors_per_step) for i in range(self.y_pred_full.shape[1])]
        self.df_ypred = pd.DataFrame(
            data=self.y_pred_full,
            columns=columns,
        )


class PredictorRemovalMitigator(MitigatorResults):
    def __init__(
        self,
        base_model,
        max_steps=None,
        predictors_per_step=20,
        verbose=1,
        cross_validate=True,
        param_search=False,
        rank_method="weighted",
    ):
        """class for running predictor removal mitigation. parameters:

        base_model: sklearn model that is fit every iteration
        max_steps: total steps to run. if 'None', it's run until the stopping condition
        predictors_per_step: number of predictors that are removed at each iteration
        verbose: 0, 1, or 2; how much info to print while running
        cross_validate (bool): whether to use cross-validated scores for calculating
            cohen's d's (and R scores) for determing which groups to mitigate
        param_search (ridge only): If True, at each iteration it will evaluate the parameter
            alpha/2, and set alpha = alpha/2 if it scores higher (out-of-sample)
        rank_method {'weighted', 'single_group'}. How to rank predictors for removal
            'weighted': use a weighted average of all difference/importance ratios for all
                subgroup pairs (weighted by |d|)
            'single_group': pick a "focal pair" at each iteration with the higheest |d|,
                then rank predictors by the difference/importance ratio of that pair
        """
        self.base_model = base_model
        self.max_steps = max_steps
        self.predictors_per_step = predictors_per_step
        self.verbose = verbose
        self.cross_validate = cross_validate
        self.param_search = param_search
        self.rank_method = rank_method
        assert rank_method in ["weighted", "single_group"]
        assert not (param_search and not cross_validate)

    def run(self, X, y, demo):
        metric = utils.get_model_metric(self.base_model)
        max_steps = self.max_steps
        if max_steps is None:
            max_steps = (X.shape[1] - 1) // self.predictors_per_step + 1

        demo_dicts = convert_df_to_demo_dicts(demo)

        # mitigation_targets = []
        high_groups = []
        low_groups = []
        models = []
        y_pred_full = []

        # Initialize for first pass
        predictor_labels = np.array(list(range(X.shape[1])))
        self.original_predictor_labels = predictor_labels
        X_pruned = X
        removed_predictor_names = []

        range_fun = trange if self.verbose > 0 else range
        model = clone(self.base_model)

        for _ in range_fun(max_steps):
            model = clone(model)

            if self.cross_validate:  # calculate biases from cross validated scores
                y_pred = utils.get_cross_validated_scores(X_pruned, y, model, n_splits=5)
                if self.param_search:
                    model_test = clone(model)
                    model_test.set_params(alpha=model.alpha / 2)
                    y_pred_test = utils.get_cross_validated_scores(
                        X_pruned, y, model_test, n_splits=5
                    )

                    score, _ci = utils.calc_model_score(y, y_pred, metric)
                    score_test, _ci = utils.calc_model_score(y, y_pred_test, metric)
                    if score_test > score:
                        y_pred = y_pred_test
                        model = model_test

                        if self.verbose > 1:
                            print(f"Updating model hyperparameters: {score_test} > {score}")
                            print(model)

                utils.fit_model_with_nans(model, X_pruned, y)
            else:  # calculate biases from in-sample scores
                utils.fit_model_with_nans(model, X_pruned, y)
                y_pred = utils.get_y_pred(model, X_pruned)

            demo_category, low_group, high_group, d = _get_mitigation_targets(y_pred, demo_dicts)
            if self.rank_method == "weighted":
                mitigation_targets = _get_weighted_mitigation_targets(y_pred, demo_dicts)

            if self.verbose > 1:
                score, _ci = utils.calc_model_score(y, y_pred, metric)
                if self.rank_method == "single_group":
                    print(
                        "mitigating {} against {} (Cohens D={:.3f}, {}={:.4f})".format(
                            low_group, high_group, d, metric, score
                        )
                    )
                else:
                    ds = [-_[-1] for _ in mitigation_targets]
                    disp_targets = [
                        f"{g1}-{g2}".replace(" ", "_") for _, g1, g2, _ in mitigation_targets
                    ]
                    disp_targets = " ".join([t for _, t in sorted(zip(ds, disp_targets))])
                    print(
                        (
                            f"mitigating {disp_targets} (Max Cohens D={d:.3f},"
                            f" {metric}={score:.4f})"
                        )
                    )

            if high_groups in low_groups or low_group in high_groups:
                if self.verbose > 1:
                    print("Exiting: Early stopping conditions")
                    removed_predictor_names.pop()  # not using most recent model
                break

            # mitigation worked, keep steps
            low_groups.append(low_group)
            high_groups.append(high_group)
            models.append(model)
            y_pred_full.append(y_pred)

            if self.rank_method == "single_group":
                high_mask = demo_dicts[demo_category][high_group]
                low_mask = demo_dicts[demo_category][low_group]

                # calculate predictors to remove
                predictor_bias = _calc_predictor_bias_ratio(
                    model, X_pruned, predictor_labels, high_mask, low_mask
                )
            else:  # weighted

                predictor_bias = _calc_weighted_predictor_bias_ratio(
                    model, X_pruned, predictor_labels, mitigation_targets, demo_dicts
                )

            # remove predictors
            predictors_to_remove = list(predictor_bias.head(self.predictors_per_step).index)
            if self.verbose > 2:
                print(f"removing predictors {predictors_to_remove}")

            X_pruned, predictor_labels = remove_named_predictors(
                X_pruned, predictor_labels, predictors_to_remove
            )

            removed_predictor_names.append(predictors_to_remove)

        self.demo = demo
        self.y = y
        self.removed_predictor_names = removed_predictor_names
        self.models = models
        y_pred_full = np.array(y_pred_full).T
        columns = [str(i * self.predictors_per_step) for i in range(y_pred_full.shape[1])]
        self.df_ypred = pd.DataFrame(
            data=np.array(y_pred_full),
            columns=columns,
        )

    def predict(self, X, step):
        # use last step
        if step >= len(self.models):
            step = len(self.models) - 1
        model = self.models[step]

        predictors_to_remove = self.removed_predictor_names[:step]
        X_pruned, _ = remove_named_predictors(
            X, self.original_predictor_labels, predictors_to_remove
        )
        return utils.get_y_pred(model, X_pruned)


class PredictorRemovalMitigatorTrainTest(MitigatorResults):
    def __init__(
        self,
        base_model,
        max_steps=None,
        predictors_per_step=20,
        param_search=False,
        cross_validate=True,
        rank_method="weighted",
        verbose=1,
    ):
        self.base_model = base_model
        self.max_steps = max_steps
        self.predictors_per_step = predictors_per_step
        self.verbose = verbose
        self.param_search = param_search
        self.cross_validate = cross_validate
        self.rank_method = rank_method

    def run(self, X_train, y_train, demo_train, X_test, y_test, demo_test):
        dd_train = prune_demo_dicts(convert_df_to_demo_dicts(demo_train))
        demo_train = convert_demo_dicts_to_df(dd_train, demo_train.index)
        dd_test = prune_demo_dicts(convert_df_to_demo_dicts(demo_test))
        demo_test = convert_demo_dicts_to_df(dd_test, demo_test.index)

        mitigator = PredictorRemovalMitigator(
            base_model=self.base_model,
            max_steps=self.max_steps,
            predictors_per_step=self.predictors_per_step,
            cross_validate=self.cross_validate,
            param_search=self.param_search,
            verbose=self.verbose,
            rank_method=self.rank_method,
        )

        mitigator.run(X_train, y_train, demo_train)
        n_steps = len(mitigator.models)
        y_pred_full = np.zeros((len(y_test), n_steps))
        for i in range(n_steps):
            y_pred_full[:, i] = mitigator.predict(X_test, i)

        # we want the results to be calculated on the test set
        self.mitigator = mitigator
        self.y_pred_full = y_pred_full
        self.y = y_test
        self.demo = demo_test

        columns = [str(i * self.predictors_per_step) for i in range(self.y_pred_full.shape[1])]
        self.df_ypred = pd.DataFrame(
            data=self.y_pred_full,
            columns=columns,
        )

    def predict(self, X, step):
        return self.mitigator.predict(X, step)


def remove_named_predictors(X, predictor_labels, predictors_to_remove):
    assert X.shape[1] == len(predictor_labels)
    predictors_to_remove = np.asarray(predictors_to_remove).ravel()
    if not predictors_to_remove.any():
        return X, predictor_labels

    predictor_mask = np.in1d(predictor_labels, predictors_to_remove)

    new_labels = predictor_labels[~predictor_mask]
    X_pruned = X[:, ~predictor_mask]

    return X_pruned, new_labels


def _calc_predictor_bias_ratio(model, X, predictor_labels, mask_high, mask_low, signed=True):
    # Calculate standard deviation - .std() method doesn't exist for sparce matrices
    colstd = _calc_std(X)
    coef = np.squeeze(model.coef_)
    fi = np.abs(colstd * coef)

    # Calculate predictor bias
    high_predictor_mean = np.squeeze(np.array(X[mask_high, :].mean(axis=0)))
    low_predictor_mean = np.squeeze(np.array(X[mask_low, :].mean(axis=0)))

    high_scores = high_predictor_mean * coef
    low_scores = low_predictor_mean * coef

    if signed:
        predictor_bias = high_scores - low_scores
    else:
        predictor_bias = np.abs(high_scores - low_scores)

    zero_mask = fi == 0
    ratio = np.zeros(len(high_scores))
    ratio[~zero_mask] = predictor_bias[~zero_mask] / fi[~zero_mask]

    df = pd.DataFrame(
        index=predictor_labels,
        data={
            "predictor_importance": fi,
            "predictor_bias": predictor_bias,
            "ratio": ratio,
        },
    )
    return df.sort_values("ratio", ascending=False)


def _calc_weighted_predictor_bias_ratio(model, X, predictor_labels, mitigation_targets, demo_dicts):
    # Calculate standard deviation - .std() method doesn't exist for sparce matrices
    colstd = _calc_std(X)
    coef = np.squeeze(model.coef_)
    fi = np.abs(colstd * coef)

    total_predictor_bias = np.zeros_like(coef)

    for demo, low_group, high_group, weight in mitigation_targets:
        mask_high = demo_dicts[demo][high_group]
        mask_low = demo_dicts[demo][low_group]

        # Calculate predictor bias
        high_predictor_mean = np.squeeze(np.array(X[mask_high, :].mean(axis=0)))
        low_predictor_mean = np.squeeze(np.array(X[mask_low, :].mean(axis=0)))

        high_scores = high_predictor_mean * coef
        low_scores = low_predictor_mean * coef

        predictor_bias = weight * (high_scores - low_scores)  # maybe want weight^2?

        total_predictor_bias += predictor_bias

    zero_mask = fi == 0
    ratio = np.zeros(len(high_scores))
    ratio[~zero_mask] = total_predictor_bias[~zero_mask] / fi[~zero_mask]

    df = pd.DataFrame(
        index=predictor_labels,
        data={
            "predictor_importance": fi,
            "predictor_bias": total_predictor_bias,
            "ratio": ratio,
        },
    )
    return df.sort_values("ratio", ascending=False)


def _get_mitigation_targets(y_pred, demo_dicts):
    worst_bias = 1

    for demo_category, demo_dict in demo_dicts.items():
        cohens_d_dict = utils.calc_cohens_d_pairwise(y_pred, demo_dict).d
        worst_pair_category = min(cohens_d_dict, key=cohens_d_dict.get)  # argmin
        worst_bias_category = cohens_d_dict[worst_pair_category]

        if worst_bias_category < worst_bias:
            worst_bias = worst_bias_category
            worst_category = demo_category
            worst_pair = worst_pair_category

    return worst_category, worst_pair[0], worst_pair[1], worst_bias


def _get_weighted_mitigation_targets(y_pred, demo_dicts):
    """Returns a list of mitigation targets where each entry is:

    (demo_category, low_scoring_subgroup, high_scoring_subgroup, |d|)

    """
    mitigation_targets = []

    for demo, demo_dict in demo_dicts.items():
        ds = []
        cohens_d_dict = utils.calc_cohens_d_reference(y_pred, demo_dict)
        reference_group = cohens_d_dict.reference_group
        for subgroup, d in cohens_d_dict.d.items():
            if subgroup == reference_group:
                continue
            ds.append(d)
            mitigation_targets.append((demo, subgroup, reference_group, np.abs(d)))
    return mitigation_targets


def _calc_std(X):
    if not issparse(X):
        return np.squeeze(X.std(axis=0))
    X = X.copy()
    colmeansq = np.squeeze(np.array(X.mean(axis=0)) ** 2)
    X.data **= 2
    colsqmean = np.squeeze(np.array(X.mean(axis=0)))
    colstd = np.sqrt(np.clip(colsqmean - colmeansq, 0, None))
    return colstd
