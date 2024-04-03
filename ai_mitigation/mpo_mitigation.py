import copy

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from . import utils
from .demo_utils import (
    convert_demo_dicts_to_df,
    convert_df_to_demo_dicts,
    prune_demo,
    prune_demo_dicts,
)
from .mitigator_results import MitigatorResults
from .models import MPOClassifier, MPORegressor


def _get_base_model(input_model):
    """Returns the appropriate 'base' (alpha=0) model for a given input Ridge/LogisticRegression/MPO Model"""
    if isinstance(input_model, (MPORegressor, MPOClassifier)):
        return input_model
    if isinstance(input_model, Ridge):
        return MPORegressor(
            beta=0, alpha=input_model.alpha, fit_intercept=input_model.fit_intercept
        )
    if isinstance(input_model, LogisticRegression):
        if input_model.penalty != "l2":
            raise RuntimeError("LogiticRegression is only allowed using an l2 penalty")
        return MPOClassifier(beta=0, C=input_model.C, fit_intercept=input_model.fit_intercept)
    if isinstance(input_model, Pipeline):
        base_model = copy.deepcopy(input_model)
        base_model.steps[-1] = (base_model.steps[-1][0], _get_base_model(input_model.steps[-1][-1]))
        return base_model
    raise RuntimeError(f"Unable to convert model '{input_model}' to a valid MPO model")


class MultiPenaltyMitigator(MitigatorResults):
    def __init__(self, sklearn_model, betas, verbose=1):

        self.base_model = _get_base_model(sklearn_model)

        if betas is None:
            betas = [0, 1, 3, 10, 30]
        self.betas = betas
        self.verbose = verbose

    def run(self, X, y, demo):
        demo = prune_demo(demo)
        demo = convert_demo_dicts_to_df(demo, list(range(len(y))))

        self.demo = demo
        self.y = y
        self.models = []

        iterator = tqdm if self.verbose > 0 else list

        self.df_ypred = pd.DataFrame()
        for beta in iterator(self.betas):
            model = clone(self.base_model)
            model.beta = beta

            model.fit(X, y, demo)
            self.df_ypred[str(beta)] = model.predict(X)
            self.models.append(model)


class MultiPenaltyMitigatorCV(MitigatorResults):
    def __init__(self, sklearn_model, betas=None, n_splits=10, verbose=1):

        self.base_model = _get_base_model(sklearn_model)

        if betas is None:
            betas = [0, 1, 3, 10, 30]
        self.betas = betas
        self.n_splits = n_splits
        self.verbose = verbose

    def run(self, X, y, demo):
        demo_dicts = prune_demo(convert_df_to_demo_dicts(demo))
        demo = convert_demo_dicts_to_df(demo_dicts, demo.index)

        self.demo = demo
        self.y = y

        iterator = tqdm if self.verbose > 0 else list

        self.df_ypred = pd.DataFrame()
        for beta in iterator(self.betas):
            model = clone(self.base_model)
            model.beta = beta

            self.df_ypred[str(float(beta))] = utils.get_cross_validated_scores(
                X, y, model, demo, n_splits=self.n_splits, verbose=self.verbose > 1
            )


class MultiPenaltyMitigatorTrainTest(MitigatorResults):
    def __init__(self, sklearn_model, betas=None, verbose=1):
        self.base_model = _get_base_model(sklearn_model)

        if betas is None:
            betas = [0, 1, 3, 10, 30]
        self.betas = betas
        self.verbose = verbose

    def run(self, X_train, y_train, demo_train, X_test, y_test, demo_test):
        dd_train = prune_demo(convert_df_to_demo_dicts(demo_train))
        demo_train = convert_demo_dicts_to_df(dd_train, demo_train.index)
        dd_test = prune_demo_dicts(convert_df_to_demo_dicts(demo_test))
        demo_test = convert_demo_dicts_to_df(dd_test, demo_test.index)

        self.demo = demo_test
        self.y = y_test
        self.models = []

        iterator = tqdm if self.verbose > 0 else list

        self.df_ypred = pd.DataFrame()
        for beta in iterator(self.betas):
            model = clone(self.base_model)
            model.beta = beta

            model.fit(X_train, y_train, demo_train)

            self.df_ypred[str(float(beta))] = model.predict(X_test)
            self.models.append(model)
