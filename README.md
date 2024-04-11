![pytest](https://github.com/HireVue-Science/Adverse-Impact-Mitigation/actions/workflows/pytest.yml/badge.svg)
[![codecov](https://codecov.io/gh/HireVue-Science/Adverse-Impact-Mitigation/graph/badge.svg?token=RTWAD5LCVY)](https://codecov.io/gh/HireVue-Science/Adverse-Impact-Mitigation)
![pylint](https://github.com/HireVue-Science/Adverse-Impact-Mitigation/actions/workflows/pylint.yml/badge.svg)
![black&isort](https://github.com/HireVue-Science/Adverse-Impact-Mitigation/actions/workflows/black.yml/badge.svg)

# Adverse Impact Mitigation

This repository is a python implementation of two methods for adverse impact mitigaiton of machine learning models. The first (and best) method is Multipenalty Optimization, which adds a total group differences term to the objective function. The second method is Iterative Predictor Removal, which in a series of steps removes predictors that contribute to group differences.

The Multipenalty Optimization code implements two new scikit-learn-style models: `MPORegressor` and `MPOClassifier`, which add the penalty terms to ridge and l2-penalized logistic regressor. Iterative Predictor Removal works with existing sklearn linear models.

The implementations of Multipenalty Optimization and Iterative Predictor include mitigator classes that implement the mitigation (with either cross validation or a single train/test split). The mitigator classes include methods for evaluating the results.

# Installation

This repository is manually installable using pip:

```
pip install "./path/to/ai_mitigation"
```

For the ability to run all examples and tests, run
```
pip install "./path/to/ai_mitigation[all]"
```

# Examples:

Useful examples in jupyter notebooks can be found in the examples directory. "Adverse Impact Mitigation Demo.ipynb" shows the usage of the mitigator classes (`PredictorRemovalMitigator`, `MultiPenaltyMitigator`, along with their their `CV` and `TrainTest` versions). "MPO Models.ipynb" shows the usage of the `MPORegressor` and `MPORegressor` sklearn-style classes.

# Citation

The paper introducing these methods is:

Rottman, C., Gardner, C., Liff, J., Mondragon, N., & Zuloaga, L. (2023). New strategies for addressing the diversity–validity dilemma with big data. Journal of Applied Psychology. Advance online publication. https://doi.org/10.1037/apl0001084


# Development

To set up a development environment, run

```
bin/setup-localdev
```

This installs `ai_mitigation` in editable/development mode, sets up git hooks (pylint, black, isort), and adds ai_mitigation to your current venv's python .path.
