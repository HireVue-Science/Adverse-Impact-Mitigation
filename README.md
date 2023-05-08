# Adverse Impact Mitigation

This repository is a python implementation of two methods for adverse impact mitigaiton of machine learning models. The first (and best) method is Multipenalty Optimization, which adds a total group differences term to the objective function. The second method is Iterative Predictor Removal, which removes predictors that contribute to group differences.

The Multipenalty Optimization code implements two new scikit-learn-style models: `MPORegressor` and `MPOClassifier`, which add the penalty terms to ridge and l2-penalized logistic regressor. Iterative Predictor Removal works with existing sklearn linear models.

The implementations of Multipenalty Optimization and Iterative Predictor include mitigator classes that implement the mitigation (with either cross validation or a single train/test split). The mitigator classes include methods for evaluating the results.


# Installation

This repository is manually installable using pip:

```
pip install "./path/to/ai_mitigation"
```

To run the examples and tests, add the `[all]` option the pip install command.

# Citation

The paper introducing these methods is:

Rottman, C., Gardner, C., Liff, J., Mondragon, N., & Zuloaga, L. (2023). New strategies for addressing the diversity–validity dilemma with big data. Journal of Applied Psychology. Advance online publication. https://doi.org/10.1037/apl0001084
