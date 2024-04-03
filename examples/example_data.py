import numpy as np
import pandas as pd


def gen_example_dataset():
    n_examples = 8000
    n_examples_test = 5000
    n_predictors = 11
    rng = np.random.RandomState(0)

    demo_df = pd.DataFrame(data={"demo_Gender": rng.choice(["Male", "Female"], n_examples)})
    demo_df_test = pd.DataFrame(
        data={"demo_Gender": rng.choice(["Male", "Female"], n_examples_test)}
    )
    male_mask = demo_df.demo_Gender.values == "Male"
    male_mask_test = demo_df_test.demo_Gender.values == "Male"

    predictors = list(range(n_predictors))
    predictor_diffs = np.linspace(-0.2, 0.4, n_predictors)

    y_nodiff = rng.randn(n_examples)
    y_nodiff_test = rng.randn(n_examples_test)
    y = y_nodiff + 0.4 * male_mask
    y_test = y_nodiff_test + 0.4 * male_mask_test

    X = np.zeros((n_examples, n_predictors))
    X_test = np.zeros((n_examples_test, n_predictors))
    for i, predictor_diff in enumerate(predictor_diffs):
        X[:, i] = y_nodiff + 6 * rng.randn(n_examples) + 6 * predictor_diff * male_mask
        X_test[:, i] = (
            y_nodiff_test + 6 * rng.randn(n_examples_test) + 6 * predictor_diff * male_mask_test
        )

    data = {
        "X": X,
        "X_test": X_test,
        "y": y,
        "y_test": y_test,
        "demo": demo_df,
        "demo_test": demo_df_test,
    }
    return data
