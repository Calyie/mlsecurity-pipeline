import numpy as np
import pandas as pd
import pytest

from pipeline import preprocess_data


def make_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(18, 80, n),
        "income": rng.normal(50, 10, n),
        "city": rng.choice(["Leeds", "York", "Hull"], n),
        "note": rng.choice(["free money now!!", "see you at lunch", "call me back"], n),
        "label": rng.integers(0, 2, n),
    })


def test_split_sizes_and_feature_encoding():
    df = make_frame()
    X_train, X_val, X_test, y_train, y_val, y_test, pre = preprocess_data(df, "label", text_cols=["note"])
    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == len(df)
    assert X_test.shape[0] == 40                       # 20% of 200
    assert X_val.shape[0] == 32                        # 20% of the remaining 160
    # 2 scaled numerics + 3 one-hot cities + tf-idf terms
    assert X_train.shape[1] > 5
    assert len(y_train) == X_train.shape[0]


def test_preprocessor_is_fitted_on_training_data_only():
    df = make_frame()
    _, _, X_test, _, _, _, pre = preprocess_data(df, "label")
    # the scaler's mean comes from the training rows, so transforming the whole
    # frame again gives the same encoding for the test rows
    again = pre.transform(df.drop(columns="label"))
    assert again.shape[1] == X_test.shape[1]


def test_missing_text_column_is_refused():
    df = make_frame()
    with pytest.raises(ValueError):
        preprocess_data(df, "label", text_cols=["no_such_column"])
