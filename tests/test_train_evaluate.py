import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_regression

from pipeline import evaluate_model, predict, train_model


def test_classifiers_learn_a_separable_problem():
    X, y = make_blobs(n_samples=300, centers=[(0, 5), (5, 0)], cluster_std=1.0, random_state=0)
    for algorithm in ("logistic_regression", "random_forest"):
        model = train_model(algorithm, X, y)
        metrics = evaluate_model(model, X, y)
        assert metrics["accuracy"] > 0.95, algorithm
        assert set(metrics) == {"accuracy", "precision", "recall", "f1", "report"}
        predictions, probabilities = predict(model, X)
        assert predictions.shape == (300,) and probabilities.shape == (300, 2)


def test_regressors_return_regression_metrics():
    X, y = make_regression(n_samples=300, n_features=4, noise=1.0, random_state=0)
    for algorithm in ("linear", "random_forest_regressor"):
        model = train_model(algorithm, X, y)
        metrics = evaluate_model(model, X, y)
        assert set(metrics) == {"rmse", "mae", "r2"}
        assert metrics["r2"] > 0.9, algorithm


def test_naive_bayes_on_counts():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 5, size=(200, 10))
    y = (X[:, 0] + X[:, 1] > 4).astype(int)
    model = train_model("naive_bayes", X, y)
    assert evaluate_model(model, X, y)["accuracy"] > 0.7


def test_unknown_algorithm_is_refused():
    with pytest.raises(ValueError):
        train_model("gradient_boosting", np.zeros((4, 2)), np.zeros(4))
