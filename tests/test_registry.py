import numpy as np
import torch
from sklearn.datasets import make_blobs

from pipeline import ResNetClassifier, list_models, load_model, preprocess_data, save_model, train_model
import pandas as pd


def test_sklearn_round_trip_with_preprocessor_and_metadata(tmp_path):
    X, y = make_blobs(n_samples=120, centers=2, random_state=0)
    df = pd.DataFrame(X, columns=["a", "b"]); df["label"] = y
    X_train, _, X_test, y_train, _, y_test, pre = preprocess_data(df, "label")
    model = train_model("logistic_regression", X_train, y_train)
    name = save_model(model, pre, {"note": "test"}, model_dir=str(tmp_path), name="lr_test")
    assert name == "lr_test" and list_models(str(tmp_path)) == ["lr_test"]

    loaded, loaded_pre, metadata = load_model("lr_test", model_dir=str(tmp_path))
    assert metadata["note"] == "test" and metadata["framework"] == "sklearn"
    np.testing.assert_array_equal(loaded.predict(X_test), model.predict(X_test))
    np.testing.assert_allclose(loaded_pre.transform(df[["a", "b"]]), pre.transform(df[["a", "b"]]))


def test_pytorch_round_trip_rebuilds_the_same_network(tmp_path):
    torch.manual_seed(0)
    model = ResNetClassifier(n_classes=3, n_channels=1, pretrained=False).eval()
    x = torch.randn(2, 1, 64, 64)
    before = model(x)
    name = save_model(model, model_dir=str(tmp_path), name="cnn_test")
    loaded, _, metadata = load_model(name, model_dir=str(tmp_path))
    assert metadata["framework"] == "pytorch" and metadata["n_classes"] == 3
    torch.testing.assert_close(loaded(x), before)


def test_list_models_on_a_missing_folder_is_empty(tmp_path):
    assert list_models(str(tmp_path / "nothing_here")) == []
