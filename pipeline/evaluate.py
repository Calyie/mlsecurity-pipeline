"""Score a trained model on held-out data."""
import numpy as np
import torch
from sklearn.base import is_regressor
from sklearn.metrics import (accuracy_score, classification_report, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)


def predict(model, X):
    """Return (predictions, probabilities) for a scikit-learn model; probabilities may be None."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return predictions, probabilities


def evaluate_model(model, X, y):
    """Return a dict of metrics: RMSE, MAE and R2 for a regressor; accuracy,
    precision, recall, F1 (weighted) and a text report for a classifier."""
    y_pred, _ = predict(model, X)
    if is_regressor(model):
        return {"rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                "mae": float(mean_absolute_error(y, y_pred)),
                "r2": float(r2_score(y, y_pred))}
    return {"accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
            "report": classification_report(y, y_pred, zero_division=0)}


def evaluate_cnn(model, loader, device=None):
    """Accuracy (0 to 1) of a PyTorch classifier over a DataLoader."""
    device = device or next(model.parameters()).device
    model.eval()
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predicted = model(inputs).argmax(1)
            n_correct += (predicted == labels).sum().item()
            n_total += labels.size(0)
    accuracy = n_correct / max(n_total, 1)
    print(f"Test accuracy: {accuracy:.3f}")
    return accuracy


def print_metrics(metrics):
    """Print a metrics dict, one line per number, the report last."""
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
    if "report" in metrics:
        print(metrics["report"])
