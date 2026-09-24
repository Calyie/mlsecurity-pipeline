"""Save and load models with their preprocessor and a metadata file.

A saved model is a set of files in `model_dir` that share a name:
    <name>.joblib or <name>.pt        the model
    <name>_preprocessor.joblib        the fitted preprocessor, if given
    <name>_metadata.json              when it was saved, what it is, your own notes
"""
import datetime
import json
import os
from pathlib import Path

import joblib
import torch

from .train import ResNetClassifier

MODEL_DIR = str(Path(__file__).resolve().parents[1] / "saved_models")   # at the repository root


def save_model(model, preprocessor=None, metadata=None, model_dir=MODEL_DIR, name=None):
    """Save `model` and return its name. PyTorch models are saved as a state dict."""
    os.makedirs(model_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = name or f"{model.__class__.__name__}_{stamp}"
    metadata = dict(metadata or {})
    metadata["saved_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    metadata["model_class"] = model.__class__.__name__

    if isinstance(model, torch.nn.Module):
        path = os.path.join(model_dir, f"{name}.pt")
        torch.save(model.state_dict(), path)
        metadata["framework"] = "pytorch"
        if isinstance(model, ResNetClassifier):
            metadata["n_classes"] = model.n_classes
            metadata["n_channels"] = model.n_channels
    else:
        path = os.path.join(model_dir, f"{name}.joblib")
        joblib.dump(model, path)
        metadata["framework"] = "sklearn"
    metadata["model_path"] = path

    if preprocessor is not None:
        preprocessor_path = os.path.join(model_dir, f"{name}_preprocessor.joblib")
        joblib.dump(preprocessor, preprocessor_path)
        metadata["preprocessor_path"] = preprocessor_path

    with open(os.path.join(model_dir, f"{name}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved {name} to {model_dir}/")
    return name


def load_model(name, model_dir=MODEL_DIR):
    """Load a saved model by name. Returns (model, preprocessor, metadata)."""
    with open(os.path.join(model_dir, f"{name}_metadata.json")) as f:
        metadata = json.load(f)

    if metadata.get("framework") == "pytorch":
        model = ResNetClassifier(metadata["n_classes"], metadata["n_channels"], pretrained=False)
        model.load_state_dict(torch.load(metadata["model_path"], map_location="cpu"))
        model.eval()
    else:
        model = joblib.load(metadata["model_path"])

    preprocessor = None
    if metadata.get("preprocessor_path"):
        preprocessor = joblib.load(metadata["preprocessor_path"])
    return model, preprocessor, metadata


def list_models(model_dir=MODEL_DIR):
    """Names of the saved models in `model_dir`, oldest first."""
    if not os.path.isdir(model_dir):
        return []
    names = [f[:-len("_metadata.json")] for f in os.listdir(model_dir) if f.endswith("_metadata.json")]
    return sorted(names)
