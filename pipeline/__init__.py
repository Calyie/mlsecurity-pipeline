"""A small, readable machine-learning pipeline.

Stages: fetch a dataset, load it, preprocess it, train a model, evaluate it,
save it with its preprocessor and metadata, and load it back. Each stage is
one function in one module, so a notebook can use them one at a time.
"""
from .data import fetch_dataset, load_csv, load_images, prepare_kdd, KDD_COLUMNS, KDD_ATTACK_URL
from .preprocess import preprocess_data
from .train import train_model, train_cnn, ResNetClassifier
from .evaluate import predict, evaluate_model, evaluate_cnn
from .registry import save_model, load_model, list_models
from .plots import plot_training, plot_confusion_matrix, plot_2d
from .results import log_result, load_results
from .attacks import flip_labels

SEED = 1337
