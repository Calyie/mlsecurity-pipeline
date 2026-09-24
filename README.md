# mlops-pipeline

A small, reusable machine-learning pipeline in Python: download a dataset, load it, preprocess it,
train a model, evaluate it, save it with its preprocessor and metadata, and load it back. The same
`main()` call runs a price regression, a spam classifier, a network-intrusion classifier and an
image classifier, so the stages are written once and the dataset is a parameter.

Alongside the pipeline, the notebook holds security experiments on the models it trains: how a
classifier degrades when a share of its training labels are flipped (data poisoning), with the
decision boundary plotted at each poisoning rate.

## Layout

| Path | What it is |
|---|---|
| `main.ipynb` | The notebook: the `main()` orchestrator, one call per dataset, and the security experiments |
| `utils.py` | The pipeline stages as functions, imported by the notebook |
| `requirements.txt` | Pinned dependencies (CPU builds of PyTorch) |
| `downloads/` | Datasets fetched by `fetch_dataset()`; created on first run, not committed |
| `saved_models/` | Models, preprocessors and metadata written by `save_model()`; not committed |

## The stages

| Stage | Function in `utils.py` | What it does |
|---|---|---|
| Fetch | `fetch_dataset(url, zipped)` | Downloads over HTTPS only, extracts a zip into `downloads/<name>/` |
| Load | `load_data(path, ...)` | A CSV into a DataFrame, or an image folder into train and test `DataLoader`s with the given mean and standard deviation |
| Preprocess | `preprocess_data(df, y_col_name, text_cols, ...)` | Train, validation and test split; one-hot encoding for categoricals, scaling for numerics, a bag of words for text columns; returns the fitted preprocessor |
| Train | `train_model(...)` | `random_forest` (randomised search over depth, leaf size and tree count, macro F1), `random_forest_regressor`, `linear`, `logistic_regression`, `naive_bayes` (grid search over `alpha`), or `cnn` (a ResNet classifier trained with Adam and cross-entropy) |
| Evaluate | `evaluate_model(model, X, y)`, `evaluate_nn(model, loader)` | Accuracy, precision, recall, F1 and a confusion matrix for classifiers; MSE, MAE and R² for regressors |
| Save and load | `save_model(...)`, `load_model_with_metadata(...)` | Writes `<Model>_<timestamp>.joblib`, the preprocessor beside it and a JSON metadata file (timestamp, paths, model type); loads a model by name with its preprocessor |
| Serve | `upload_model_result()` | Posts a saved model to a local model-serving API (`http://localhost:8000/api/upload`) |

`main()` in the notebook chains the stages. Give it the algorithm, the data location (a URL or a
local path), the target column and, for text, the text columns; for image data give `dataset_type="image"`,
the normalisation statistics and the epoch count.

## The datasets in the notebook

| Cell | Dataset | Task | Model |
|---|---|---|---|
| Diamonds | seaborn's diamonds table | price regression | random forest regressor |
| SMS Spam Collection | UCI | spam or ham from the message text | multinomial naive Bayes, grid-searched |
| KDD | network connection records | attack category (DoS, probe, privilege, access, normal) | random forest, randomised search |
| Malimg | malware images | malware family | ResNet classifier |

The KDD cell maps the raw attack names to four categories before training (`prepare_kdd`).

## Security experiments

The label-flipping experiment trains a logistic-regression baseline on a two-cluster synthetic
dataset, then retrains it with 5% to 50% of the training labels flipped, records the accuracy at
each rate, and plots the decision boundaries together so the drift is visible. Further experiments
on image classifiers (MNIST and GTSRB) are kept as their own notebooks in this repository.

## Install and run

Python 3.10 or later.

```bash
git clone https://github.com/Calyie/mlops-pipeline
cd mlops-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
jupyter lab main.ipynb
```

`requirements.txt` pins CPU builds of PyTorch and torchvision; drop the extra index URL for a GPU
build. The plotting style comes from `htb-ai-library`, which is installed from GitHub by the same
command. The Malimg cell downloads from Kaggle, which needs a Kaggle API token in `~/.kaggle/`.

Every random seed is fixed at 1337 (`numpy`, `torch` and the scikit-learn searches), so a rerun
reproduces the reported numbers on the same versions.

## Licence

MIT.
