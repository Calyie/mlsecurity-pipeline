"""Train a scikit-learn model or a small ResNet classifier."""
import time

import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB

ALGORITHMS = ("random_forest", "random_forest_regressor", "linear", "logistic_regression", "naive_bayes")


def train_model(algorithm, X, y, seed=1337, **params):
    """Train one of the ALGORITHMS on (X, y) and return the fitted model.

    random_forest and naive_bayes run a small hyper-parameter search with
    cross-validation; the others use `params` directly.
    """
    print(f"Training {algorithm}...")
    if algorithm == "random_forest":
        grid = {"n_estimators": [200, 300], "max_depth": [20, None], "min_samples_leaf": [1, 2]}
        search = RandomizedSearchCV(RandomForestClassifier(n_jobs=-1, class_weight="balanced", random_state=seed),
                                    grid, n_iter=6, scoring="f1_macro", cv=3, random_state=seed)
        search.fit(X, y)
        print("Best parameters:", search.best_params_)
        return search.best_estimator_
    if algorithm == "random_forest_regressor":
        params.setdefault("n_estimators", 100)
        params.setdefault("max_depth", 10)
        model = RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    elif algorithm == "linear":
        model = LinearRegression(**params)
    elif algorithm == "logistic_regression":
        params.setdefault("max_iter", 1000)
        model = LogisticRegression(random_state=seed, **params)
    elif algorithm == "naive_bayes":
        grid = {"alpha": [0.001, 0.01, 0.1, 0.25, 0.5, 1.0]}
        search = GridSearchCV(MultinomialNB(), grid, cv=5, scoring="f1_macro")
        search.fit(X, y)
        print("Best parameters:", search.best_params_)
        return search.best_estimator_
    else:
        raise ValueError(f"Unknown algorithm {algorithm!r}; choose one of {ALGORITHMS}")
    model.fit(X, y)
    print("Training complete")
    return model


class ResNetClassifier(nn.Module):
    """A ResNet-50 backbone with a new classification head.

    The backbone is frozen unless `fine_tune=True`, which unfreezes its last
    block. `n_channels=1` adapts the first layer to grey-scale images.
    `pretrained=False` skips the weight download (used by the tests).
    """

    def __init__(self, n_classes, n_channels=3, fine_tune=False, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)
        if n_channels == 1:
            old = self.model.conv1
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.conv1.weight.data = old.weight.data.mean(dim=1, keepdim=True)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        if fine_tune:
            for parameter in self.model.layer4.parameters():
                parameter.requires_grad = True
        n_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(n_features, 1000), nn.ReLU(), nn.Dropout(0.3), nn.Linear(1000, n_classes))
        self.n_classes = n_classes
        self.n_channels = n_channels

    def forward(self, x):
        return self.model(x)


def train_cnn(train_loader, n_classes, n_channels, n_epochs=10, device=None, lr=1e-3,
              fine_tune=False, pretrained=True):
    """Train a ResNetClassifier with Adam and cross-entropy.

    Returns (model, history) where history holds the accuracy and loss per epoch.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier(n_classes, n_channels, fine_tune=fine_tune, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    history = {"accuracy": [], "loss": []}

    for epoch in range(n_epochs):
        model.train()
        start = time.time()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_correct += (outputs.argmax(1) == labels).sum().item()
            n_total += labels.size(0)
        accuracy = n_correct / n_total
        history["accuracy"].append(accuracy)
        history["loss"].append(total_loss / len(train_loader))
        print(f"epoch {epoch + 1}/{n_epochs}: accuracy {accuracy:.3f}, loss {history['loss'][-1]:.4f} "
              f"({time.time() - start:.0f} s)")
    return model, history
