"""Plots used by the notebooks. Plain matplotlib, one figure per call."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

COLOURS = ["#2a78d6", "#d1541f", "#0f8f68", "#a8770a", "#c4306b"]


def plot_training(history):
    """Accuracy and loss per epoch from `train_cnn`'s history."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(epochs, history["accuracy"], marker="o", color=COLOURS[0])
    ax1.set_title("Training accuracy"); ax1.set_xlabel("epoch"); ax1.set_ylim(0, 1)
    ax2.plot(epochs, history["loss"], marker="o", color=COLOURS[1])
    ax2.set_title("Training loss"); ax2.set_xlabel("epoch")
    fig.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """A confusion matrix with the counts written in each cell."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ticks = labels if labels is not None else np.unique(np.concatenate([y_true, y_pred]))
    ax.set_xticks(range(len(ticks))); ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_yticks(range(len(ticks))); ax.set_yticklabels(ticks)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title("Confusion matrix")
    fig.tight_layout()
    plt.show()


def plot_2d(X, y, title="", model=None, highlight=None, ax=None):
    """Scatter a 2-feature dataset coloured by class.

    `model` adds its decision boundary; `highlight` marks a set of row indices
    (for example, the rows whose labels an attacker flipped).
    """
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(6, 4.5))
    if model is not None:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
        zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], colors=COLOURS[:2], alpha=0.15)
    for label in np.unique(y):
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], s=18, color=COLOURS[int(label) % len(COLOURS)],
                   label=f"class {label}", alpha=0.8)
    if highlight is not None and len(highlight):
        ax.scatter(X[highlight, 0], X[highlight, 1], s=60, facecolors="none",
                   edgecolors="black", linewidths=1.2, label="flipped label")
    ax.set_title(title); ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax.legend(loc="best", fontsize=8)
    if own_figure:
        fig.tight_layout()
        plt.show()
