from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns


def create_prediction_grid(X, padding=0.5, points=100):
    """Create a 2-D grid covering the supplied feature matrix."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(
            "Decision-boundary plots require a feature matrix with two columns."
        )
    if points < 2:
        raise ValueError(
            "The prediction grid must contain at least two points per axis."
        )

    x_min, y_min = X.min(axis=0) - padding
    x_max, y_max = X.max(axis=0) + padding
    XX, YY = np.mgrid[
        x_min : x_max : complex(points),
        y_min : y_max : complex(points),
    ]
    return np.column_stack((XX.ravel(), YY.ravel())), XX, YY


def plot_decision_boundary(
    X, y, plot_title, file_name=None, XX=None, YY=None, preds=None
):
    """Plot labeled examples and, when supplied, a model's decision boundary."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    sns.set_style("whitegrid")

    figure, axes = plt.subplots(figsize=(10, 8))
    axes.set(xlabel="$X_1$", ylabel="$X_2$")
    axes.set_title(plot_title, fontsize=20)

    if XX is not None or YY is not None or preds is not None:
        if XX is None or YY is None or preds is None:
            raise ValueError("XX, YY, and preds must be supplied together.")
        probabilities = np.asarray(preds).ravel()
        if probabilities.size != XX.size:
            raise ValueError("The prediction count must match the plotting grid.")
        axes.contourf(
            XX, YY, probabilities.reshape(XX.shape), 25, alpha=1, cmap="Spectral"
        )
        axes.contour(
            XX, YY, probabilities.reshape(XX.shape), levels=[0.5], colors="black"
        )

    axes.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap="Spectral", edgecolors="black")
    figure.tight_layout()

    if file_name is not None:
        output_path = Path(file_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
    return figure


def make_boundary_gif(frame_paths, output_path, duration=100):
    """Combine boundary plot frames into an animated GIF."""
    frame_paths = list(frame_paths)
    if not frame_paths:
        raise ValueError("At least one boundary frame is required to create a GIF.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    try:
        for frame_path in frame_paths:
            with Image.open(frame_path) as frame:
                frames.append(frame.convert("RGB"))
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
    finally:
        for frame in frames:
            frame.close()


def _make_plot(X, y, plot_title, file_name=None, XX=None, YY=None, preds=None):
    """Backward-compatible alias for the decision-boundary plotting helper."""
    return plot_decision_boundary(X, y, plot_title, file_name, XX, YY, preds)
