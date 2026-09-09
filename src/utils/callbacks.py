from pathlib import Path

import keras
import numpy as np

from src.utils.visualize import plot_decision_boundary


def make_keras_plot_callback(X, y, grid_2d, XX, YY, frame_dir):
    """Return a Keras callback that saves one decision-boundary frame per epoch."""
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    class PlotCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            predictions = self.model.predict(grid_2d, verbose=0)
            plot_decision_boundary(
                X,
                y,
                f"Keras Model - Epoch {epoch:05}",
                file_name=frame_dir / f"keras_model_{epoch:05}.png",
                XX=XX,
                YY=YY,
                preds=predictions,
            )

    return PlotCallback()


def make_numpy_plot_callback(X, y, grid_2d, XX, YY, frame_dir, predict):
    """Return a NumPy training callback that saves each received boundary frame."""
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    def callback(iteration, params):
        predictions = predict(grid_2d, params)
        plot_decision_boundary(
            X,
            y,
            f"NumPy Model - Iteration {iteration:05}",
            file_name=frame_dir / f"numpy_model_{iteration:05}.png",
            XX=XX,
            YY=YY,
            preds=predictions,
        )

    return callback
