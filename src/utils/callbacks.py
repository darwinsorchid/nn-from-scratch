import keras
import os
import numpy as np 
from src.utils.visualize import _make_plot


""" Define callback functions for plotting learning process.
"""

# Define graph grid boundaries
GRID_X_START = -1.5
GRID_X_END = 2.5
GRID_Y_START = -1.0
GRID_Y_END = 2
grid = np.mgrid[GRID_X_START:GRID_X_END:100j, GRID_Y_START:GRID_Y_END:100j]
grid_2d = grid.reshape(2, -1).T
XX, YY = grid


def make_plot_callback(X_test, y_test, grid_2d, XX, YY, output_dir):

    class PlotCallback(keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs=None):
            preds = self.model.predict(grid_2d, verbose=0)

            _make_plot(
                X_test,
                y_test,
                f"Keras Model - Epoch {epoch:05}",
                file_name=os.path.join(
                    output_dir,
                    f"keras_model_{epoch:05}.png"
                ),
                XX=XX,
                YY=YY,
                preds=preds,
            )

    return PlotCallback()