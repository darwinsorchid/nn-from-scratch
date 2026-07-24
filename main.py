import os
import keras
from src.models.keras_model import build_model, train_keras, test_keras
from data.generate_data import generate_data
from src.utils.visualize import _make_plot
from data.preprocess_data import process_data
from src.training.train import train
from src.models import nn
import numpy as np

OUTPUT_DIR = "results/"

DF_FILE_NAME = "data/datasets/moon.csv"

# Set-up numpy NN
NN_ARCHITECTURE= [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"}
]

LEARNING_RATE = 0.01
EPOCHS = 10000


def main():

    # Get moon data
    X, y = generate_data()
    # Get data plot
    _make_plot(X, y, plot_title="Dataset", file_name=os.path.join(OUTPUT_DIR, "moon_data.png"))

    # Preprocessing
    x_train, x_test, y_train, y_test = process_data(DF_FILE_NAME)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ NUMPY NN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Train
    params_values = train(np.transpose(x_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), NN_ARCHITECTURE, EPOCHS, LEARNING_RATE)

    # Get predictions on test set
    y_preds, _ = nn.full_forward_propagation(np.transpose(x_test), params_values, NN_ARCHITECTURE)

    # Get numpy nn test accuracy
    test_acc_nn = nn.accuracy(y_preds, np.transpose(y_test.reshape((y_test.shape[0], 1))))

    print(f"Test Accuracy of Numpy NN: {test_acc_nn:.2f}")


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ KERAS NN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Build
    keras_model = build_model(x_train)

    # Train 
    keras_history = train_keras(keras_model, x_train, y_train, epochs=50, batch_size=16)

    # Get keras test accuracy
    test_acc_keras = test_keras(keras_model, x_test, y_test)

    print(f"Test Accuracy of Keras NN: {test_acc_keras:.2f}")



if __name__ == "__main__":
    main()

