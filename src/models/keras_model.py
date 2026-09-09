import keras
from sklearn.metrics import accuracy_score


def build_model(x_train):
    """Build model architecture."""
    model = keras.models.Sequential(
        [
            keras.layers.Dense(32, activation="relu", input_shape=(x_train.shape[1],)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model


def train_keras(model, x_train, y_train, epochs=50, batch_size=16, callbacks=None):
    """Compile and train keras model."""
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0,
    )

    model.save("results/keras_model.h5")

    return history


def test_keras(model, x_test, y_test):
    """Get test accuracy."""
    preds = model.predict(x_test) > 0.5  # threshold for 0/1 label
    acc = accuracy_score(y_test, preds)

    return acc
