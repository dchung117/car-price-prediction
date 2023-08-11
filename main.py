import pathlib
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.preprocess import preprocess_data
from src.model import get_model

LOSSES = {
    "mse": tf.keras.losses.MeanSquaredError,
    "mae": tf.keras.losses.MeanAbsoluteError,
    "huber": tf.keras.losses.Huber
}

TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

def train(model: tf.keras.Model, X: tf.Tensor, y: tf.Tensor, 
    batch_size: int = 32, n_epochs: int = 100) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Function to train a model.
    
    :param: model: Keras model to train.
    :dtype: tf.keras.Model
    :param: X: Model training input tensor.
    :dtype: tf.Tensor
    :param: y: Model training target tensor.
    :dtype: tf.Tensor
    :param: batch_size: Training batch size (def. 32).
    :dtype: int
    :param: n_epochs: Number of training epochs (def. 100).
    :dtype: int
    :return: trained model and training history
    :rtype: tuple[tf.keras.Model, tf.keras.callbacks.History]
    """
    hist = model.fit(X, y, epochs=n_epochs, batch_size=batch_size, verbose=1)
    return model, hist

def save_plot(x: np.ndarray, y: np.ndarray, save_file: pathlib.Path, x_label: Optional[str] = None, 
    y_label: Optional[str] = None, title: Optional[str] = None,
    ) -> None:
    """
    Creates and saves plot of training metrics.
    
    :param: x: x-axis data
    :dtype: np.ndarray
    :param: y: y-axis data (metric)
    :dtype: np.ndarray
    :param: save_file: path to save figure
    :dtype: pathlib.Path
    :param: x_label: optional x-axis label
    :dtype: Optional[str]
    :param: y_label: optional y-axis label
    :dtype: Optional[str]
    :param: title: optional title
    :dtype: Optional[str]
    """
    fig, ax = plt.subplots()
    ax.plot(x, y)
    if title:
        ax.title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    plt.savefig(save_file)
    
if __name__ == "__main__":
    # Load, preprocess data
    data_file = pathlib.Path("data/train.csv")
    X, y, X_normalizer, y_normalizer = preprocess_data(data_file)
    
    # Split data into train, val, test
    X_train, y_train = X[:int(len(X)*TRAIN_FRACTION)], y[:int(len(X)*TRAIN_FRACTION)]
    X_val, y_val = X[int(len(X)*TRAIN_FRACTION):int(len(X)*(TRAIN_FRACTION+VAL_FRACTION))], \
        y[int(len(X)*TRAIN_FRACTION):int(len(X)*(TRAIN_FRACTION+VAL_FRACTION))]
    X_test, y_test =X[-int(len(X)*TEST_FRACTION):], y[-int(len(X)*TEST_FRACTION):]
    print("Training size: ", X_train.shape[0])
    print("Validation size: ", X_val.shape[0])
    print("Test size: ", X_test.shape[0])


    # Create model
    model = get_model(X_normalizer, n_feats=X.shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss=LOSSES["mae"](),
        metrics=tf.keras.metrics.RootMeanSquaredError())
    tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

    # Train model
    model, hist = train(model, X, y)
    save_plot(
        np.arange(len(hist.history["loss"])),
        hist.history["loss"],
        pathlib.Path("train_loss.png"),
        x_label="Epoch",
        y_label="Loss"
    )
    save_plot(
        np.arange(len(hist.history["root_mean_squared_error"])),
        hist.history["root_mean_squared_error"],
        pathlib.Path("train_rmse.png"),
        x_label="Epoch",
        y_label="RMSE"
    )

    # Evaluate model
    model.evaluate(X, y, verbose=1)


