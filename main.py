import pathlib
from typing import Optional, Iterable

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.preprocess import preprocess_data, train_val_test_split
from src.model import get_model

LOSSES = {
    "mse": tf.keras.losses.MeanSquaredError,
    "mae": tf.keras.losses.MeanAbsoluteError,
    "huber": tf.keras.losses.Huber
}

def train(model: tf.keras.Model, X: tf.Tensor, y: tf.Tensor, 
    batch_size: int = 32, n_epochs: int = 100,
    val_data: Optional[tuple[tf.Tensor, tf.Tensor]] = None) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
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
    :param: val_data: Optional validation data
    :dtype: Optional[tuple[tf.Tensor, tf.Tensor]]
    :return: trained model and training history
    :rtype: tuple[tf.keras.Model, tf.keras.callbacks.History]
    """
    if val_data:
        hist = model.fit(X, y, validation_data=val_data, epochs=n_epochs, batch_size=batch_size, verbose=1)
    else:
        hist = model.fit(X, y, epochs=n_epochs, batch_size=batch_size, verbose=1)
    return model, hist

def save_plot(x: np.ndarray, y: Iterable[np.ndarray], save_file: pathlib.Path, x_label: Optional[str] = None, 
    y_label: Optional[str] = None, title: Optional[str] = None,
    legend: Optional[Iterable[str]] = None
    ) -> None:
    """
    Creates and saves plot of training metrics.
    
    :param: x: x-axis data
    :dtype: np.ndarray
    :param: y: y-axis data (metrics)
    :dtype: Iterable[np.ndarray]
    :param: save_file: path to save figure
    :dtype: pathlib.Path
    :param: x_label: optional x-axis label
    :dtype: Optional[str]
    :param: y_label: optional y-axis label
    :dtype: Optional[str]
    :param: title: optional title
    :dtype: Optional[str]
    :param: legend: optional legend
    :dtype: Optional[Iterable[str]]
    :return: None
    :rtype: None
    """
    fig, ax = plt.subplots()
    for y_tgt in y:
        ax.plot(x, y_tgt)
    if title:
        ax.title(title)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if len(legend):
        ax.legend(legend)
    plt.savefig(save_file)

    
if __name__ == "__main__":
    # Load, preprocess data
    data_file = pathlib.Path("data/train.csv")
    X, y, X_normalizer, y_normalizer = preprocess_data(data_file)
    
    # Split data into train, val, test
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    print("Training size: ", X_train.shape[0])
    print("Validation size: ", X_val.shape[0])
    print("Test size: ", X_test.shape[0])

    # Re-normalize to training set
    X_normalizer.adapt(X_train)
    y_normalizer.adapt(y_train)

    # Create model
    model = get_model(X_normalizer, n_feats=X.shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss=LOSSES["mae"](),
        metrics=tf.keras.metrics.RootMeanSquaredError())
    tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

    # Train model
    model, hist = train(model, X_train, y_train, val_data=(X_val, y_val))
    save_plot(
        np.arange(len(hist.history["loss"])),
        [hist.history["loss"], hist.history["val_loss"]],
        pathlib.Path("loss.png"),
        x_label="Epoch",
        y_label="Loss",
        legend=["Train", "Val"]
    )
    
    # rmse
    save_plot(
        np.arange(len(hist.history["root_mean_squared_error"])),
        [hist.history["root_mean_squared_error"], hist.history["val_root_mean_squared_error"]],
        pathlib.Path("rmse.png"),
        x_label="Epoch",
        y_label="RMSE",
        legend=["Train", "Val"]
    )

    # Evaluate model on test set
    model.evaluate(X_test, y_test, verbose=1)


