import pathlib
from typing import Optional, Iterable

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.preprocess import preprocess_data, train_val_test_split
from src.model import get_linear_model, get_mlp
from src.utils import save_plot, predict

LOSSES = {
    "mse": tf.keras.losses.MeanSquaredError,
    "mae": tf.keras.losses.MeanAbsoluteError,
    "huber": tf.keras.losses.Huber
}

def train(model: tf.keras.Model, train_data: tf.data.Dataset, 
    batch_size: int = 32, n_epochs: int = 100,
    val_data: Optional[tf.data.Dataset] = None) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Function to train a model.
    
    :param: model: Keras model to train.
    :dtype: tf.keras.Model
    :param: train_data: Training data
    :dtype: tf.data.Dataset
    :param: batch_size: Training batch size (def. 32).
    :dtype: int
    :param: n_epochs: Number of training epochs (def. 100).
    :dtype: int
    :param: val_data: Optional validation data
    :dtype: Optional[tf.data.Dataset]
    :return: trained model and training history
    :rtype: tuple[tf.keras.Model, tf.keras.callbacks.History]
    """
    if val_data:
        hist = model.fit(train_data, validation_data=val_data, epochs=n_epochs, batch_size=batch_size, verbose=1)
    else:
        hist = model.fit(train_data, epochs=n_epochs, batch_size=batch_size, verbose=1)
    return model, hist
    
if __name__ == "__main__":
    # Load, preprocess data
    data_file = pathlib.Path("data/train.csv")
    X, y, X_normalizer, y_normalizer = preprocess_data(data_file)
    
    # Split data into train, val, test
    train_dataset, val_dataset, test_dataset = train_val_test_split(X, y)
    print("Training size: ", len(train_dataset))
    print("Validation size: ", len(val_dataset))
    print("Test size: ", len(test_dataset))

    # Re-normalize to training set
    X_normalizer.adapt(
        tf.convert_to_tensor(
        np.concatenate([x for x,y in train_dataset], axis=0)
        )
    )
    y_normalizer.adapt(
        tf.convert_to_tensor(
        np.concatenate([y for x,y in train_dataset], axis=0)
        )
    )

    # Create model
    model = get_mlp(X_normalizer, n_feats=X.shape[1],
        hidden_layer_sizes=[16, 16, 16])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss=LOSSES["mae"](),
        metrics=tf.keras.metrics.RootMeanSquaredError())
    tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

    # Train model
    model, hist = train(model, train_dataset, val_data=val_dataset)
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
    model.evaluate(test_dataset, verbose=1)

    # Get prediction on test example
    x_batch, y_batch = next(iter(test_dataset))
    x_ex, y_ex = x_batch[0], y_batch[0].numpy()[0]
    y_pred = predict(x_ex, model)
    print("Car price prediction: ", y_pred)
    print("Prediction error: ", y_pred - y_ex)