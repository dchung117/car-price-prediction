import pathlib
import numpy as np
import tensorflow as tf

from src.preprocess import preprocess_data
from src.model import get_model

LOSSES = {
    "mse": tf.keras.losses.MeanSquaredError,
    "mae": tf.keras.losses.MeanAbsoluteError,
    "huber": tf.keras.losses.Huber
}

def train(model: tf.keras.Model, X: tf.Tensor, y: tf.Tensor, 
    batch_size: int = 32, n_epochs: int = 100):
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
    :return: trained model
    :rtype: tf.keras.Model
    """
    model.fit(X, y, epochs=n_epochs, batch_size=batch_size, verbose=1)
    return model

if __name__ == "__main__":
    # Load, preprocess data
    data_file = pathlib.Path("data/train.csv")
    X, y, X_normalizer, y_normalizer = preprocess_data(data_file)
    
    # Create model
    model = get_model(X_normalizer, n_feats=X.shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=LOSSES["mae"]())
    tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

    model = train(model, X, y)


