import pathlib
import pandas as pd
import tensorflow as tf

FEAT_COLS = [
    "years",
    "km",
    "rating",
    "condition",
    "economy",
    "top speed",
    "hp",
    "torque"
]

TARGET_COL = "current price"

def preprocess_data(data_file: pathlib.Path) \
    -> tuple[tf.Tensor, tf.Tensor, tf.keras.layers.Normalization, tf.keras.layers.Normalization]:
    """
    Prepare data for training.
    
    :param: data_file - path to training data csv.
    :dtype: pathlib.Path
    :return: feature and target tensors for training.
    :rtype: tuple[tf.Tensor, tf.Tensor]
    """
    # Load data
    data_file = pathlib.Path("data/train.csv")
    df = pd.read_csv(data_file)

    # Shuffle data
    data = tf.random.shuffle(
        tf.constant(
            df[FEAT_COLS + [TARGET_COL]].values,
            dtype=tf.float32
        )
    )

    # Get data and features
    X = data[:, :-1]
    y = tf.expand_dims(data[:, -1], axis=-1)
    
    # Normalize data
    X_normalizer = tf.keras.layers.Normalization()
    y_normalizer = tf.keras.layers.Normalization()
    X_normalizer.adapt(X)
    y_normalizer.adapt(y)
    X = X_normalizer(X)
    y = y_normalizer(y)

    return X, y, X_normalizer, y_normalizer

if __name__ == "__main__":
    # Load data
    data_file = pathlib.Path("data/train.csv")
    X, y, X_normalizer, y_normalizer = preprocess_data(data_file)
    print(X, y)
    print(X.shape, y.shape)