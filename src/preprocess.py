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

def train_val_test_split(X: tf.Tensor, y: tf.Tensor,
    train_fraction: float = 0.8, val_fraction: float = 0.1, test_fraction: float = 0.1) \
    -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Split features and targets into training, validation, test sets.
    
    :param: X - feature tensor
    :dtype: tf.Tensor
    :param: y - target tensor
    :dtype: tf.Tensor
    :param: train_fraction - training set split (def. 0.8)
    :dtype: float
    :param: val_fraction - validation set split (def. 0.1)
    :dtype: float
    :param: test_fraction - test set split (def. 0.1)
    :dtype: float
    :return: Training, validation, test features and targets
    :rtype: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
    """
    assert train_fraction + val_fraction + test_fraction == 1, "Train, val, test fractions must sum to 1."
    X_train, y_train = X[:int(len(X)*train_fraction)], y[:int(len(X)*train_fraction)]
    X_val, y_val = X[int(len(X)*train_fraction):int(len(X)*(train_fraction+val_fraction))], \
        y[int(len(X)*train_fraction):int(len(X)*(train_fraction+val_fraction))]
    X_test, y_test =X[-int(len(X)*test_fraction):], y[-int(len(X)*test_fraction):]

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    # Load data
    data_file = pathlib.Path("data/train.csv")
    X, y, X_normalizer, y_normalizer = preprocess_data(data_file)
    print(X, y)
    print(X.shape, y.shape)