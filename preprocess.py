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
if __name__ == "__main__":
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
    print(X.shape, y.shape)