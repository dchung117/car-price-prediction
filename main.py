import pathlib
import tensorflow as tf

from src.preprocess import preprocess_data
from src.model import get_model

if __name__ == "__main__":
    # Load, preprocess data
    data_file = pathlib.Path("data/train.csv")
    X, y, X_normalizer, y_normalizer = preprocess_data(data_file)
    
    # Create model
    model = get_model(X_normalizer, n_feats=X.shape[1])
    tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
