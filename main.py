import pathlib
from preprocess import preprocess_data
from model import get_model

if __name__ == "__main__":
    # Load, preprocess data
    data_file = pathlib.Path("data/train.csv")
    X, y, X_normalizer = preprocess_data(data_file)
    
    model = get_model(X_normalizer)

