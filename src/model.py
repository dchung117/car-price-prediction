import tensorflow.keras as keras
from tensorflow.keras.layers import Normalization, Dense, InputLayer

def get_model(normalizer: Normalization, n_feats: int) -> keras.Model:
    """
    Define linear model for car price prediction.
    
    :param: normalizer: Normalization layer for input features
    :dtype: Normalization
    :param: n_feats: Number of input features
    :dtype: int
    :return: Linear model for car price prediction
    :rtype: keras.Model
    """
    
    model = keras.Sequential()
    model.add(InputLayer(input_shape=(8, )))
    model.add(normalizer)
    model.add(Dense(1))
    print("Model Summary: ")
    print(model.summary())
    return model