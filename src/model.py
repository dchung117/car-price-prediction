from typing import Iterable

import tensorflow.keras as keras
from tensorflow.keras.layers import Normalization, Dense, InputLayer

def get_linear_model(normalizer: Normalization, n_feats: int) -> keras.Model:
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
    model.add(InputLayer(input_shape=(n_feats, )))
    model.add(normalizer)
    model.add(Dense(1))
    print("Model Summary: ")
    print(model.summary())
    return model

def get_mlp(normalizer: Normalization, n_feats: int,
    hidden_layer_sizes: Iterable[int]) -> keras.Model:
    """
    Define dense feed forward network for car price prediction.
    
    :param: normalizer: Normalization layer for input features
    :dtype: Normalization
    :param: n_feats: Number of input features
    :dtype: int
    :param: hidden_layer_sizes: List of hidden layer sizes
    :dtype: Iterable[int]
    :return: MLP for car price prediction
    :rtype: keras.Model
    """
    
    model = keras.Sequential()
    model.add(InputLayer(input_shape=(n_feats, )))
    model.add(normalizer)
    for h_sz in hidden_layer_sizes:
        model.add(Dense(h_sz, activation="relu"))
    model.add(Dense(1))
    print("Model Summary: ")
    print(model.summary())
    return model