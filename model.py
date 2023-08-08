import tensorflow.keras as keras
from tensorflow.keras.layers import Normalization, Dense

def get_model(normalizer: Normalization) -> keras.Model:
    """
    Define linear model for car price prediction.
    
    :param: normalizer: Normalization layer for input features
    :return: Linear model for car price prediction
    :rtype: keras.Model
    """
    
    model = keras.Sequential()
    model.add(normalizer)
    model.add(Dense(1))
    print("Model Summary: ")
    print(model.summary())
    return model