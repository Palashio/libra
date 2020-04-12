import tensorflow as tf 
import keras 
from tensorflow.python.keras.layers import Dense, Input
import numpy as np

def getKerasModel(dataset): 
    model = tf.keras.Sequential()
    model.add(Dense(dataset.shape[1], input_dim=dataset.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer="normal", activation="softmax"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def convergence_test(history):
    return all(x>y for x, y in zip(history.history['loss'], history.history['loss'][1:])) and np.mean(history.history['val_loss'] + (2 * np.std(history.history['val_loss']))