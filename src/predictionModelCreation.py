import tensorflow as tf 
import keras 
from tensorflow.python.keras.layers import Dense, Input

def getKerasModel(dataset): 
    model = tf.keras.Sequential()
    model.add(Dense(dataset.shape[1], input_dim=dataset.shape[1], kernel_initializer='normal', activation='relu'))


    
    model.add(Dense(1, kernel_initializer="normal", activation="softmax"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model