import tensorflow as tf
import keras
from tensorflow.python.keras.layers import Dense, Input
import numpy as np


# Creates a regression neural network
def getKerasModelRegression(dataset, i):
    size_max_network = 10
    # base model
    model = tf.keras.Sequential()
    model.add(
        Dense(
            dataset.shape[1],
            input_dim=dataset.shape[1],
            kernel_initializer='normal',
            activation='relu'))
    model.add(Dense(64, activation="relu"))

    # Adds values depending on what the i value is
    for x in range(i):
        if(i < 5):
            model.add(
                Dense(
                    i * 64,
                    kernel_initializer="normal",
                    activation="relu"))
        else:
            model.add(
                Dense(
                    i * 64,
                    kernel_initializer="normal",
                    activation="relu"))

    model.add(Dense(1, kernel_initializer="normal", activation="softmax"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Creates a classification neural network


def getKerasModelClassification(dataset, i, num_classes):
    size_max_network = 10

    # base model
    model = tf.keras.Sequential()
    model.add(
        Dense(
            dataset.shape[1],
            input_dim=dataset.shape[1],
            kernel_initializer='normal',
            activation='relu'))
    model.add(Dense(64, activation="relu"))

    # Adds values depending on what the i value is
    for x in range(i):
        if(i < 5):
            model.add(
                Dense(
                    i * 64,
                    kernel_initializer="normal",
                    activation="relu"))
        else:
            model.add(
                Dense(
                    i * 64,
                    kernel_initializer="normal",
                    activation="relu"))

    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model


def getKerasConvolutional():
    print("entered")
