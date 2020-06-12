import os

import tensorflow as tf
import keras
from keras import Model
from keras.layers import LSTM, Embedding, TimeDistributed, Concatenate
from tensorflow.python.keras.layers import Dense, Input
import numpy as np

# Creates a regression neural network

def get_keras_model_reg(dataset, i):
    size_max_network = 10
    # base model
    model = tf.keras.Sequential()
    model.add(
        Dense(
            dataset['train'].shape[1],
            input_dim=dataset['train'].shape[1],
            kernel_initializer='normal',
            activation='relu'))
    model.add(Dense(64, activation="relu"))

    # Adds values depending on what the i value is
    for x in range(i):
        if (i < 5):
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

    model.add(Dense(1, kernel_initializer="normal"))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Creates a classification neural network


def get_keras_model_class(dataset, i, num_classes):
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
        if (i < 5):
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


# Sentiment analysis model
def get_keras_text_class(max_features, num_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(max_features, 128))
    model.add(tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def getKerasConvolutional():
    print("entered")
