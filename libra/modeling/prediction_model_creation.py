import tensorflow as tf
import keras
from keras import Model
from keras.layers import LSTM, Embedding, TimeDistributed, Concatenate
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from keras.models import model_from_json
import os

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

    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    return model


# Sentiment analysis model
def get_keras_text_class(max_features, num_classes, lr):
    model = tf.keras.models.Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def getKerasConvolutional():
    print("entered")


# Loading the user defined model given that the user provides the model path.
# The user must provide the model Directory.
def load_model(self, load_path):
    load_file = load_path[load_path.rindex("/") + 1:load_path.rindex(".")]
    # loading model json file
    json_file = open(load_path, 'r')
    json_model = json_file.read()
    json_file.close()
    loaded_model = model_from_json(json_model)
    # loading weights in the model
    loaded_model.load_weights(load_file + ".h5")
    print("Model loaded...")
    return loaded_model
