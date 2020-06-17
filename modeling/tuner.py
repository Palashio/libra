from data_reader import DataReader
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from kerastuner.tuners import Hyperband
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from tensorflow.keras import layers
from sklearn import preprocessing
import kerastuner as kt
from tensorflow import keras
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense, Input
import keras
import tensorflow as tf
from data_preprocesser import structured_preprocesser, clustering_preprocessor
import sys

sys.path.insert(1, './preprocessing')
sys.path.insert(1, './data_generation')
sys.path.insert(1, './modeling')
sys.path.insert(1, './plotting')


# creates hypermodel class for CNN tuning

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    # model builder: creates different layers with a selection of parameters
    # that during runtime it selects to use
    def build(self, hp):
        model = keras.Sequential()
        model.add(
            Conv2D(
                filters=16,
                kernel_size=3,
                activation='relu',
                input_shape=self.input_shape
            )
        )
        model.add(
            Conv2D(
                filters=16,
                activation='relu',
                kernel_size=3
            )
        )
        model.add(MaxPooling2D(pool_size=2))
        model.add(
            Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        model.add(
            Conv2D(
                filters=32,
                kernel_size=3,
                activation='relu'
            )
        )
        model.add(
            Conv2D(
                filters=16,
                activation='relu',
                kernel_size=3
            )
        )
        model.add(MaxPooling2D(pool_size=2))
        model.add(
            Dropout(rate=hp.Float(
                'dropout_2',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            ))
        )
        model.add(Flatten())
        model.add(
            Dense(
                units=hp.Int(
                    'units',
                    min_value=32,
                    max_value=512,
                    step=32,
                    default=128
                ),
                activation=hp.Choice(
                    'dense_activation',
                    values=['relu', 'tanh', 'sigmoid'],
                    default='relu'
                )
            )
        )
        model.add(
            Dropout(
                rate=hp.Float(
                    'dropout_3',
                    min_value=0.0,
                    max_value=0.5,
                    default=0.25,
                    step=0.05
                )
            )
        )
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='LOG',
                    default=1e-3
                )
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


def tuneReg(
        data,
        target,
        max_layers=10,
        min_layers=2,
        min_dense=32,
        max_dense=512,
        executions_per_trial=3,
        max_trials=1):
    print("entered1")
    # function build model using hyperparameter

    def build_model(hp):
        model = keras.Sequential()
        for i in range(hp.Int('num_layers', min_layers, max_layers)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=min_dense,
                                                max_value=max_dense,
                                                step=32),
                                   activation='relu'))
        model.add(layers.Dense(1, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='mean_squared_error')
        return model

    # random search for the model
    tuner = RandomSearch(
        build_model,
        objective='loss',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial)

    # tuner.search_space_summary()
    # del data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=49)

    # searches the tuner space defined by hyperparameters (hp) and returns the
    # best model
    tuner.search(X_train, y_train,
                 epochs=5,
                 validation_data=(X_test, y_test),
                 callbacks=[tf.keras.callbacks.TensorBoard('my_dir')])

    models = tuner.get_best_models(num_models=1)
    return models[0]


def tuneClass(
        X,
        y,
        num_classes,
        max_layers=10,
        min_layers=2,
        min_dense=32,
        max_dense=512,
        executions_per_trial=3,
        max_trials=1,
        activation='relu',
        loss='categorical_crossentropy',
        metrics='accuracy'):
    # function build model using hyperparameter
    le = preprocessing.LabelEncoder()
    y = tf.keras.utils.to_categorical(
        le.fit_transform(y), num_classes=num_classes)

    def build_model(hp):
        model = keras.Sequential()
        for i in range(hp.Int('num_layers', min_layers, max_layers)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=min_dense,
                                                max_value=max_dense,
                                                step=32),
                                   activation=activation))
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss=loss,
            metrics=[metrics])
        return model

    # tuners, establish the object to look through the tuner search space
    tuner = RandomSearch(
        build_model,
        objective='loss',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='models',
        project_name='class_tuned')

    # tuner.search_space_summary()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=49)

    # searches the tuner space defined by hyperparameters (hp) and returns the
    # best model
    tuner.search(X_train, y_train,
                 epochs=5,
                 validation_data=(X_test, y_test))
    models = tuner.get_best_models(num_models=1)
    return models[0]


def tuneCNN(X, y, num_classes):

    # creates hypermodel object based on the num_classes and the input shape
    hypermodel = CNNHyperModel(input_shape=(
        224, 224, 3), num_classes=num_classes)

    # tuners, establish the object to look through the tuner search space
    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        seed=42,
        max_trials=3,
        executions_per_trial=3,
        directory='random_search',
    )

    X_train, X_test, y_train, y_test = train_test_split(
        np.asarray(X), np.asarray(y), test_size=0.33, random_state=42)

    # searches the tuner space defined by hyperparameters (hp) and returns the
    # best model
    tuner.search(X_train, y_train,
                 validation_data=(X_test, y_test),
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])

    # returns the best model
    return tuner.get_best_models(1)[0]
