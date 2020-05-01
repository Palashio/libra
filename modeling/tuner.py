import tensorflow as tf 
import keras 
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sys 
from tensorflow import keras
import kerastuner as kt
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel
from kerastuner.tuners import Hyperband
from tensorflow import keras 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from kerastuner.tuners import RandomSearch

sys.path.insert(1, '/Users/palashshah/Desktop/Libra/preprocessing')
sys.path.insert(1, '/Users/palashshah/Desktop/Libra/data generation')
sys.path.insert(1, '/Users/palashshah/Desktop/Libra/modeling')
sys.path.insert(1, '/Users/palashshah/Desktop/Libra/plotting')
sys.path.insert(1, '/Users/palashshah/Desktop/Libra/plotting') 

from data_preprocesser import singleRegDataPreprocesser

#creates hypermodel class for CNN tuning
class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    #model builder: creates different layers with a selection of parameters that during runtime it selects to use 
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
                filters=hp.Choice(
                    'num_filters',
                    values=[32, 64],
                    default=64,
                ),
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



def tuneReg(data, target):
    #reads in dataset and processes it 
    data = pd.read_csv(data)
    data = singleRegDataPreprocesser(data)

    #function build model using hyperparameter 
    def build_model(hp):
        model = keras.Sequential()
        for i in range(hp.Int('num_layers', 2, 10)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=32,
                                                max_value=512,
                                                step=32),
                                activation='relu'))
        model.add(layers.Dense(1, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='mean_squared_error',
            metrics=['accuracy'])
        return model
    #random search for the model 
    tuner = RandomSearch(
        build_model,
        objective='loss',
        max_trials=1,
        executions_per_trial=3,)

    # tuner.search_space_summary()

    y = data[target]
    del data[target]

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=49)

    #searches the tuner space defined by hyperparameters (hp) and returns the best model 
    tuner.search(X_train.values, y_train.values,
                epochs=5,
                validation_data=(X_test.values, y_test.values),
                callbacks=[tf.keras.callbacks.TensorBoard('my_dir')])

    models = tuner.get_best_models(num_models=1)
    return models[0]

def tuneClass(X, y, num_classes):
    #function build model using hyperparameter 
    def build_model(hp):
        model = keras.Sequential()
        for i in range(hp.Int('num_layers', 2, 10)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=32,
                                                max_value=512,
                                                step=32),
                                activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return model

    #tuners, establish the object to look through the tuner search space 
    tuner = RandomSearch(
        build_model,
        objective='loss',
        max_trials=5,
        executions_per_trial=3,
        directory='models',
        project_name='class_tuned')

    # tuner.search_space_summary()

    y = data[target_class]
    del data[target_class]

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=49)

    #searches the tuner space defined by hyperparameters (hp) and returns the best model 
    tuner.search(X_train.values, y_train.values,
                epochs=5,
                validation_data=(X_test.values, y_test.values))
    models = tuner.get_best_models(num_models=1)
    return models[0]


def tuneCNN(X, y, num_classes):

    #creates hypermodel object based on the num_classes and the input shape
    hypermodel = CNNHyperModel(input_shape=(224,224,3), num_classes=num_classes)

    #tuners, establish the object to look through the tuner search space 
    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        seed=42,
        max_trials=3,
        executions_per_trial=3,
        directory='random_search',
    )
    
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(y), test_size=0.33, random_state=42)

    #searches the tuner space defined by hyperparameters (hp) and returns the best model
    tuner.search(X_train, y_train,
                 validation_data=(X_test, y_test),
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])

    #returns the best model
    return tuner.get_best_models(1)[0]


