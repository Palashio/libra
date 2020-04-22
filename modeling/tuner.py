import tensorflow as tf 
import keras 
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sys 
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

sys.path.insert(1, '/Users/palashshah/Desktop/Libra/preprocessing')
sys.path.insert(1, '/Users/palashshah/Desktop/Libra/data generation')
sys.path.insert(1, '/Users/palashshah/Desktop/Libra/modeling')
sys.path.insert(1, '/Users/palashshah/Desktop/Libra/plotting')
sys.path.insert(1, '/Users/palashshah/Desktop/Libra/plotting') 

from data_preprocesser import singleRegDataPreprocesser


data = pd.read_csv("./data/housing.csv")
data = singleRegDataPreprocesser(data)
def tuneReg(data):

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

    tuner = RandomSearch(
        build_model,
        objective='loss',
        max_trials=5,
        executions_per_trial=3,
        directory='my_dir',
        project_name='helloworld')

    # tuner.search_space_summary()

    y = data['median_house_value']
    del data['median_house_value']

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=49)

    tuner.search(X_train.values, y_train.values,
                epochs=5,
                validation_data=(X_test.values, y_test.values))

    models = tuner.get_best_models(num_models=1)
    return models[0]

def tuneClass(data, target_class, num_classes):
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

    tuner = RandomSearch(
        build_model,
        objective='loss',
        max_trials=5,
        executions_per_trial=3,
        directory='my_dir',
        project_name='helloworld')

    # tuner.search_space_summary()

    y = data[target_class]
    del data[target_class]

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=49)

    tuner.search(X_train.values, y_train.values,
                epochs=5,
                validation_data=(X_test.values, y_test.values))
    models = tuner.get_best_models(num_models=1)
    return models[0]


tuneReg(data)
