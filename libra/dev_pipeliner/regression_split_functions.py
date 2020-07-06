import json
import keras
import numpy as np
from colorama import Fore, Style
import pprint
import pandas as pd
import tensorflow as tf
from tabulate import tabulate
from scipy.spatial.distance import cosine
from sklearn.model_selection import cross_val_score
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, svm
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Input
from libra.data_generation.dataset_labelmatcher import get_similar_column, get_similar_model
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from libra.data_generation.grammartree import get_value_instruction
from libra.preprocessing.data_preprocesser import structured_preprocesser
from libra.modeling.prediction_model_creation import get_keras_model_reg
from libra.modeling.prediction_model_creation import get_keras_model_class
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from libra.data_generation import dataGen
from keras.models import Sequential
from keras.layers import (Dense, Conv2D, Flatten)
from os import listdir
from libra.modeling.tuner import tuneReg, tuneClass, tuneCNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing, tree
import sys

def initializer(params):
    data = pd.read_csv(params['path_to_set'])
    data.fillna(0, inplace=True)
    params['data'] = data
    return params


def preprocesser(params):
    data = structured_preprocesser(params['data'])
    params['data'] = data


def instruction_identifier(params):
    remove = get_similar_column(
        get_value_instruction(
            params['instruction']),
        params['data'])
    params['y'] = params['data'][remove]
    del params['data'][remove]


def set_splitter(params):
    params['X_train'], params['X_test'], params['y_train'], params['y_test'] = train_test_split(
        params['data'], params['y'], test_size=0.2, random_state=49)


def modeler(params):
    params['maximizer'] = 'loss'
    models = []
    losses = []

    # callback function to store lowest loss value
    params['es'] = EarlyStopping(
        monitor=params['maximizer'],
        mode='min',
        verbose=1,
        patience=5)

    i = 0

    # get the first 3 layer model
    model = get_keras_model_reg(params['data'], i)
    history = model.fit(
        params['X_train'],
        params['y_train'],
        epochs=5,
        validation_data=(
            params['X_test'],
            params['y_test']),
        callbacks=[params['es']])
    models.append(history)

    losses.append(models[i].history[params['maximizer']]
                  [len(models[i].history[params['maximizer']]) - 1])

    # keeps running model and fit functions until the validation loss stops
    # decreasing

    while(all(x > y for x, y in zip(losses, losses[1:]))):
        model = get_keras_model_reg(params['data'], i)
        history = model.fit(
            params['X_train'],
            params['y_train'],
            epochs=5,
            validation_data=(
                params['X_test'],
                params['y_test']),
            callbacks=[params['es']])
        models.append(history)
        losses.append(models[i].history[params['maximizer']]
                      [len(models[i].history[params['maximizer']]) - 1])
        i += 1

    params['models'] = models
    params['losses'] = losses


def plotter(params):
    init_plots, plot_names = generateRegressionPlots(
        params['models'][len(params['models']) - 1], params['data'], params['y'])
    plots = {}
    for x in range(len(plot_names)):
        plots[str(plot_names[x])] = init_plots[x]
    params['plots'] = plots
