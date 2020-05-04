import sys 

sys.path.insert(1, './preprocessing')
sys.path.insert(1, './data generation')
sys.path.insert(1, './modeling')
sys.path.insert(1, './plotting')

from sklearn import preprocessing, tree
import json 
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from tuner import tuneReg, tuneClass, tuneCNN
from os import listdir
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from dataGen import generate_data
from generatePlots import generateClusteringPlots, generateRegressionPlots, generateClassificationPlots, generateClassificationTogether
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.utils import np_utils
from keras.utils import to_categorical
from predictionModelCreation import getKerasModelClassification
from predictionModelCreation import getKerasModelRegression
from data_preprocesser import singleRegDataPreprocesser, preProcessImages
from grammartree import getValueFromInstruction
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
from dataset_labelmatcher import getmostSimilarColumn, getmostSimilarModel
from tensorflow.python.keras.layers import Dense, Input
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cosine
from tabulate import tabulate
import tensorflow as tf
import pandas as pd
import pprint
from colorama import Fore, Style
import numpy as np
import keras




def initializer(params):
    data = pd.read_csv(params['path_to_set'])
    data.fillna(0, inplace=True)
    params['data'] = data
    return params 

def preprocesser(params):
    data = singleRegDataPreprocesser(params['data'])
    params['data'] = data

def instruction_identifier(params):
    remove = getmostSimilarColumn(getValueFromInstruction(params['instruction']), params['data'])
    params['y'] = params['data'][remove]
    del params['data'][remove]

def set_splitter(params):
    params['X_train'], params['X_test'], params['y_train'], params['y_test'] = train_test_split(params['data'], params['y'], test_size=0.2, random_state=49)

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
        model = getKerasModelRegression(params['data'], i)
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
            model = getKerasModelRegression(params['data'], i)
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
    init_plots, plot_names = generateRegressionPlots(params['models'][len(params['models']) - 1], params['data'], params['y'])
    plots = {}
    for x in range(len(plot_names)):
        plots[str(plot_names[x])] = init_plots[x]
    params['plots'] = plots 
    

