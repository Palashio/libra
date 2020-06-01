import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import sys

sys.path.insert(1, './preprocessing')
sys.path.insert(1, './data generation')
sys.path.insert(1, './modeling')
sys.path.insert(1, './plotting')

from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from matplotlib import pyplot as PLT
from data_preprocesser import structured_preprocesser
from predictionModelCreation import get_keras_model_reg
from predictionModelCreation import get_keras_model_class
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve

# generates all of the plots in clustering


def generate_clustering_plots(kmeans, dataPandas, dataset):
    plots = []
    plot_names = []
    # for all of the columns that're present in clustering it plots the
    # columns with each other based on the cluster they're in
    for x in range(len(dataPandas.columns) - 1):
        for y in range(len(dataPandas.columns) - 1):
            plt.scatter(dataset[:, x], dataset[:, y],
                        c=kmeans.labels_, cmap='rainbow')
            plt.xlabel(str(dataPandas.columns[x]))
            plt.ylabel(str(dataPandas.columns[y]))
            plt.show()
            plots.append(plt)
            plot_names.append(
                dataPandas.columns[x] +
                "_vs_" +
                dataPandas.columns[y])
    return plots, plot_names

# generates all of the plots for regression


def generate_regression_plots(history, data, label):
    plots = []
    plot_names = []
    # generating plots for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plots.append(plt)
    plot_names.append('lossvsval_los')

    return plots, plot_names


def generate_classification_plots(history, data, label, model, X_test, y_test):
    plots = []
    plot_names = []

    # generating plots for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plots.append(plt)
    plot_names.append('accuracyvsval_accuracy')

    # generating plots for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plots.append(plt)
    plot_names.append('lossvsval_los')


    # dynamic way to return all possible plots in case it expands together
    return_plots = {}
    for x in range(len(plots)):
        return_plots[str(plot_names[x])] = plots[x]

    return return_plots

# function to return both val and accuracy plots on one pane


def generate_classification_together(history, data, model, X_test, y_test):
    plots = []
    plot_names = []

    arrEpochs = []
    # stores all of the history information
    for x in range(len(history.history['loss'])):
        arrEpochs.append(x + 1)

    # the first loss plot on the top
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # the second accuracy plot on the bottom
    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
