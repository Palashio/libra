import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from data_preprocesser import singleRegDataPreprocesser
from predictionModelCreation import getKerasModelRegression
from predictionModelCreation import getKerasModelClassification
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def generateClusteringPlots(kmeans, dataPandas, dataset):
        plots = []
        plot_names = []
        for x in range(len(dataPandas.columns) - 1):
                for y in range(len(dataPandas.columns) - 1):
                        plt.scatter(dataset[:,x], dataset[:,y], c=kmeans.labels_, cmap='rainbow')
                        plt.xlabel(str(dataPandas.columns[x]))
                        plt.ylabel(str(dataPandas.columns[y]))
                        plots.append(plt)
                        plot_names.append(dataPandas.columns[x] + "vs"  + dataPandas.columns[y])
        return plots, plot_names 

def generateRegressionPlots(history, data, label):
        plots = []
        plot_names = []
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        
        plots.append(plt)
        plot_names.append('lossvsval_los')

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        
        plots.append(plt)
        plot_names.append('accuracyvsval_accuracy')

        return plots, plot_names

def generateClassificationPlots(history, data, label, model, X_test, y_test):
        plots = []
        plot_names = []
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        
        plots.append(plt)
        plot_names.append('lossvsval_los')

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        
        plots.append(plt)
        plot_names.append('accuracyvsval_accuracy')

        # y_pred = model.predict(X_test)
        # y_pred = np.argmax(y_pred, axis=1)
        # plots.append(confusion_matrix(y_test, y_pred))
        # plot_names.append("confusion_matrix")

        return_plots = {}
        for x in range(len(plots)):
                return_plots[str(plot_names[x])] = plots[x]

        return return_plots
