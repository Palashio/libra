#Making functions in other directories accesible to this file by inserting into sis path
import sys

sys.path.insert(1, '/Users/palashshah/Desktop/Libra/preprocessing')
sys.path.insert(1, '/Users/palashshah/Desktop/Libra/data generation')
sys.path.insert(1, '/Users/palashshah/Desktop/Libra/modeling')
sys.path.insert(1, '/Users/palashshah/Desktop/Libra/plotting')
sys.path.insert(1, '/Users/palashshah/Desktop/Libra/plotting')

#function imports from other files
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Input
from dataset_labelmatcher import getmostSimilarColumn
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from grammartree import getValueFromInstruction
from data_preprocesser import singleRegDataPreprocesser, preProcessImages
from predictionModelCreation import getKerasModelRegression
from predictionModelCreation import getKerasModelClassification
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from generatePlots import generateClusteringPlots, generateRegressionPlots, generateClassificationPlots
from dataGen import generate_data
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
from os import listdir

#allows for all columns to be displayed when printing()
pd.set_option('display.max_columns', None)

#class to store all query information
class client:
    def __init__(self, data):
        self.dataset = data
        self.models = {} 

    #returns models with a specific string 
    def getModels(self, model_needed): 
        return self.models[str(model_needed)]

    def getAttributes(self, model_name):
        print(model_name['plots'])

    # single regression query using a feed-forward neural network
    # instruction should be the value of a column
    def SingleRegressionQueryANN(self, instruction):
            data = pd.read_csv(self.dataset)
            data.fillna(0, inplace=True)
            
            #identifies the categorical and numerical columns
            categorical_columns = data.select_dtypes(exclude=["number"]).columns
            numeric_columns = data.columns[data.dtypes.apply(lambda c: np.issubdtype(c, np.number))]

            #preprocesses data
            data = singleRegDataPreprocesser(data)
            remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)
            y = data[remove]
            del data[remove]

            X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=49)

            models=[]
            losses = []
            epochs = 5

            #callback function to store lowest loss value
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

            i = 0

            #get the first 3 layer model
            model = getKerasModelRegression(data, i)

            history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])
            models.append(history)

            losses.append(models[i].history['val_loss'][len(models[i].history['val_loss']) - 1])

            #keeps running model and fit functions until the validation loss stops decreasing
            while(all(x > y for x, y in zip(losses, losses[1:]))):
                model = getKerasModelRegression(data, i)
                history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])
                models.append(history)
                losses.append(models[i].history['val_loss'][len(models[i].history['val_loss']) - 1])
                print("The number of layers " + str(len(model.layers)))
                i += 1

            #calls function to generate plots in plot generation
            init_plots, plot_names = generateRegressionPlots(models[len(models) - 1], data, y)
            plots = {}
            for x in range(len(plot_names)):
                plots[str(plot_names[x])] = init_plots[x]

            #stores values in the client object models dictionary field 
            self.models['regression_ANN'] = {'model' : model, "plots" : plots, 'losses' : {'training_loss' : history.history['loss'], 'val_loss' : history.history['val_loss']},
                        'accuracy' : {'training_accuracy' : history.history['accuracy'], 'validation_accuracy' : history.history['val_accuracy']}}

            #returns the best model
            return models[len(models) - 1]


    #query for multilabel classification query, does not work for binaryclassification, fits to feed-forward neural network
    def classificationQueryANN(self, instruction):

        #reads dataset and fills n/a values with zeroes
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)

        remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)
        y = data[remove]
        del data[remove]

        #prepcoess the dataset
        data = singleRegDataPreprocesser(data)
        #classification_column = getmostSimilarColumn(getLabelwithInstruction(instruction), data)

        num_classes = len(np.unique(y))

        #encodes the label dataset into 0's and 1's 
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        y = np_utils.to_categorical(y)

        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=49)

        models=[]
        losses = []
        epochs = 5
        
        #early stopping callback
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

        i = 0
        model = getKerasModelClassification(data, i, num_classes)

        history = model.fit(data, y, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])
        models.append(history)

        losses.append(models[i].history['val_loss'][len(models[i].history['val_loss']) - 1])

        #keeps running model and fit functions until the validation loss stops decreasing
        while(all(x > y for x, y in zip(losses, losses[1:]))):
            model = getKerasModelClassification(data, i, num_classes)
            history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])
            models.append(history)
            losses.append(models[i].history['val_loss'][len(models[i].history['val_loss']) - 1])
            print("The number of layers " + str(len(model.layers)))
            i += 1

        #genreates appropriate classification plots by feeding all information
        plots = generateClassificationPlots(models[len(models) - 1], data, y, model, X_test, y_test)

        #stores the values and plots into the object dictionary
        self.models["classification_ANN"] = {"model" : model, "plots" : plots, 'losses' : {'training_loss' : history.history['loss'], 'val_loss' : history.history['val_loss']},
                        'accuracy' : {'training_accuracy' : history.history['accuracy'], 'validation_accuracy' : history.history['val_accuracy']}}

        #returns the last model 
        return model

    def kMeansClusteringQuery(self):
        #loads dataset and replaces n/a with zero
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)
        dataPandas = data.copy()
        data = np.asarray(singleRegDataPreprocesser(data))
        modelStorage = []
        inertiaStor = []

        #processes dataset and runs KMeans algorithm on one cluster as baseline
        i = 1
        kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
        modelStorage.append(kmeans)

        #stores SSE values in an array for later comparison
        inertiaStor.append(kmeans.inertia_)
        i += 1

        #continues to increase cluster size until SSE values don't decrease by 1000 - this value was decided based on precedence
        while(all(earlier >= later for earlier, later in zip(inertiaStor, inertiaStor[1:]))):
            kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
            modelStorage.append(kmeans)
            inertiaStor.append(kmeans.inertia_)
            #minimize inertia up to 10000
            i += 1

            #checks to see if it should continue to run; need to improve this algorithm
            if i > 3 and inertiaStor[len(inertiaStor) - 2] - 1000 <= inertiaStor[len(inertiaStor) - 1]:
                break

        #generates the clustering plots approiately
        init_plots, plot_names = generateClusteringPlots(modelStorage[len(modelStorage) - 1], dataPandas, data)
        
        plots = {}

        for x in range(len(plot_names)):
            plots[str(plot_names[x])] = init_plots[x]

        #stores plots and information in the dictionary client model
        self.models['kmeans_clustering'] = {"model" : modelStorage[len(modelStorage) - 1] ,"plots" : plots}
        #return modelStorage[len(modelStorage) - 1], inertiaStor[len(inertiaStor) - 1], i

    def createCNNClassification(self, class1, class2):

        #generates the dataset based on instructions using a selenium query on google chrome
        firstNumpy = generate_data(class1)
        secNumpy = generate_data(class2)

        #creates the appropriate dataset 
        firstLabels = [0] * len(firstNumpy)
        secLabels = [1] * len(secNumpy)
        y = []
        X = []

        #processes dataset and stores them in the data and label variables
        for x in range(len(firstLabels)):
            y.append(firstLabels[x])
            X.append(firstNumpy[x])

        for x in range(len(secLabels)):
            y.append(secLabels[x])
            X.append(secNumpy[x])
        
        X_train, X_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(y), test_size=0.33, random_state=42)

        print(X_train.shape)
        print(y_train.shape)

        #categorically encodes them for CNN processing
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        model = Sequential()

        #Convolutional Neural Network
        model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(224,224,3)))
        model.add(Conv2D(32, kernel_size=3, activation="relu"))
        model.add(Flatten())
        model.add(Dense(2, activation="softmax"))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)




newClient = client("./data/housing.csv")
newClient.classificationQueryANN("Model ocean proximity")




