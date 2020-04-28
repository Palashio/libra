#Making functions in other directories accesible to this file by inserting into sis path
import sys

sys.path.insert(1, './preprocessing')
sys.path.insert(1, './data generation')
sys.path.insert(1, './modeling')
sys.path.insert(1, './plotting')

#function imports from other files
import keras
import numpy as np
from colorama import Fore, Style
import pandas as pd
import tensorflow as tf
from tabulate import tabulate
from scipy.spatial.distance import cosine
from pandas import DataFrame
from sklearn import preprocessing, tree 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, svm
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
from generatePlots import generateClusteringPlots, generateRegressionPlots, generateClassificationPlots, generateClassificationTogether
from dataGen import generate_data
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
from os import listdir
from tuner import tuneReg, tuneClass
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

currLog = ""
counter = 0

#allows for all columns to be displayed when printing()
pd.options.display.width=None

def clearLog():
    global currLog
    currLog = ""

def logger(instruction, found = ""):
    global currLog 
    global counter 

    if counter == 0:
        currLog += (" " * 2 * counter) + instruction + found 
        currLog += "\n"        
    else:
        currLog += (" " * 2 * counter) + "|" 
        currLog += "\n"
        currLog += (" " * 2 * counter) + "|- " + instruction + found 
        currLog += "\n"
        if instruction == "done...":
            currLog +="\n"
            currLog += "\n"

    counter += 1
    print(currLog)

#class to store all query information
class client:
    def __init__(self, data):
        logger("creating object...")
        self.dataset = data
        logger("loading dataset...")
        self.models = {} 
        logger("done...")
        clearLog()


        
    #returns models with a specific string 
    def getModels(self): 
        return self.models

    def getAttributes(self, model_name):
        print(model_name['plots'])

    # single regression query using a feed-forward neural network
    # instruction should be the value of a column
    def SingleRegressionQueryANN(self, instruction):
            global currLog
            logger("reading in dataset...")
            data = pd.read_csv(self.dataset)
            logger("filling n/a values...")
            data.fillna(0, inplace=True)
            
            #identifies the categorical and numerical columns
            logger("identifying column types...")
            categorical_columns = data.select_dtypes(exclude=["number"]).columns
            numeric_columns = data.columns[data.dtypes.apply(lambda c: np.issubdtype(c, np.number))]

            #preprocesses data

            logger("hot encoding values and preprocessing...")
            data = singleRegDataPreprocesser(data)
            logger("identifying target from instruction...")
            logger("establishing callback function...")
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

            logger("training initial model...")
            history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])
            models.append(history)
            print(currLog)

            losses.append(models[i].history['val_loss'][len(models[i].history['val_loss']) - 1])

            #keeps running model and fit functions until the validation loss stops decreasing
            logger("testing number of layers...")
            print(currLog)
            while(all(x > y for x, y in zip(losses, losses[1:]))):
                model = getKerasModelRegression(data, i)
                history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])
                models.append(history)
                losses.append(models[i].history['val_loss'][len(models[i].history['val_loss']) - 1])
                i += 1

            #calls function to generate plots in plot generation
            init_plots, plot_names = generateRegressionPlots(models[len(models) - 1], data, y)
            plots = {}
            for x in range(len(plot_names)):
                plots[str(plot_names[x])] = init_plots[x]

            print(currLog)
            #stores values in the client object models dictionary field 
            self.models['regression_ANN'] = {'model' : model, "target" : remove, "plots" : plots, 'losses' : {'training_loss' : history.history['loss'], 'val_loss' : history.history['val_loss']},
                        'accuracy' : {'training_accuracy' : history.history['accuracy'], 'validation_accuracy' : history.history['val_accuracy']}}

            #returns the best model
            clearLog()
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
        self.models["classification_ANN"] = {"model" : model, 'num_classes' : num_classes, "plots" : plots, "target" : remove, 'losses' : {'training_loss' : history.history['loss'], 'val_loss' : history.history['val_loss']},
                        'accuracy' : {'training_accuracy' : history.history['accuracy'], 'validation_accuracy' : history.history['val_accuracy']}}

        #returns the last model 
        return model

    def kMeansClusteringQuery(self):
        logger("Reading dataset...")
        #loads dataset and replaces n/a with zero
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)
        dataPandas = data.copy()

        logger("Preprocessing datase...")
        data = np.asarray(singleRegDataPreprocesser(data))
        modelStorage = []
        inertiaStor = []

        #processes dataset and runs KMeans algorithm on one cluster as baseline
        i = 1
        logger("Creating unsupervised clustering task...")
        kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
        modelStorage.append(kmeans)

        #stores SSE values in an array for later comparison
        inertiaStor.append(kmeans.inertia_)
        i += 1

        logger("Identifying best centroid count and optimizing accuracy")
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
        logger("Generating plots and storing in model")
        init_plots, plot_names = generateClusteringPlots(modelStorage[len(modelStorage) - 1], dataPandas, data)
        
        plots = {}

        for x in range(len(plot_names)):
            plots[str(plot_names[x])] = init_plots[x]

        #stores plots and information in the dictionary client model
        self.models['kmeans_clustering'] = {"model" : modelStorage[len(modelStorage) - 1] ,"plots" : plots}
        clearLog()
        #return modelStorage[len(modelStorage) - 1], inertiaStor[len(inertiaStor) - 1], i

    def createCNNClassification(self, class1, class2):
        logger("Creating CNN generation query")
        #generates the dataset based on instructions using a selenium query on google chrome
        logger("Generating datasets for classes...")
        firstNumpy = generate_data(class1)
        secNumpy = generate_data(class2)

        #creates the appropriate dataset 
        firstLabels = [0] * len(firstNumpy)
        secLabels = [1] * len(secNumpy)
        y = []
        X = []

        logger("Applying resizing transformation algorithms...")
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

        logger("Creating convolutional neural network dynamically...")
        #Convolutional Neural Network
        model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(224,224,3)))
        model.add(Conv2D(32, kernel_size=3, activation="relu"))
        model.add(Flatten())
        model.add(Dense(2, activation="softmax"))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

        logger("Finishing task and storing information in model...")
        plots = generateClassificationPlots(history, X, y, model, X_test, y_test)
        generateClassificationTogether(history, X, y, model, X_test, y_test)
        self.models["classification_CNN"] = {"model" : model, 'num_classes' : len(np.unique(y_test)), "plots" : plots, "target" : class1 + "_" + class2, 'losses' : {'training_loss' : history.history['loss'], 'val_loss' : history.history['val_loss']},
                    'accuracy' : {'training_accuracy' : history.history['accuracy'], 'validation_accuracy' : history.history['val_accuracy']}}
        
        clearLog()
    
    def svmQuery(self, instruction):
        logger("Reading in dataset....")
        #reads dataset and fills n/a values with zeroes
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)

        logger("Identifying target columns...")
        remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)
        y = data[remove]
        del data[remove]

        #prepcoess the dataset
        logger("Preprocessing dataset")
        data = singleRegDataPreprocesser(data)
        #classification_column = getmostSimilarColumn(getLabelwithInstruction(instruction), data)

        num_classes = len(np.unique(y))

        #encodes the label dataset into 0's and 1's 
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=49)

        logger("Fitting Support Vector Machine")
        clf = svm.SVC()
        clf.fit(X_train, y_train)
        logger("Storing information in client object...")
        self.models["svm"] = {"model" : clf, "accuracy_score" : accuracy_score(clf.predict(X_test), y_test), "target" : remove}
        clearLog()
        return svm
    
    def nearestNeighborQuery(self, instruction):
        logger("Reading in dataset....")
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)

        logger("Identifying target columns...")
        remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)
        y = data[remove]
        del data[remove]

        #prepcoess the dataset
        logger("Preprocessing dataset...")
        data = singleRegDataPreprocesser(data)
        #classification_column = getmostSimilarColumn(getLabelwithInstruction(instruction), data)

        num_classes = len(np.unique(y))

        #encodes the label dataset into 0's and 1's 
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=49)

        models = []
        scores = []
        logger("Fitting Nearest Neighbor...")
        logger("Identifying optimal number of neighbors...")
        for x in range(3, 10):
            knn = KNeighborsClassifier(n_neighbors=x)
            knn.fit(X_train, y_train)
            models.append(knn)
            scores.append(accuracy_score(knn.predict(X_test), y_test))

        logger("Storing information in client object...")
        knn = models[scores.index(min(scores))]
        self.models["nearest_neighbors"] = {"model" : knn, "accuracy_score" : scores.index(min(scores)), "target" : remove}

        clearLog()
        return knn

    def decisionTreeQuery(self, instruction):
        logger("Reading in dataset....")
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)

        logger("Identifying target columns...")
        remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)
        y = data[remove]
        del data[remove]

        #prepcoess the dataset
        logger("Preprocessing dataset...")
        data = singleRegDataPreprocesser(data)
        #classification_column = getmostSimilarColumn(getLabelwithInstruction(instruction), data)

        num_classes = len(np.unique(y))

        #encodes the label dataset into 0's and 1's 
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        logger("Fitting Decision Tree...")
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=49)
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)

        logger("Storing information in client object...")
        self.models["decision_tree"] = {"model" : clf, "target" : remove}

        clearLog()
        return clf
    
    def allClassQuery(self, instruction):
        logger("Reading in dataset....")
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)

        logger("Identifying target columns...")
        remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)
        y = data[remove]
        del data[remove]

        #prepcoess the dataset
        logger("Preprocessing dataset...")
        data = singleRegDataPreprocesser(data)
        #classification_column = getmostSimilarColumn(getLabelwithInstruction(instruction), data)

        num_classes = len(np.unique(y))

        #encodes the label dataset into 0's and 1's 
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=49)
        scores = []
        models = []

        logger("Testing various classification models....")
        models.append(self.decisionTreeQuery(instruction))
        models.append(self.nearestNeighborQuery(instruction))
        models.append(self.svmQuery(instruction))

        logger("Identifying top scores...")
        for model in models:
            scores.append(accuracy_score(model.predict(X_test), y_test))

        clearLog()
        return models[scores.index(max(scores))]


    def tune(self, model_to_tune):
        logger("Getting target model for tuning...")
        for key in self.models:
            if key == model_to_tune:
                logger("Tuning model hyperparameters")
                returned_model = tuneReg(self.dataset, self.models[key]["target"])
                self.models['regression_ANN'] = {'model' : returned_model}
                return returned_model
            if key == model_to_tune:
                logger("Tuning model hyperparameters")
                returned_model = tuneClass(self.models[key]["target"], self.models[key]["num_classes"])
                self.models['classification_ANN'] = {'model' : returned_model}
                return returned_model

    def stat_analysis(self, column_name = "none"):
        logger("Reading in dataset....")
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)
        logger("Creating lambda object to format...")
        pdtabulate=lambda df:tabulate(df,headers='keys',tablefmt='psql')
        
        logger("Identifying columns to transform....")
        categor = data.select_dtypes(exclude=['int', 'float'])
        categor = categor.apply(LabelEncoder().fit_transform)
        for value in categor.columns:
            data[str(value)] = categor[str(value)]

        if column_name == "none":
            columns = []
            sim = []
            for first_val in data.columns:
                for sec_val in data.columns:
                    if first_val == sec_val:
                        continue
                    columns.append(str(first_val) + "_" + str(sec_val))
                    sim.append(1 - cosine(data[first_val], data[sec_val]))
                df = pd.DataFrame(columns = columns)
                df.loc[len(df)] = sim

            #print(np.argpartition(np.asarray(df.iloc[0]), -5)[-5:])
            cols = []
            vals = []
            logger("Restructuring dataset for similarity...")
            for val in np.argpartition(np.asarray(df.iloc[0]), -5)[-5:]:
                cols.append(df.columns[val])
                vals.append(df[df.columns[val]].iloc[0])
                frame = pd.DataFrame(columns=cols)
                frame.loc[len(df)] = vals
            print("Similarity Spectrum")
            print(pdtabulate(frame))
            print()
            print("Dataset Description")
            print(pdtabulate(data.describe()))

        else:
            logger("Performing similarity calculations....")
            columns = []
            sim = []
            for val in data.columns:
                    if val == column_name:
                        continue
                    columns.append(str(column_name) + "_" + str(val))
                    sim.append(1 - cosine(data[column_name], data[val]))
            df = pd.DataFrame(columns = columns)
            df.loc[len(df)] = sim

            cols = []
            vals = []
            for val in np.argpartition(np.asarray(df.iloc[0]), -5)[-5:]:
                cols.append(df.columns[val])
                vals.append(df[df.columns[val]].iloc[0])
                frame = pd.DataFrame(columns=cols)
                frame.loc[len(df)] = vals
            
            print("Similarity Spectrum")
            print("-------------------------")
            print(pdtabulate(frame))
            print()
            print("Dataset Description")
            print("-------------------------")
            print(pdtabulate(data[column_name]).describe())



        
        #print("Most important features: " + str(data.columns[indices]))






newClient = client("./data/housing.csv")
newClient.SingleRegressionQueryANN("Predict median house value")



