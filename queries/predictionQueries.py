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
from sklearn.model_selection import cross_val_score
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
from dataset_labelmatcher import getmostSimilarColumn, getmostSimilarModel
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
from tuner import tuneReg, tuneClass, tuneCNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

currLog = ""
counter = 0

#allows for all columns to be displayed when printing()
pd.options.display.width=None

def clearLog():
    global currLog
    global counter 

    currLog = ""
    counter = 0 

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
        self.old_models = {}
        logger("done...")
        clearLog()

    #returns models with a specific string 
    def getModels(self, model_requested): 
        logger("Getting model...")
        return getmostSimilarModel(model_requested, self.models.keys())
        clearLog()

    # single regression query using a feed-forward neural network
    # instruction should be the value of a column
    def SingleRegressionQueryANN(self, instruction, preprocess=True, test_size=0.2, epochs=5, generate_plots=True, maximizer = "val_loss"):
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
            if preprocess:
                logger("hot encoding values and preprocessing...")
                data = singleRegDataPreprocesser(data)
            
            logger("identifying target from instruction...")
            logger("establishing callback function...")
            remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)
            y = data[remove]
            del data[remove]

            X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=test_size, random_state=49)

            models=[]
            losses = []

            #callback function to store lowest loss value
            
            es = EarlyStopping(monitor=maximizer, mode='min', verbose=1, patience=5)

            i = 0

            #get the first 3 layer model
            model = getKerasModelRegression(data, i)

            logger("training initial model...")
            history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])
            models.append(history)
            print(currLog)

            losses.append(models[i].history[maximizer][len(models[i].history[maximizer]) - 1])

            #keeps running model and fit functions until the validation loss stops decreasing
            logger("testing number of layers...")
            print(currLog)
            while(all(x > y for x, y in zip(losses, losses[1:]))):
                model = getKerasModelRegression(data, i)
                history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])
                models.append(history)
                losses.append(models[i].history[maximizer][len(models[i].history[maximizer]) - 1])
                i += 1

            #calls function to generate plots in plot generation
            if generate_plots:
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
    def classificationQueryANN(self, instruction, preprocess=True, test_size=0.2, epochs=5, generate_plots=True, maximizer = "val_loss"):

        #reads dataset and fills n/a values with zeroes
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)

        remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)
        y = data[remove]
        del data[remove]

        #prepcoess the dataset
        if preprocess:
            data = singleRegDataPreprocesser(data)
            #classification_column = getmostSimilarColumn(getLabelwithInstruction(instruction), data)

            num_classes = len(np.unique(y))

            #encodes the label dataset into 0's and 1's 
            le = preprocessing.LabelEncoder()
            y = le.fit_transform(y)
            y = np_utils.to_categorical(y)

        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=test_size, random_state=49)

        models=[]
        losses = []
        
        #early stopping callback
        es = EarlyStopping(monitor=maximizer, mode='min', verbose=1, patience=5)

        i = 0
        model = getKerasModelClassification(data, i, num_classes)

        history = model.fit(data, y, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])
        models.append(history)

        losses.append(models[i].history[maximizer][len(models[i].history[maximizer]) - 1])

        #keeps running model and fit functions until the validation loss stops decreasing
        while(all(x > y for x, y in zip(losses, losses[1:]))):
            model = getKerasModelClassification(data, i, num_classes)
            history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[es])
            models.append(history)
            losses.append(models[i].history[maximizer][len(models[i].history[maximizer]) - 1])
            print("The number of layers " + str(len(model.layers)))
            i += 1

        #genreates appropriate classification plots by feeding all information
        plots = generateClassificationPlots(models[len(models) - 1], data, y, model, X_test, y_test)

        #stores the values and plots into the object dictionary
        self.models["classification_ANN"] = {"model" : model, 'num_classes' : num_classes, "plots" : plots, "target" : remove, 'losses' : {'training_loss' : history.history['loss'], 'val_loss' : history.history['val_loss']},
                        'accuracy' : {'training_accuracy' : history.history['accuracy'], 'validation_accuracy' : history.history['val_accuracy']}}

        #returns the last model 
        return model

    def kMeansClusteringQuery(self, preprocess=True, generate_plots = True):
        logger("Reading dataset...")
        #loads dataset and replaces n/a with zero
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)
        dataPandas = data.copy()

        if preprocess:
            logger("Preprocessing data...")
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
        if generate_plots:
            logger("Generating plots and storing in model")
            init_plots, plot_names = generateClusteringPlots(modelStorage[len(modelStorage) - 1], dataPandas, data)
        
            plots = {}

            for x in range(len(plot_names)):
                plots[str(plot_names[x])] = init_plots[x]

        #stores plots and information in the dictionary client model
        self.models['kmeans_clustering'] = {"model" : modelStorage[len(modelStorage) - 1] ,"plots" : plots}
        clearLog()
        #return modelStorage[len(modelStorage) - 1], inertiaStor[len(inertiaStor) - 1], i
    
    def svmQuery(self, instruction, test_size = 0.2, kernel='linear'):
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

        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=test_size, random_state=49)

        logger("Fitting Support Vector Machine")
        clf = svm.SVC('linear')
        clf.fit(X_train, y_train)
        logger("Storing information in client object...")
        self.models["svm"] = {"model" : clf, "accuracy_score" : accuracy_score(clf.predict(X_test), y_test), "target" : remove, "cross_val_score" : cross_val_score(clf, data, y, cv=3)}
        clearLog()
        return svm
    
    def nearestNeighborQuery(self, instruction, preprocess = True, min_neighbors = 3, max_neighbors = 10):
        logger("Reading in dataset....")
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)

        logger("Identifying target columns...")
        remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)
        y = data[remove]
        del data[remove]

        #prepcoess the dataset
        if preprocess:
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
        for x in range(min_neighbors, max_neighbors):
            knn = KNeighborsClassifier(n_neighbors=x)
            knn.fit(X_train, y_train)
            models.append(knn)
            scores.append(accuracy_score(knn.predict(X_test), y_test))

        logger("Storing information in client object...")
        knn = models[scores.index(min(scores))]
        self.models["nearest_neighbors"] = {"model" : knn, "accuracy_score" : scores.index(min(scores)), "target" : remove, "cross_val_score" : cross_val_score(knn, data, y, cv=3) }

        clearLog()
        return knn

    def decisionTreeQuery(self, instruction, preprocess = True, test_size=0.2):
        logger("Reading in dataset....")
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)

        logger("Identifying target columns...")
        remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)
        y = data[remove]
        del data[remove]

        #prepcoess the dataset
        if preprocess:
            logger("Preprocessing dataset...")
            data = singleRegDataPreprocesser(data)
        #classification_column = getmostSimilarColumn(getLabelwithInstruction(instruction), data)

        num_classes = len(np.unique(y))

        #encodes the label dataset into 0's and 1's 
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        logger("Fitting Decision Tree...")
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=test_size, random_state=49)
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)

        logger("Storing information in client object...")
        self.models["decision_tree"] = {"model" : clf, "target" : remove, "cross_val_score" : cross_val_score(clf, data, y, cv=3)}

        clearLog()
        return clf
    
    def allClassQuery(self, instruction, preprocess=True, test_size=0.2):
        logger("Reading in dataset....")
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)

        logger("Identifying target columns...")
        remove = getmostSimilarColumn(getValueFromInstruction(instruction), data)
        y = data[remove]
        del data[remove]

        #prepcoess the dataset
        if preprocess:
            logger("Preprocessing dataset...")
            data = singleRegDataPreprocesser(data)
        #classification_column = getmostSimilarColumn(getLabelwithInstruction(instruction), data)

        num_classes = len(np.unique(y))

        #encodes the label dataset into 0's and 1's 
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=test_size, random_state=49)
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
            if key == model_to_tune and key == "regression_ANN":
                logger("Tuning model hyperparameters")
                returned_model = tuneReg(self.dataset, self.models[key]["target"])
                self.models['regression_ANN'] = {'model' : returned_model}
                return returned_model
            if key == model_to_tune and key == "classification_ANN":
                logger("Tuning model hyperparameters")
                returned_model = tuneClass(self.models[key]["target"], self.models[key]["num_classes"])
                self.models['classification_ANN'] = {'model' : returned_model}
                return returned_model
            if key == model_to_tune and key == "convolutional_NN":
                logger("Tuning model hyperparameters")
                X = self.models['convolutional_NN']["X"]
                y = self.models['convolutional_NN']["y"]
                model = tuneCNN(np.asarray(X), np.asarray(y), 2)
                self.models["convolutional_NN"]["model"] = model


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

    def convolutionalNNQuery(self, *argv):
        X = []
        y = []
        i = 0

        for location in argv:
            data = preProcessImages(location)
            for image in data:
                X.append(image)
                y.append(i)
            i += 1
 
        X_train, X_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(y), test_size=0.33, random_state=42)

      
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        y = to_categorical(y)

        print(X_train.shape)
        model = Sequential()

        logger("Creating convolutional neural network dynamically...")
        #Convolutional Neural Network
        model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(224, 224, 3)))
        model.add(Conv2D(32, kernel_size=3, activation="relu"))
        model.add(Flatten())
        model.add(Dense(2, activation="softmax"))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

        self.models["convolutional_NN"] = {"model" : model, "X" : X, "y" : y, 'num_classes' : len(np.unique(y_test)), 'losses' : {'training_loss' : history.history['loss'], 'val_loss' : history.history['val_loss']},
                'accuracy' : {'training_accuracy' : history.history['accuracy'], 'validation_accuracy' : history.history['val_accuracy']}}
        
        

    def generateSetFitCNN(self, *argv):
        logger("Creating CNN generation query")
        #generates the dataset based on instructions using a selenium query on google chrome
        logger("Generating datasets for classes...")
        input_shape = (224,224,3)
        y = []
        X = []
        q = 0

        num_classes = len(argv)
        for a_class in argv: 
            a_numpy = generate_data(a_class)
            a_label = [q] * len(a_numpy)
            q += 1

            for i in range(len(a_numpy)):
                X.append(a_numpy[i])
                y.append(a_label[i])

        #creates the appropriate dataset 

        logger("Applying resizing transformation algorithms...")
        #processes dataset and stores them in the data and label variables

        
        X_train, X_test, y_train, y_test = train_test_split(np.asarray(X), np.asarray(y), test_size=0.33, random_state=42)

        #categorically encodes them for CNN processing
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        model = Sequential()

        logger("Creating convolutional neural network dynamically...")
        #Convolutional Neural Network
        model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=input_shape))
        model.add(Conv2D(32, kernel_size=3, activation="relu"))
        model.add(Flatten())
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

        logger("Finishing task and storing information in model...")

       
        plots = generateClassificationPlots(history, X, y, model, X_test, y_test)
        generateClassificationTogether(history, X, y, model, X_test, y_test)

        self.models["classification_CNN"] = {"model" : model, 'num_classes' : len(np.unique(y_test)), "plots" : plots, 'losses' : {'training_loss' : history.history['loss'], 'val_loss' : history.history['val_loss']},
                    'accuracy' : {'training_accuracy' : history.history['accuracy'], 'validation_accuracy' : history.history['val_accuracy']}}
        
        clearLog()
            

            

        #class_one_data = preProcessImages(fir_class_loc)





        



        
        #print("Most important features: " + str(data.columns[indices]))






newClient = client("./data/housing.csv")
#newClient.createCNNClassification("apples", "oranges")
newClient.convolutionalNNQuery("./apples", "./oranges")
newClient.tune("convolutional_NN")
# newClient.SingleRegressionQueryANN("Predict median house value")
# newClient.getModels('regression')



