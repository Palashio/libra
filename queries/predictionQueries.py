# Making functions in other directories accesible to this file by
# inserting into sis path
import sys
sys.path.insert(1, './preprocessing')
sys.path.insert(1, './data generation')
sys.path.insert(1, './modeling')
sys.path.insert(1, './plotting')

import numpy as np
import pandas as pd
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
from dataset_labelmatcher import get_similar_column, get_similar_model
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from grammartree import get_value_instruction
from data_preprocesser import structured_preprocesser
from predictionModelCreation import get_keras_model_reg
from predictionModelCreation import get_keras_model_class
from keras.utils import to_categorical
from keras.utils import np_utils
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from generatePlots import (generate_clustering_plots, 
generate_regression_plots, 
generate_classification_plots, 
generate_classification_together)

from keras.models import Sequential
from keras.layers import (Dense, Conv2D, Flatten, Input,MaxPooling2D,)
from dimensionality_red_queries import dimensionality_reduc
from os import listdir
from tuner import tuneReg, tuneClass, tuneCNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing, tree
from keras.preprocessing.image import ImageDataGenerator
import os



# function imports from other files

currLog = ""
counter = 0

# allows for all columns to be displayed when printing()
pd.options.display.width = None

# clears the log when new process is started up


def clearLog():
    global currLog
    global counter

    currLog = ""
    counter = 0

# logging function that creates hierarchial display of the processes of
# different functions. Copied into different python files to maintain
# global variable parallels


def logger(instruction, found=""):
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
            currLog += "\n"
            currLog += "\n"

    counter += 1
    print(currLog)
    currLog=""

# class to store all query information

class client:
    def __init__(self, data):
        logger("creating object...")
        self.dataset = data
        logger("loading dataset...")
        self.models = {}
        self.old_models = {}
        logger("done...")
        clearLog()

    # returns models with a specific string
    def get_models(self, model_requested):
        logger("Getting model...")
        return get_similar_model(model_requested, self.models.keys())
        clearLog()

    # single regression query using a feed-forward neural network
    # instruction should be the value of a column
    def regression_query_ann(
            self,
            instruction,
            preprocess=True,
            test_size=0.2,
            random_state=49,
            epochs=50,
            generate_plots=True,
            callback_mode='min',
            maximizer="val_loss"):

        global currLog
        logger("reading in dataset...")
        data = pd.read_csv(self.dataset)
        logger("filling n/a values...")
        data.fillna(0, inplace=True)

        # identifies the categorical and numerical columns
        logger("identifying column types...")
        categorical_columns = data.select_dtypes(exclude=["number"]).columns
        numeric_columns = data.columns[data.dtypes.apply(
            lambda c: np.issubdtype(c, np.number))]

        # preprocesses data
        if preprocess:
            logger("hot encoding values and preprocessing...")
            data = structured_preprocesser(data)

        # identifies the most similar column and creates dataset appropriately.
        logger("identifying target from instruction...")
        logger("establishing callback function...")
        remove = get_similar_column(
            get_value_instruction(instruction), data)
        y = data[remove]
        del data[remove]

        X_train, X_test, y_train, y_test = train_test_split(
            data, y, test_size=test_size, random_state=random_state)

        models = []
        losses = []
        accuracies=[]

        # callback function to store lowest loss value
        es = EarlyStopping(
            monitor=maximizer,
            mode=callback_mode,
            verbose=0,
            patience=5)

        i = 0

        # get the first 3 layer model
        model = get_keras_model_reg(data, i)

        logger("training initial model...")
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            validation_data=(
                X_test,
                y_test),
            verbose=0,
            callbacks=[es])
        models.append(history)
        print(currLog)

        losses.append(models[i].history[maximizer]
                      [len(models[i].history[maximizer]) - 1])
        accuracies.append(models[i].history['val_accuracy']
                      [len(models[i].history['val_accuracy']) - 1])
        # keeps running model and fit functions until the validation loss stops
        # decreasing
        logger("testing number of layers...")
        print(currLog)
        while(all(x > y for x, y in zip(losses, losses[1:]))):
            model = get_keras_model_reg(data, i)
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                validation_data=(
                    X_test,
                    y_test),
                verbose=0)
            models.append(history)
            losses.append(models[i].history[maximizer]
                          [len(models[i].history[maximizer]) - 1])
            accuracies.append(models[i].history['val_accuracy']
                      [len(models[i].history['val_accuracy']) - 1])
            i += 1

        # calls function to generate plots in plot generation
        if generate_plots:
            init_plots, plot_names = generate_regression_plots(
                models[len(models) - 1], data, y)
            plots = {}
            for x in range(len(plot_names)):
                plots[str(plot_names[x])] = init_plots[x]

        print(currLog)
        # stores values in the client object models dictionary field
        self.models['regression_ANN'] = {
            'model': model,
            "target": remove,
            "plots": plots,
            'losses': {
                'training_loss': history.history['loss'],
                'val_loss': history.history['val_loss']},
            'accuracies':{'training_accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy']}}

        # returns the best model
        clearLog()
        return models[len(models) - 1]

    # query for multilabel classification query, does not work for
    # binaryclassification, fits to feed-forward neural network
    def classification_query_ann(
            self,
            instruction,
            preprocess=True,
            callback_mode='min',
            random_state=49,
            test_size=0.2,
            epochs=5,
            generate_plots=True,
            maximizer="val_loss"):

        # reads dataset and fills n/a values with zeroes
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)

        remove = get_similar_column(
            get_value_instruction(instruction), data)
        y = data[remove]
        del data[remove]

        # prepcoess the dataset
        if preprocess:
            data = structured_preprocesser(data)
            num_classes = len(np.unique(y))

            # encodes the label dataset into 0's and 1's
            le = preprocessing.LabelEncoder()
            y = le.fit_transform(y)
            y = np_utils.to_categorical(y)

        X_train, X_test, y_train, y_test = train_test_split(
            data, y, test_size=test_size, random_state=random_state)

        models = []
        losses = []
        accuracies=[]

        # early stopping callback
        es = EarlyStopping(
            monitor=maximizer,
            mode='min',
            verbose=0,
            patience=5)

        i = 0
        model = get_keras_model_class(data, i, num_classes)

        history = model.fit(
            data, y, epochs=epochs, validation_data=(
                X_test, y_test), verbose=0, callbacks=[es])
        models.append(history)

        losses.append(models[i].history[maximizer]
                      [len(models[i].history[maximizer]) - 1])
        accuracies.append(models[i].history['val_accuracy']
                      [len(models[i].history['val_accuracy']) - 1])
        # keeps running model and fit functions until the validation loss stops
        # decreasing
        while(all(x > y for x, y in zip(losses, losses[1:]))):
            model = get_keras_model_class(data, i, num_classes)
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                validation_data=(
                    X_test,
                    y_test),
                verbose=0,
                callbacks=[es])
            models.append(history)
            losses.append(models[i].history[maximizer]
                          [len(models[i].history[maximizer]) - 1])
            accuracies.append(models[i].history['val_accuracy']
                      [len(models[i].history['val_accuracy']) - 1])
            print("The number of layers " + str(len(model.layers)))
            i += 1

        # genreates appropriate classification plots by feeding all information
        plots = generate_classification_plots(
            models[len(models) - 1], data, y, model, X_test, y_test)

        # stores the values and plots into the object dictionary
        self.models["classification_ANN"] = {
            "model": model,
            'num_classes': num_classes,
            "plots": plots,
            "target": remove,
            'losses': {
                'training_loss': history.history['loss'],
                'val_loss': history.history['val_loss']},
            'accuracy': {
                'training_accuracy': history.history['accuracy'],
                'validation_accuracy': history.history['val_accuracy']}}

        # returns the last model
        return model

    def kmeans_clustering_query(
            self,
            preprocess=True,
            generate_plots=True,
            base_clusters=1):
        logger("Reading dataset...")
        # loads dataset and replaces n/a with zero
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)
        dataPandas = data.copy()

        if preprocess:
            logger("Preprocessing data...")
            data = np.asarray(structured_preprocesser(data))

        modelStorage = []
        inertiaStor = []

        # processes dataset and runs KMeans algorithm on one cluster as
        # baseline
        i = base_clusters
        logger("Creating unsupervised clustering task...")
        kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
        modelStorage.append(kmeans)

        # stores SSE values in an array for later comparison
        inertiaStor.append(kmeans.inertia_)
        i += 1

        logger("Identifying best centroid count and optimizing accuracy")
        # continues to increase cluster size until SSE values don't decrease by
        # 1000 - this value was decided based on precedence
        while(all(earlier >= later for earlier, later in zip(inertiaStor, inertiaStor[1:]))):
            kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
            modelStorage.append(kmeans)
            inertiaStor.append(kmeans.inertia_)
            # minimize inertia up to 10000
            i += 1

            # checks to see if it should continue to run; need to improve this
            # algorithm
            if i > 3 and inertiaStor[len(
                    inertiaStor) - 2] - 1000 <= inertiaStor[len(inertiaStor) - 1]:
                break

        # generates the clustering plots approiately
        if generate_plots:
            logger("Generating plots and storing in model")
            init_plots, plot_names = generate_clustering_plots(
                modelStorage[len(modelStorage) - 1], dataPandas, data)

            plots = {}

            for x in range(len(plot_names)):
                plots[str(plot_names[x])] = init_plots[x]

        # stores plots and information in the dictionary client model
        self.models['kmeans_clustering'] = {
            "model": modelStorage[len(modelStorage) - 1], "plots": plots}
        clearLog()
        # return modelStorage[len(modelStorage) - 1],
        # inertiaStor[len(inertiaStor) - 1], i

    def svm_query(
            self,
            instruction,
            test_size=0.2,
            kernel='linear',
            cross_val_size=0.3):
        logger("Reading in dataset....")
        # reads dataset and fills n/a values with zeroes
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)

        logger("Identifying target columns...")
        remove = get_similar_column(
            get_value_instruction(instruction), data)
        y = data[remove]
        del data[remove]

        # prepcoess the dataset
        logger("Preprocessing dataset")
        data = structured_preprocesser(data)
        #classification_column = get_similar_column(getLabelwithInstruction(instruction), data)

        num_classes = len(np.unique(y))

        # encodes the label dataset into 0's and 1's
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            data, y, test_size=test_size, random_state=49)

        # Fitting to SVM and storing in the model dictionary
        logger("Fitting Support Vector Machine")
        clf = svm.SVC(kernel=kernel)
        clf.fit(X_train, y_train)
        logger("Storing information in client object...")
        self.models["svm"] = {
            "model": clf,
            "accuracy_score": accuracy_score(
                clf.predict(X_test),
                y_test),
            "target": remove,
            "cross_val_score": cross_val_score(
                clf,
                data,
                y,
                cv=cross_val_size)}
        clearLog()
        return svm

    def nearest_neighbor_query(
            self,
            instruction,
            preprocess=True,
            min_neighbors=3,
            max_neighbors=10):
        logger("Reading in dataset....")
        #Reads in dataset
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)

        logger("Identifying target columns...")
        remove = get_similar_column(
            get_value_instruction(instruction), data)
        y = data[remove]
        del data[remove]

        # prepcoess the dataset
        if preprocess:
            logger("Preprocessing dataset...")
            data = structured_preprocesser(data)
        #classification_column = get_similar_column(getLabelwithInstruction(instruction), data)

        num_classes = len(np.unique(y))

        # encodes the label dataset into 0's and 1's
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            data, y, test_size=0.2, random_state=49)

        models = []
        scores = []
        logger("Fitting Nearest Neighbor...")
        logger("Identifying optimal number of neighbors...")
        # Tries all neighbor possibilities, based on either defaults or user
        # specified values
        for x in range(min_neighbors, max_neighbors):
            knn = KNeighborsClassifier(n_neighbors=x)
            knn.fit(X_train, y_train)
            models.append(knn)
            scores.append(accuracy_score(knn.predict(X_test), y_test))

        logger("Storing information in client object...")
        knn = models[scores.index(min(scores))]
        self.models["nearest_neighbors"] = {
            "model": knn, "accuracy_score": scores.index(
                min(scores)), "target": remove, "cross_val_score": cross_val_score(
                knn, data, y, cv=3)}

        clearLog()
        return knn

    def decision_tree_query(self, instruction, preprocess=True, test_size=0.2):
        logger("Reading in dataset....")
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)

        logger("Identifying target columns...")
        remove = get_similar_column(
            get_value_instruction(instruction), data)
        y = data[remove]
        del data[remove]

        # prepcoess the dataset
        if preprocess:
            logger("Preprocessing dataset...")
            data = structured_preprocesser(data)
        #classification_column = get_similar_column(getLabelwithInstruction(instruction), data)

        num_classes = len(np.unique(y))

        # encodes the label dataset into 0's and 1's
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        #fitting and storing
        logger("Fitting Decision Tree...")
        X_train, X_test, y_train, y_test = train_test_split(
            data, y, test_size=test_size, random_state=49)
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)

        logger("Storing information in client object...")
        self.models["decision_tree"] = {
            "model": clf,
            "target": remove,
            "cross_val_score": cross_val_score(
                clf,
                data,
                y,
                cv=3)}

        clearLog()
        return clf

    def allClassQuery(
            self,
            instruction,
            preprocess=True,
            test_size=0.2,
            random_state=49):
        logger("Reading in dataset....")
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)

        logger("Identifying target columns...")
        remove = get_similar_column(
            get_value_instruction(instruction), data)
        y = data[remove]
        del data[remove]

        # prepcoess the dataset
        if preprocess:
            logger("Preprocessing dataset...")
            data = structured_preprocesser(data)
        #classification_column = get_similar_column(getLabelwithInstruction(instruction), data)

        num_classes = len(np.unique(y))

        # encodes the label dataset into 0's and 1's
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            data, y, test_size=test_size, random_state=random_state)
        scores = []
        models = []

        # appends all models to the model list in order to evaluate best
        # accuracy
        logger("Testing various classification models....")
        models.append(self.decisionTreeQuery(instruction))
        models.append(self.nearestNeighborQuery(instruction))
        models.append(self.svmQuery(instruction))

        logger("Identifying top scores...")
        for model in models:
            scores.append(accuracy_score(model.predict(X_test), y_test))

        clearLog()

        # returns classificaiton model with the highest score
        return models[scores.index(max(scores))]

    def tune(self, model_to_tune):
        logger("Getting target model for tuning...")

        # checks to see which requested model is in the self.models
        for key in self.models:

            # processing for regression feed forward NN
            if key == 'regression_ANN':
                logger("Tuning model hyperparameters")
                returned_model = tuneReg(
                    self.dataset, self.models[key]["target"])
                self.models['regression_ANN'] = {'model': returned_model}
                return returned_model
            # processing for classification feed forward NN
            if key == model_to_tune and key == "classification_ANN":
                logger("Tuning model hyperparameters")
                returned_model = tuneClass(
                    self.models[key]["target"],
                    self.models[key]["num_classes"])
                self.models['classification_ANN'] = {'model': returned_model}
                return returned_model
            # processing for convolutional NN
            if key == model_to_tune and key == "convolutional_NN":
                logger("Tuning model hyperparameters")
                X = self.models['convolutional_NN']["X"]
                y = self.models['convolutional_NN']["y"]
                model = tuneCNN(
                    np.asarray(X),
                    np.asarray(y),
                    self.models["convolutional_NN"]["num_classes"])
                self.models["convolutional_NN"]["model"] = model

    def stat_analysis(self, column_name="none"):
        logger("Reading in dataset....")
        # Reading in dataset and creating pdtabulate variable to format outputs
        data = pd.read_csv(self.dataset)
        data.fillna(0, inplace=True)
        logger("Creating lambda object to format...")
        def pdtabulate(df): return tabulate(
            df, headers='keys', tablefmt='psql')

        logger("Identifying columns to transform....")

        # identifying categorical and numerical columns, and encoding
        # appropriately
        categor = data.select_dtypes(exclude=['int', 'float'])
        categor = categor.apply(LabelEncoder().fit_transform)
        for value in categor.columns:
            data[str(value)] = categor[str(value)]

        # if user doesn't specify column analysis on performed on the whole
        # dataset
        if column_name == "none":
            columns = []
            sim = []
            for first_val in data.columns:
                for sec_val in data.columns:
                    if first_val == sec_val:
                        continue
                    columns.append(str(first_val) + "_" + str(sec_val))
                    sim.append(1 - cosine(data[first_val], data[sec_val]))
                df = pd.DataFrame(columns=columns)
                df.loc[len(df)] = sim

            cols = []
            vals = []
            logger("Restructuring dataset for similarity...")
            # identifying top 5 feature importances and appending them to an
            # array for display
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
            # identifying columns to be compared
            for val in data.columns:
                if val == column_name:
                    continue
                columns.append(str(column_name) + "_" + str(val))
                sim.append(1 - cosine(data[column_name], data[val]))
            df = pd.DataFrame(columns=columns)
            df.loc[len(df)] = sim

            cols = []
            vals = []
            # identifying top 5 feature importances and appending them to a
            # dataset
            for val in np.argpartition(np.asarray(df.iloc[0]), -5)[-5:]:
                cols.append(df.columns[val])
                vals.append(df[df.columns[val]].iloc[0])
                frame = pd.DataFrame(columns=cols)
                frame.loc[len(df)] = vals

            # displaying the similarity spectrum and the formatted
            # data.describe()
            print("Similarity Spectrum")
            print("-------------------------")
            print(pdtabulate(frame))
            print()
            print("Dataset Description")
            print("-------------------------")
            print(pdtabulate(data[column_name]).describe())

    def convolutional_query(self, *argv):
        logger("Creating CNN generation query")
        # generates the dataset based on instructions using a selenium query on
        # google chrome
        logger("Generating datasets for classes...")
        input_shape = (224, 224, 3)
        #Assuming Downloaded Images in current Directory
        data_path=os.getcwd()
        num_classes=0
        loss_func=""
        for a_class in argv:
            num_classes = num_classes + 1
        if num_classes>2:
            loss_func="categorical_crossentropy"
        elif num_classes==2:
            loss_func="binary_crossentropy"
        
        logger("Creating convolutional neural network dynamically...")
        #Convolutional Neural Network
        model = Sequential()
        model.add(
            Conv2D(
                64,
                kernel_size=3,
                activation="relu",
                input_shape=(input_shape)))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Conv2D(64, kernel_size=3, activation="relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Flatten())
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss=loss_func,
            metrics=['accuracy'])
        
        train_data = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
                                   
        X_train = train_data.flow_from_directory(data_path+'/training_set',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
        test_data=ImageDataGenerator(rescale = 1./255)
        X_test = test_data.flow_from_directory(data_path+'/test_set',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
		#Fitting/Training the model
        print(X_train)
        history=model.fit_generator(generator=X_train,
                    steps_per_epoch=X_train.n//X_train.batch_size,
                    validation_data=X_test,
                    validation_steps=X_test.n//X_test.batch_size,
                    epochs=10,
                    verbose=0
         )
        # storing values the model dictionary
        self.models["convolutional_NN"] = {
            "model": model,
            'num_classes': (2 if num_classes==1 else num_classes),
            'losses': {
                'training_loss': history.history['loss'],
                'val_loss': history.history['val_loss']},
            'accuracy': {
                'training_accuracy': history.history['accuracy'],
                'validation_accuracy': history.history['val_accuracy']}}

    def dimensionality_reducer(self, instruction):
        dimensionality_reduc(instruction, self.dataset)

    def show_plots(self, model):
        print(self.models[model]['plots'].keys())


newClient = client('./data/housing.csv').regression_query_ann('Model median house value')

