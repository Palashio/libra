import sys
import os 
import pandas as pd 

sys.path.insert(1, './preprocessing')
sys.path.insert(1, './data_generation')
sys.path.insert(1, './modeling')
sys.path.insert(1, './plotting')

from data_reader import DataReader
from grammartree import get_value_instruction
from sklearn import preprocessing, svm, tree
from sklearn.metrics import accuracy_score
from data_preprocesser import structured_preprocesser, initial_preprocesser, clustering_preprocessor
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from supplementaries import generate_id
from generate_plots import (generate_clustering_plots,
                           generate_regression_plots,
                           generate_classification_plots)

currLog = ""
counter = 0

def clearLog():
    global currLog
    global counter

    currLog = ""
    counter = 0


# logging function that creates hierarchial display of the processes of
# different functions. Copied into different python files to maintain
# global variable parallels


def logger(instruction, found="", slash=''):
    global currLog
    global counter
    if counter == 0:
        currLog += (" " * 2 * counter) + str(instruction) + str(found)
    elif instruction == "->":
        counter = counter - 1
        if slash == '|':
            currLog += (" " * 2 * counter) + slash + str(found)
        else:
            currLog += (" " * 2 * counter) + str(instruction) + str(found)
    else:
        currLog += (" " * 2 * counter) + "|" + "\n"
        currLog += (" " * 2 * counter) + "|- " + str(instruction) + str(found)
        if instruction == "done...":
            currLog += "\n" + "\n"

    counter += 1
    if instruction == "->":
        print(currLog, end="")
    else:
        print(currLog)
    currLog = ""

def k_means_clustering(dataset= None,
            preprocess=True,
            generate_plots=True,
            drop=None,
            base_clusters=1):
        logger("Reading dataset...")
        # loads dataset and replaces n/a with zero
        # data = pd.read_csv(self.dataset)

        dataReader = DataReader(dataset)
        data = dataReader.data_generator()

        if drop is not None:
            data.drop(drop, axis=1, inplace=True)

        dataPandas = data.copy()

        full_pipeline = None
        if preprocess:
            logger("Preprocessing data...")
            data, full_pipeline = clustering_preprocessor(data)
            data = np.array(data)

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
        while (all(earlier >= later for earlier,
                                        later in zip(inertiaStor, inertiaStor[1:]))):
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
        return {
            'id': generate_id(),
            "model": modelStorage[len(modelStorage) - 1],
            "preprocesser": full_pipeline,
            "plots": plots}
        clearLog()


def train_svm(instruction,
        dataset=None,
            test_size=0.2,
            kernel='linear',
            preprocess=True,
            drop=None,
            cross_val_size=0.3):

        logger("Reading in dataset....")
        # reads dataset and fills n/a values with zeroes
        #data = pd.read_csv(self.dataset)

        dataReader = DataReader(dataset)
        data = dataReader.data_generator()

        if drop is not None:
            data.drop(drop, axis=1, inplace=True)

        data, y, target, full_pipeline = initial_preprocesser(data, instruction, preprocess)


        X_train = data['train']
        y_train = y['train']
        X_test = data['test']
        y_test = y['test']

        # classification_column = get_similar_column(getLabelwithInstruction(instruction), data)
        num_classes = len(np.unique(y))

        # Needed to make a custom label encoder due to train test split changes
        # Can still be inverse transformed, just a bit of extra work
        y_vals = np.unique(pd.concat([y['train'], y['test']], axis=0))
        label_mappings = {}
        for i in range(len(y_vals)):
            label_mappings[y_vals[i]] = i

        y_train = y_train.apply(lambda x: label_mappings[x]).values
        y_test = y_test.apply(lambda x: label_mappings[x]).values


        # Fitting to SVM and storing in the model dictionary
        logger("Fitting Support Vector Machine")
        clf = svm.SVC(kernel=kernel)
        clf.fit(X_train, y_train)
        logger("Storing information in client object...")
        return {
            'id': generate_id(),
            "model": clf,
            "accuracy_score": accuracy_score(
                clf.predict(X_test),
                y_test),
            "target": target,
            "preprocesser": full_pipeline,
            "interpreter": label_mappings,
            "cross_val_score": cross_val_score(
                clf,
                X_train,
                y_train)}
        clearLog()


def nearest_neighbors(instruction=None,
            dataset = None, 
            preprocess=True,
            drop=None,
            min_neighbors=3,
            max_neighbors=10):
        logger("Reading in dataset....")
        # Reads in dataset
        # data = pd.read_csv(self.dataset)
        dataReader = DataReader(dataset)
        data = dataReader.data_generator()
        if drop is not None:
            data.drop(drop, axis=1, inplace=True)
        data, y, remove, full_pipeline = initial_preprocesser(
            data, instruction, preprocess)
        X_train = data['train']
        y_train = y['train']
        X_test = data['test']
        y_test = y['test']
        # classification_column = get_similar_column(getLabelwithInstruction(instruction), data)
        num_classes = len(np.unique(y))
        # encodes the label dataset into 0's and 1's
        y_vals = np.unique(pd.concat([y['train'], y['test']], axis=0))
        label_mappings = {}
        for i in range(len(y_vals)):
            label_mappings[y_vals[i]] = i
        y_train = y_train.apply(lambda x: label_mappings[x]).values
        y_test = y_test.apply(lambda x: label_mappings[x]).values
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
        return {
            'id': generate_id(),
            "model": knn, "accuracy_score": scores.index(
                min(scores)),
            "preprocesser": full_pipeline,
            "interpreter": label_mappings,
            "target": remove, "cross_val_score": cross_val_score(
                knn, X_train, y_train, cv=3)}
        clearLog()

def decision_tree(instruction,
            dataset=None,
            preprocess=True,
            test_size=0.2,
            drop=None):
    logger("Reading in dataset....")

    dataReader = DataReader(dataset)
    data = dataReader.data_generator()

    if drop is not None:
        data.drop(drop, axis=1, inplace=True)

    data, y, remove, full_pipeline = initial_preprocesser(
        data, instruction, preprocess)

    X_train = data['train']
    y_train = y['train']
    X_test = data['test']
    y_test = y['test']

    # classification_column = get_similar_column(getLabelwithInstruction(instruction), data)

    # Needed to make a custom label encoder due to train test split changes
    # Can still be inverse transformed, just a bit of extra work
    y_vals = np.unique(pd.concat([y['train'], y['test']], axis=0))
    label_mappings = {}
    for i in range(len(y_vals)):
        label_mappings[y_vals[i]] = i

    # Custom label encoder due to train test split
    y_train = y_train.apply(lambda x: label_mappings[x]).values
    y_test = y_test.apply(lambda x: label_mappings[x]).values
    num_classes = len(np.unique(y))

    # fitting and storing
    logger("Fitting Decision Tree...")

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    logger("Storing information in client object...")

    clearLog()

    return {
            'id': generate_id(),
            "model": clf,
            "target": remove,
            "accuracy_score": accuracy_score(
                clf.predict(X_test),
                y_test),
            "preprocesser": full_pipeline,
            "interpeter": label_mappings,
            "cross_val_score": cross_val_score(
                clf,
                X_train,
                y_train,
                cv=3)}
