import pandas as pd
from libra.preprocessing.data_reader import DataReader
from sklearn import svm, tree
from sklearn.metrics import accuracy_score
from libra.preprocessing.data_preprocesser import initial_preprocesser, clustering_preprocessor
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from libra.query.supplementaries import generate_id
from libra.plotting.generate_plots import (generate_clustering_plots)
from colorama import Fore, Style
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

counter = 0
currLog = ""


# # allows for all columns to be displayed when printing()
# pd.options.display.width = None


# # clears the log when new process is started up


def clearLog():
    global counter
    global currLog

    currLog = ""
    counter = 0


# logging function that creates hierarchial display of the processes of
# different functions. Copied into different python files to maintain
# global variable parallels
def logger(instruction, found=""):
    '''
    logging function that creates hierarchial display of the processes of
    different functions. Copied into different python files to maintain
    global variables.

    :param instruction: what you want to be displayed
    :param found: if you want to display something found like target column
    '''
    
    global counter
    if counter == 0:
        print((" " * 2 * counter) + str(instruction) + str(found))
    elif instruction == "->":
        counter = counter - 1
        print(Fore.BLUE + (" " * 2 * counter) +
              str(instruction) + str(found) + (Style.RESET_ALL))
    else:
        print((" " * 2 * counter) + "|- " + str(instruction) + str(found))
        if instruction == "done...":
            print("\n" + "\n")

    counter += 1

# function that prints a table consisting of a summary of the columns founnd
# in the dataset (is copied into different python files to maintain global variable
# parallels
def printtable(col_name, col_width):
    '''
    function that prints a table consisting of a summary of the columns found
    in the dataset (is copied into different python files to maintain global variable
    parallels)

    :param col_name: name of the column in dataset one wishes to see displayed
    :param col_width: width of the column in dataset one wishes to see displayed
    '''
    
    global counter
    for row in col_name:
        print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
                                                    for word in row)) + " |")

def k_means_clustering(dataset=None,
                       scatters=[],
                       preprocess=True,
                       generate_plots=True,
                       drop=None,
                       base_clusters=1,
                       verbose=0,
                       n_init=10,
                       max_iter=300,
                       random_state=42,
                       text=[]):
    '''
    function to train a k means clustering algorithm
    :param many params: used to hyperparametrize the function.
    :return a dictionary object with all of the information for the algorithm.
    '''
        
    logger("Reading in dataset")

    dataReader = DataReader(dataset)
    data = dataReader.data_generator()

    if drop is not None:
        data.drop(drop, axis=1, inplace=True)

    dataPandas = data.copy()

    full_pipeline = None
    if preprocess:
        logger("Preprocessing data")
        data, full_pipeline = clustering_preprocessor(data)
        data = np.array(data)

    modelStorage = []
    inertiaStor = []

    # processes dataset and runs KMeans algorithm on one cluster as
    # baseline
    i = base_clusters
    logger("Creating unsupervised clustering task")
    kmeans = KMeans(
        n_clusters=i,
        random_state=random_state,
        verbose=verbose,
        n_init=n_init,
        max_iter=max_iter).fit(data)
    modelStorage.append(kmeans)
    # stores SSE values in an array for later comparison
    inertiaStor.append(kmeans.inertia_)

    logger("Identifying best centroid count and optimizing accuracy")

    col_name=[["Number of clusters   ",
                     "| Inertia  "]] 
    col_width=max(len(word) for row in col_name for word in row) + 2
    printtable(col_name,col_width)
    values = []
    values.append(str(i))
    values.append(
        "| " + str(inertiaStor[i-base_clusters]))
    datax = []
    datax.append(values)
    printtable(datax,
               col_width)
    
    i += 1

    # continues to increase cluster size until SSE values don't decrease by
    # 1000 - this value was decided based on precedence
    while (all(earlier >= later for earlier,
               later in zip(inertiaStor, inertiaStor[1:]))):
        kmeans = KMeans(
            n_clusters=i,
            random_state=random_state,
            verbose=verbose,
            n_init=n_init,
            max_iter=max_iter).fit(data)
        modelStorage.append(kmeans)
        inertiaStor.append(kmeans.inertia_)
        
        values = []
        values.append(str(i))
        values.append(
            "| " + str(inertiaStor[i-base_clusters]))
        datax = []
        datax.append(values)
        printtable(datax,
                   col_width)

        # minimize inertia up to 10000
        i += 1

        # checks to see if it should continue to run; need to improve this
        # algorithm
        if i > 3 and inertiaStor[len(
                inertiaStor) - 2] - 1000 <= inertiaStor[len(inertiaStor) - 1]:
            print()
            break
    
    # generates the clustering plots approiately
    logger("->", "Optimal number of clusters found: {}".format(i))
    logger(
        "->", "Final inertia of {}".format(inertiaStor[len(inertiaStor) - 1]))

    plots = {}
    if generate_plots:
        logger("Generating plots and storing in model")
        init_plots, plot_names, elbow = generate_clustering_plots(modelStorage[len(
            modelStorage) - 1], dataPandas, data, scatters, inertiaStor, base_clusters)
        for x in range(len(plot_names)):
            plots[str(plot_names[x])] = init_plots[x]
        plots['elbow'] = elbow

    logger("Stored model under 'k_means_clustering' key")
    clearLog()
    # stores plots and information in the dictionary client model
    return {
        'id': generate_id(),
        "model": modelStorage[len(modelStorage) - 1],
        "preprocesser": full_pipeline,
        "plots": plots}


def train_svm(instruction,
              dataset=None,
              test_size=0.2,
              kernel='linear',
              text=[],
              preprocess=True,
              ca_threshold=None,
              drop=None,
              cross_val_size=0.3,
              degree=3,
              gamma='scale',
              coef0=0.0,
              max_iter=-1,
              random_state=49):
    '''
    function to train a support vector machine clustering algorithm
    :param many params: used to hyperparametrize the function.
    :return a dictionary object with all of the information for the algorithm.
    '''

    logger("Reading in dataset")

    dataReader = DataReader(dataset)
    data = dataReader.data_generator()

    if drop is not None:
        data.drop(drop, axis=1, inplace=True)

    logger("Preprocessing data")
    data, y, target, full_pipeline = initial_preprocesser(
        data, instruction, preprocess, ca_threshold, text, test_size=test_size, random_state=random_state)
    logger("->", "Target column found: {}".format(target))

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
    logger("Labels being mapped to appropriate classes")
    for i in range(len(y_vals)):
        label_mappings[y_vals[i]] = i

    y_train = y_train.apply(lambda x: label_mappings[x]).values
    y_test = y_test.apply(lambda x: label_mappings[x]).values

    # Fitting to SVM and storing in the model dictionary
    logger("Fitting Support Vector Machine")
    clf = svm.SVC(
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        max_iter=max_iter)
    clf.fit(X_train, y_train)

    score = accuracy_score(
        clf.predict(X_test),
        y_test)

    logger("->", "Accuracy found on testing set: {}".format(score))

    logger('->', "Stored model under 'svm' key")
    clearLog()
    return {
        'id': generate_id(),
        "model": clf,
        "accuracy": {
            'cross_val_score': cross_val_score(
                clf,
                X_train,
                y_train),
            'accuracy_score': score},
        "target": target,
        "preprocesser": full_pipeline,
        "interpreter": label_mappings,
        'test_data': {
            'X': X_test,
            'y': y_test}}
    clearLog()


def nearest_neighbors(instruction=None,
                      dataset=None,
                      ca_threshold=None,
                      preprocess=True,
                      drop=None,
                      min_neighbors=3,
                      max_neighbors=10,
                      leaf_size=30,
                      p=2,
                      test_size=0.2,
                      random_state=49,
                      algorithm='auto',
                      text=[]):
    '''
    function to train a nearest neighbor algorithm
    :param many params: used to hyperparametrize the function.
    :return a dictionary object with all of the information for the algorithm.
    '''
    
    logger("Reading in dataset")
    # Reads in dataset
    # data = pd.read_csv(self.dataset)
    dataReader = DataReader(dataset)
    data = dataReader.data_generator()
    if drop is not None:
        data.drop(drop, axis=1, inplace=True)
    logger("Preprocessing data")
    data, y, remove, full_pipeline = initial_preprocesser(
        data, instruction, preprocess, ca_threshold, text, test_size=test_size, random_state=random_state)
    logger("->", "Target column found: {}".format(remove))
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
    logger("Labels being mapped to appropriate classes")
    models = []
    scores = []
    logger("Fitting nearest neighbors model")
    logger("Identifying optimal number of neighbors")
    # Tries all neighbor possibilities, based on either defaults or user
    # specified values
    num_neighbors = []
    for x in range(min_neighbors, max_neighbors):
        knn = KNeighborsClassifier(
            n_neighbors=x,
            leaf_size=leaf_size,
            p=p,
            algorithm=algorithm)
        knn.fit(X_train, y_train)
        models.append(knn)
        scores.append(accuracy_score(knn.predict(X_test), y_test))
        num_neighbors.append(x)

    logger("->", "Optimal number of neighbors found: {}".format(
        num_neighbors[scores.index(max(scores))]))
    logger(
        "->", "Accuracy found on testing set: {}".format(scores[scores.index(max(scores))]))
    logger("Stored model under 'nearest_neighbors' key")
    knn = models[scores.index(min(scores))]
    clearLog()
    return {'id': generate_id(),
            "model": knn,
            "accuracy": {'accuracy_score': scores[scores.index(max(scores))],
                         'cross_val_score': cross_val_score(knn,
                                                            X_train,
                                                            y_train,
                                                            cv=3)},
            "preprocesser": full_pipeline,
            "interpreter": label_mappings,
            'test_data': {'X': X_test,
                          'y': y_test},
            "target": remove}
    clearLog()


def decision_tree(instruction,
                  dataset=None,
                  preprocess=True,
                  ca_threshold=None,
                  text=[],
                  test_size=0.2,
                  drop=None,
                  criterion='gini',
                  splitter='best',
                  max_depth=None,
                  min_samples_split=2,
                  min_samples_leaf=1,
                  min_weight_fraction_leaf=0.0,
                  max_leaf_nodes=None,
                  min_impurity_decrease=0.0,
                  ccp_alpha=0.0):
    '''
    function to train a decision tree algorithm.
    :param many params: used to hyperparametrize the function.
    :return a dictionary object with all of the information for the algorithm.
    '''
    logger("Reading in dataset")

    dataReader = DataReader(dataset)
    data = dataReader.data_generator()
    logger("Preprocessing data")
    if drop is not None:
        data.drop(drop, axis=1, inplace=True)

    data, y, remove, full_pipeline = initial_preprocesser(
        data, instruction, preprocess, ca_threshold, text)
    logger("->", "Target column found: {}".format(remove))

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

    logger("Labels being mapped to appropriate classes")
    num_classes = len(np.unique(y))

    # fitting and storing
    logger("Fitting Decision Tree")

    clf = tree.DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        ccp_alpha=ccp_alpha)
    clf = clf.fit(X_train, y_train)

    score = accuracy_score(
        clf.predict(X_test),
        y_test)
    logger("->", "Score found on testing set: {}".format(score))
    logger("Stored model under 'decision_tree' key")
    clearLog()

    return {
        'id': generate_id(),
        "model": clf,
        "target": remove,
        "accuracy": {'cross_val_score': cross_val_score(
            clf,
            X_train,
            y_train,
            cv=3), 'accuracy_score': score},
        "accuracy_score": score,
        "preprocesser": full_pipeline,
        "interpreter": label_mappings,
        'test_data': {'X': X_test, 'y': y_test}}
