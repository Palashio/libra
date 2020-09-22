from libra.query.nlp_queries import (image_caption_query,
                                     generate_caption, classify_text,
                                     text_classification_query, get_summary,
                                     summarization_query, generate_text, get_ner)

from libra.query.classification_models import (k_means_clustering,
                                               train_svm, nearest_neighbors,
                                               decision_tree, train_xgboost)
from libra.query.supplementaries import tune_helper, get_model_data, get_operators, get_accuracy, get_losses, \
    get_target, get_plots, get_vocab

from libra.query.feedforward_nn import (regression_ann,
                                        classification_ann,
                                        convolutional)
from libra.data_generation.grammartree import get_value_instruction
from libra.data_generation.dataset_labelmatcher import (get_similar_column,
                                                        get_similar_model)
from libra.query.generative_models import dcgan

from libra.plotting.generate_plots import analyze
from libra.query.recommender_systems import ContentBasedRecommender
from libra.dashboard.auto_eda import edaDashboard
from colorama import Fore, Style
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import warnings
import os
import nltk
import ssl
import numpy as np
from tkinter import filedialog
from tkinter import *
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from matplotlib import pyplot as plt

# suppressing warnings for cleaner dialogue box
warnings.simplefilter(action='error', category=FutureWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# function imports from other files
currLog = ""
counter = 0


# clears log when needed - currently not being used
def clearLog():
    global currLog
    global counter

    currLog = ""
    counter = 0
    print("")


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


def get_folder_dir(self):
    dir_path = filedialog.askdirectory()
    return dir_path


def get_file():
    filename = filedialog.askopenfilename()
    if os.path.isfile(filename):
        return filename
    else:
        print('No file chosen')


class client:
    '''
    class to store all query information. Currently, old_models is not being used.
    '''

    def __init__(self, data):
        '''
        initializer for the client class, reads in dataset and records by calling logger function
        :param data: represents the dataset that you're trying to read
        :return: a completely initialized client class
        '''
        self.required_installations()
        logger("Creating client object")
        self.dataset = data
        logger("Reading in dataset")
        print("")
        self.models = {}
        self.latest_model = None
        clearLog()

    def required_installations(self):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)

    # param model_requested: string representation of the name of the model user seeks to retrieve
    # returns models with a specific string - currently deprecated, should not be used.

    def get_models(self, model_requested):
        '''
        returns models with a specific string - currently deprecated, should not be used.
        :param model_requested: represents the name of the model from which you want to retrieve
        :return: the model dictionary for your specific model
        '''
        logger("Getting model...")
        return get_similar_model(model_requested, self.models.keys())
        clearLog()

    # recommend items based on search criteria(for recommender systems only)

    def recommend(self, search_term):
        if self.latest_model == 'content_recommender':
            model = self.models[self.latest_model]
            return model.recommend(search_term)
        else:
            pass

    # param modelKey: string representation of the model to make prediction
    # param data: dataframe version of desired prediction set
    def predict(self, data, model=None):
        '''
        Uses a model from the self.models dictionary to make a prediction. Also fits it based on the operator stored in the models dictionary.
        :param data: is the data that you want to predict for using model
        :param model: is the specific model you want to use to predict
        :return: a prediction, most likely an array
        '''
        if model is None:
            model = self.latest_model
        if model == 'text_classification':
            map_func = np.vectorize(lambda x: self.classify_text(x))
            predictions = map_func(data)
            return predictions
        else:
            modeldict = self.models[model]

            if modeldict.get('preprocesser'):
                data = modeldict['preprocesser'].transform(data)
            predictions = modeldict['model'].predict(data)
        clearLog()
        return self.interpret(model, predictions)

    def interpret(self, model, predictions):
        '''
        Function to interpret predictions from a neural network for the creation for graphs / user understanding.
        :param model: is the model in the self.models dictionary that you want to use to interpret
        :param predictions: the predictions that come out of the model
        :return: a prediction, most likely an array
        '''
        modeldict = self.models[model]
        if modeldict.get('interpreter'):
            predictions = modeldict['interpreter'].inverse_transform(
                predictions)
        clearLog()
        return predictions

    # determines type of solution based of type of problem posed by query using a feed-forward neural network
    # instruction should be the value of a column
    def neural_network_query(self,
                             instruction,
                             callback=False,
                             text=[],
                             ca_threshold=None,
                             drop=None,
                             preprocess=True,
                             test_size=0.2,
                             random_state=49,
                             epochs=50,
                             generate_plots=True,
                             callback_mode='min',
                             maximizer="val_loss",
                             save_model=False,
                             save_path=os.getcwd(),
                             add_layer={}):
        '''
        Detects to see if it's a regression/classification problem and then calls the correct query.
        :param hyperparameters: all of these are hyperparameters that're passed to the algorithm
        :return: a model, plots, accuracy information all stored in the self.models dictionary
        '''

        data = pd.read_csv(self.dataset)

        if preprocess:

            remove = get_similar_column(
                get_value_instruction(instruction), data)

            if len(data) < 50:
                raise Exception(
                    "Only datasets larger then 50 rows are supported for neural networks")
            if len(data[remove].value_counts()) <= 50:
                callback_mode = 'max'
                maximizer = "val_accuracy"
                self.classification_query_ann(
                    instruction,
                    text=text,
                    callback=callback,
                    ca_threshold=ca_threshold,
                    preprocess=preprocess,
                    test_size=test_size,
                    random_state=random_state,
                    epochs=epochs,
                    generate_plots=generate_plots,
                    callback_mode=callback_mode,
                    maximizer=maximizer,
                    save_model=save_model,
                    save_path=save_path,
                    add_layer=add_layer)
            else:
                self.regression_query_ann(
                    instruction,
                    callback=callback,
                    text=text,
                    ca_threshold=ca_threshold,
                    preprocess=preprocess,
                    test_size=test_size,
                    random_state=random_state,
                    epochs=epochs,
                    generate_plots=generate_plots,
                    callback_mode=callback_mode,
                    maximizer=maximizer,
                    drop=drop,
                    save_model=save_model,
                    save_path=save_path,
                    add_layer=add_layer)
        clearLog()

    # single regression query using a feed-forward neural network
    # instruction should be the value of a column
    def regression_query_ann(
            self,
            instruction,
            callback=False,
            text=[],
            drop=None,
            ca_threshold=None,
            preprocess=True,
            test_size=0.2,
            random_state=49,
            epochs=50,
            generate_plots=True,
            callback_mode='min',
            maximizer="val_loss",
            save_model=True,
            save_path=os.getcwd(),
            add_layer={}):
        '''
        Calls the body of the regression_query__ code in the supplementaries.py file. Used for a regression feed forward neural network.
        :param instruction: The objective that you want to model (str).
        :param callback: Applying a set of functions/actions at various stages of training (bool).
        :param ca_threshold: Threshold for multiple correspondence analysis (float).
        :param dataset: The dataset being used in the regression feed forward neural network (str).
        :param text: A list of columns to perform text embedding on.
        :param drop: A list of the dataset's columns to drop.
        :param preprocess: Preprocess the data (bool).
        :param test_size: Size of the testing set (float).
        :param random_state: Initialize a pseudo-random number generator (int).
        :param epochs: Number of epochs (int).
        :param generate_plots: Generate plots for the model (bool).
        :param callback_mode: The type of callback (str).
        :param maximizer: The accuracy/loss type to optimize (str).
        :param save_model: Save the model (bool).
        :param save_path: Filepath of where to save the model (str).

        :return: a model and information to along with it stored in the self.models dictionary.
        '''

        self.models['regression_ANN'] = regression_ann(
            instruction=instruction,
            callback=False,
            ca_threshold=.25 if ca_threshold is None else ca_threshold,
            dataset=self.dataset,
            text=text,
            drop=drop,
            preprocess=preprocess,
            test_size=test_size,
            random_state=random_state,
            epochs=epochs,
            generate_plots=generate_plots,
            callback_mode=callback_mode,
            maximizer=maximizer,
            save_model=save_model,
            save_path=save_path,
            add_layer=add_layer)

        self.latest_model = 'regression_ANN'
        clearLog()

    # query for multilabel classification query, does not work for
    # binaryclassification, fits to feed-forward neural network

    def classification_query_ann(
            self,
            instruction,
            callback=False,
            text=[],
            ca_threshold=None,
            preprocess=True,
            callback_mode='min',
            drop=None,
            random_state=49,
            test_size=0.2,
            epochs=50,
            generate_plots=True,
            maximizer="val_loss",
            save_model=False,
            save_path=os.getcwd(),
            add_layer={}):
        '''
        Calls the body of the classification code in the supplementaries.py file. Used for a classification feed forward neural network.
        :param instruction: The objective that you want to model (str).
        :param callback: Applying a set of functions/actions at various stages of training (bool).
        :param dataset: The dataset being used in the classification feed forward neural network (str).
        :param text: A list of columns to perform text embedding on.
        :param ca_threshold: Threshold for multiple correspondence analysis (float).
        :param drop: A list of the dataset's columns to drop.
        :param preprocess: Preprocess the data (bool).
        :param test_size: Size of the testing set (float).
        :param random_state: Initialize a pseudo-random number generator (int).
        :param epochs: Number of epochs (int).
        :param generate_plots: Generate plots for the model (bool).
        :param callback_mode: The type of callback (str).
        :param maximizer: The accuracy/loss type to optimize (str).
        :param save_model: Save the model (bool).
        :param save_path: Filepath of where to save the model (str).

        :return: a model and information to along with it stored in the self.models dictionary.
        '''

        self.models['classification_ANN'] = classification_ann(
            instruction=instruction,
            callback=callback,
            dataset=self.dataset,
            text=text,
            ca_threshold=.25 if ca_threshold is None else ca_threshold,
            drop=drop,
            preprocess=preprocess,
            test_size=test_size,
            random_state=random_state,
            epochs=epochs,
            generate_plots=generate_plots,
            callback_mode=callback_mode,
            maximizer=maximizer,
            save_model=save_model,
            save_path=save_path,
            add_layer=add_layer)

        self.latest_model = 'classification_ANN'
        clearLog()

    # query to perform k-means clustering

    def kmeans_clustering_query(self,
                                preprocess=True,
                                scatters=[],
                                generate_plots=True,
                                drop=None,
                                clusters=None,
                                base_clusters=2,
                                verbose=0,
                                n_init=10,
                                max_iter=300,
                                random_state=42,
                                text=[]
                                ):
        '''
        Calls the body of the kmeans_clustering code in the supplementaries.py file. Can be used without any preprocessing and/or parameters.

        :param dataset: The dataset being used in the k-means clustering algorithm (str).
        :param scatters: A list of various types of scatter plots.
        :param preprocess: Preprocess the data (bool).
        :param generate_plots: Generate plots for the model (bool).
        :param drop: A list of the dataset's columns to drop.
        :param base_clusters: Number of clusters to generate (int).
        :param verbose: Printing the logging information (int).
        :param n_init: Number of times the function will run with different seeds (int).
        :param max_iter: Maximum number of iterations the function will run (int).
        :param random_state: Initialize a pseudo-random number generator (int).
        :param text: A list of columns to perform text embedding on.

        :return: a model and information to along with it stored in the self.models dictionary.
        '''

        self.models['k_means_clustering'] = k_means_clustering(
            dataset=self.dataset,
            scatters=scatters,
            clusters=clusters,
            preprocess=preprocess,
            generate_plots=generate_plots,
            drop=drop,
            base_clusters=base_clusters,
            verbose=verbose,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            text=text
        )

        self.latest_model = 'k_means_clustering'
        clearLog()

    # query to create a support vector machine

    def svm_query(self,
                  instruction,
                  test_size=0.2,
                  text=[],
                  random_state=49,
                  kernel='linear',
                  preprocess=True,
                  drop=None,
                  cross_val_size=0.3,
                  degree=3,
                  gamma='scale',
                  coef0=0.0,
                  max_iter=-1
                  ):
        '''
        Calls the body of the svm query code in the supplementaries.py file. Used to create a classification support vector machine.
        :param dataset: The dataset being used in the classification support vector machine (str).
        :param text: A list of columns to perform text embedding on.
        :param random_state: Initialize a pseudo-random number generator (int).
        :param test_size: Size of the testing set (float).
        :param kernel: The type of kernel to be used (str).
        :param preprocess: Preprocess the data (bool).
        :param drop: A list of the dataset's columns to drop.
        :param cross_val_size: Cross-Validation score (float).
        :param degree: Degree of the polynomial kernel function (int).
        :param gamma: Kernel coefficient (int).
        :param coef0: Significant term in 'poly' and 'sigmoid' kernel functions (float).
        :param max_iter: Maximum number of iterations the function will run (int).


        :return: a model and information to go along with it stored in the self.models dictionary.
        '''

        self.models['svm'] = train_svm(instruction,
                                       dataset=self.dataset,
                                       text=text,
                                       random_state=random_state,
                                       test_size=test_size,
                                       kernel=kernel,
                                       preprocess=preprocess,
                                       drop=drop,
                                       cross_val_size=cross_val_size,
                                       degree=degree,
                                       gamma=gamma,
                                       coef0=coef0,
                                       max_iter=max_iter
                                       )

        self.latest_model = 'svm'
        clearLog()

    # query to create a nearest neighbors model

    def nearest_neighbor_query(
            self,
            instruction=None,
            text=[],
            random_state=49,
            test_size=0.2,
            preprocess=True,
            drop=None,
            min_neighbors=3,
            max_neighbors=10,
            leaf_size=30,
            p=2,
            algorithm='auto'
    ):
        '''
        Calls the body of the nearest neighbor code in the supplementaries.py file. Used to create a nearest neighbor algorithm.
        :param instruction: The objective that you want to model (str).
        :param text: A list of columns to perform text embedding on.
        :param random_state: Initialize a pseudo-random number generator (int).
        :param test_size: Size of the testing set (float).
        :param dataset: The dataset being used in the nearest neighbor algorithm (str).
        :param preprocess: Preprocess the data (bool).
        :param drop: A list of the dataset's columns to drop.
        :param min_neighbors: Minimum number of neighbors (int).
        :param max_neighbors: Maximum number of neighbors (int).
        :param leaf_size: Leaf size passed to BallTree or KDTree (int).
        :param p: Power parameter for the Minkowski metric (int).
        :param algorithm: Algorithm used to compute the nearest neighbors (str).


        :return: a model and information to along with it stored in the self.models dictionary.
        '''
        self.models['nearest_neighbor'] = nearest_neighbors(
            instruction=instruction,
            text=text,
            random_state=random_state,
            test_size=test_size,
            dataset=self.dataset,
            preprocess=preprocess,
            drop=drop,
            min_neighbors=min_neighbors,
            max_neighbors=max_neighbors,
            leaf_size=leaf_size,
            p=p,
            algorithm=algorithm
        )

        self.latest_model = 'nearest_neighbor'
        clearLog()

    # query to create a decision tree model

    def decision_tree_query(
            self,
            instruction,
            preprocess=True,
            test_size=0.2,
            text=[],
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
        Calls the body of the decision tree code in the classification_models.py file. Used to create a decision tree algorithm.
        :param instruction: The objective that you want to model (str).
        :param text: A list of columns to perform text embedding on.
        :param dataset: The dataset being used in the decision tree algorithm (str).
        :param preprocess: Preprocess the data (bool).
        :param test_size: Size of the testing set (float).
        :param drop: A list of the dataset's columns to drop.
        :param criterion: The function to measure the quality of a split (str).
        :param splitter: The technique used to choose each node's split (str).
        :param max_depth: The maximum depth of the tree (int).
        :param min_samples_split: The minimum number of samples a node must have to split (int).
        :param min_samples_leaf: The minimum number of samples a leaf node must have (int).
        :param min_weight_fraction_leaf: The fraction of the input samples required to be at a leaf node (float).
        :param max_leaf_nodes: Maximum number of leaf nodes (int).
        :param min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than or equal
         to this value (float).
        :param ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning (float).


        :return: a model and information to along with it stored in the self.models dictionary.
        '''

        self.models['decision_tree'] = decision_tree(
            instruction=instruction,
            text=text,
            dataset=self.dataset,
            preprocess=preprocess,
            test_size=test_size,
            drop=drop,
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha)

        self.latest_model = 'decision_tree'
        clearLog()

    def content_recommender_query(self, feature_names=[], n_recommendations=10, indexer='title'):
        self.models['content_recommender'] = ContentBasedRecommender(
            data=self.dataset,
            feature_names=feature_names,
            indexer=indexer)

        self.latest_model = 'content_recommender'
        clearLog()

    # query to create a xgboost model

    def xgboost_query(self,
                      instruction,
                      text=[],
                      preprocess=True,
                      test_size=0.2,
                      drop=None,
                      random_state=49,
                      learning_rate=0.1,
                      n_estimators=1000,
                      max_depth=6,
                      min_child_weight=1,
                      gamma=0,
                      subsample=0.8,
                      colsample_bytree=0.8,
                      verbosity=0,
                      objective='binary:logistic'):

        '''
        Calls the body of the xgboost code in the classification_models.py file. Used to create a xgboost algorithm.
        :param instruction: The objective that you want to model (str).
        :param text: A list of columns to perform text embedding on.
        :param dataset: The dataset being used in the xgboost algorithm (str).
        :param preprocess: Preprocess the data (bool).
        :param test_size: Size of the testing set (float).
        :param drop: A list of the dataset's columns to drop.
        :param random_seed: Initialize a pseudo-random number generator (int).
        :param learning_rate:  Boosting learning rate(float).
        :param n_estimators: Number of gradient boosted trees. Equivalent to number of boosting rounds(in   ).
        :param max_depth: Maximum tree depth for base learners(int).
        :param min_child_weight: Minimum sum of instance weight(hessian) needed in a child(int).
        :param gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree(int).
        :param subsample: Subsample ratio of the training instance(float).
        :param colsample_bytree: Subsample ratio of columns when constructing each tree(float).
        :param objective: Specify the learning task and the corresponding learning objective or a custom
        objective function to be used (string or callable).
        :param scale_pos_weight: Balancing of positive and negative weights(float).
        :param verbose: Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug).

        :return: a model and information to along with it stored in the self.models dictionary.
        '''

        self.models['xgboost'] = train_xgboost(instruction,
                                               dataset=self.dataset,
                                               text=[],
                                               random_state=random_state,
                                               preprocess=preprocess,
                                               drop=drop,
                                               learning_rate=learning_rate,
                                               n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               min_child_weight=min_child_weight,
                                               gamma=gamma,
                                               subsample=subsample,
                                               verbosity=verbosity,
                                               colsample_bytree=colsample_bytree,
                                               objective=objective)

        self.latest_model = 'xgboost'
        clearLog()

    # tunes a specific neural network based on the input model_to_tune

    def tune(self,
             model_to_tune=None,
             max_layers=10,
             min_layers=2,
             min_dense=32,
             max_dense=512,
             executions_per_trial=3,
             max_trials=1,
             generate_plots=True,
             activation='relu',
             loss='categorical_crossentropy',
             metrics='accuracy',
             patience=1,
             epochs=10,
             objective='val_accuracy',
             seed=42,
             directory='my_dir',
             verbose=0,
             test_size=0.2
             ):
        '''
        Calls the body of the tune identifier which is located in the supplementaries.py which then calls the appropriate tuner depending on the model
        :param model_to_tune: The model to tune.
        :param patience: Number of epochs with no improvement after which training will be stopped (int).
        :param dataset: The dataset being used in the tuner (str).
        :param models: The model dictionary (dict).
        :param generate_plots: Generate plots for the model (bool).
        :param max_layers: Maximum number of layers (int).
        :param min_layers: Minimum number of layers (int).
        :param min_dense: Minimum kernel density (int).
        :param max_dense: Maximum kernel density (int).
        :param executions_per_trial: Number of executions per trial (int).
        :param max_trials: Maximum number of trials
        :param activation: Activation Function (str).
        :param loss: Loss Function (str).
        :param metrics: Type of metrics function (str).
        :param epochs: Number of epochs (int).
        :param objective: Name of model metric to maximize/minimize (str).
        :param seed: Random seed (int).
        :param directory: Path to the directory (str).
        :param verbose: Printing the logging information (int).
        :param test_size: Size of the testing set (float).


        :return: an updated model and history stored in the models dictionary
        '''

        if model_to_tune is None:
            model_to_tune = self.latest_model

        self.models = tune_helper(
            model_to_tune=model_to_tune,
            patience=patience,
            dataset=self.dataset,
            models=self.models,
            generate_plots=generate_plots,
            max_layers=max_layers,
            min_layers=min_layers,
            min_dense=min_dense,
            max_dense=max_dense,
            executions_per_trial=executions_per_trial,
            max_trials=max_trials,
            activation=activation,
            loss=loss,
            metrics=metrics,
            epochs=epochs,
            objective=objective,
            seed=seed,
            directory=directory,
            verbose=verbose,
            test_size=test_size
        )
        clearLog()

    # query to build a convolutional neural network

    def convolutional_query(self,
                            instruction=None,
                            read_mode=None,
                            verbose=0,
                            preprocess=True,
                            data_path=None,
                            new_folders=True,
                            image_column=None,
                            test_size=0.2,
                            augmentation=True,
                            custom_arch=None,
                            pretrained=None,
                            epochs=10,
                            height=None,
                            width=None,
                            show_feature_map=False,
                            save_as_tfjs=None,
                            save_as_tflite=None,
                            generate_plots=None):
        '''
        Calls the body of the convolutional neural network query which is located in the feedforward.py file
        :param instruction: The objective that you want to model (str).
        :param read_mode: The type of dataset (str).
        :param verbose: Printing the logging information (int).
        :param preprocess: Preprocess the data (bool).
        :param data_path: Path to the dataset (str).
        :param new_folders: Create new folders for the image during preprocessing (bool).
        :param image_column: The column in the csv file where the filepaths for the images exist (str).
        :param test_size: Ratio of dataset allotted to the testing data (float).
        :param augmentation: Perform image data augmentation (bool).
        :param epochs: Number of epochs (int).
        :param height: Height of the input image (int).
        :param width: Width of the input image (int).
        :param show_feature_map: Displays feature map graphic (bool).


        :return: an updated model and history stored in the models dictionary
        '''

        # storing values in the model dictionary
        self.models["convolutional_NN"] = convolutional(
            instruction=instruction,
            read_mode=read_mode,
            verbose=verbose,
            preprocess=preprocess,
            data_path=self.dataset,
            new_folders=new_folders,
            image_column=image_column,
            training_ratio=1 - test_size,
            augmentation=augmentation,
            custom_arch=custom_arch,
            pretrained=pretrained,
            epochs=epochs,
            height=height,
            width=width,
            save_as_tfjs=save_as_tfjs,
            save_as_tflite=save_as_tflite,
            generate_plots=generate_plots)

        if show_feature_map:
            model = self.models["convolutional_NN"]["model"]
            X_test = self.models["convolutional_NN"]["data"]["test"]

            # Get first image in test images and format it
            img = X_test[0][0]
            img /= 255
            successive_outputs = [layer.output for layer in model.layers[1:]]
            visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)
            successive_feature_maps = visualization_model.predict(img)

            # Add main title to figure
            firstPlot = True

            # Include names of layers in plot
            layer_names = [layer.name for layer in model.layers]
            for layer_name, feature_map in zip(layer_names, successive_feature_maps):
                if len(feature_map.shape) == 4:

                    # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers
                    n_features = feature_map.shape[-1]  # number of features in the feature map
                    height = feature_map.shape[1]  # feature map shape (1, size, size, n_features)
                    width = feature_map.shape[2]
                    display_grid = np.zeros((height, width * n_features))

                    # Format features appropriately
                    for i in range(n_features):
                        img = feature_map[0, :, :, i]
                        img -= img.mean()
                        img /= img.std()
                        img *= 64
                        img += 128
                        img = np.clip(img, 0, 255).astype('uint8')

                        # Tile each filter into a horizontal grid
                        display_grid[:, i * width: (i + 1) * width] = img

                    # Display the grid
                    scale = 20. / n_features
                    plt.figure(figsize=(scale * n_features, scale))
                    if firstPlot:
                        plt.title(f'Network Visualization\n\n{layer_name}')
                        firstPlot = False
                    else:
                        plt.title(layer_name)
                    plt.grid(False)
                    plt.imshow(display_grid, aspect='auto', cmap='viridis')
                    plt.show()

        self.latest_model = 'convolutional_NN'
        clearLog()

    def gan_query(self,
                  instruction=None,
                  type='dcgan',
                  num_images=3,
                  preprocess=True,
                  data_path=None,
                  verbose=0,
                  epochs=10,
                  height=None,
                  width=None,
                  output_path=None):
        if type == 'dcgan':
            self.models["DCGAN"] = dcgan(instruction=instruction,
                                 num_images=num_images,
                                 preprocess=preprocess,
                                 data_path=self.dataset,
                                 verbose=verbose,
                                 epochs=epochs,
                                 height=height,
                                 width=width,
                                 output_path=output_path
                                 )
            self.latest_model = 'DCGAN'
            clearLog()

    # sentiment analysis prediction wrapper

    def classify_text(self, text):
        """
        Calls the body of the text_classification neural network query which is located in the nlp_queries.py file. This can only be called
        if text_classification_query has been called previously.
        :param text: The new text that you want to classify (str).
        :return: a classification of text that you've provided
        """
        clearLog()
        return classify_text(self=self, text=text)

    # sentiment analysis query
    def text_classification_query(self, instruction, label_column=None, drop=None,
                                  preprocess=True,
                                  test_size=0.2,
                                  random_state=49,
                                  learning_rate=1e-2,
                                  epochs=5,
                                  monitor="val_loss",
                                  batch_size=32,
                                  max_text_length=20,
                                  generate_plots=True,
                                  save_model=False,
                                  save_path=os.getcwd()):
        '''
        Calls the body of the text_classification query which is located in the nlp_queries.py file
        :param instruction: The objective that you want to model (str).
        :param drop: A list of the dataset's columns to drop.
        :param preprocess: Preprocess the data (bool).
        :param test_size: Size of the testing set (float).
        :param random_state: Initialize a pseudo-random number generator (int).
        :param learning_rate: The learning rate of the model (float).
        :param epochs: Number of epochs (int).
        :param batch_size: The batch size for the dataset (int).
        :param generate_plots: Generate plots for the model (bool).
        :param save_model: Save the model (bool).
        :param save_path: Filepath of where to save the model (str).


        :return: an updated model and history stored in the models dictionary
        '''

        # storing values the model dictionary
        self.models["text_classification"] = text_classification_query(
            self=self, instruction=instruction, label_column=label_column, drop=drop,
            preprocess=preprocess,
            test_size=test_size,
            random_state=random_state,
            learning_rate=learning_rate,
            monitor=monitor,
            epochs=epochs,
            batch_size=batch_size,
            max_text_length=max_text_length,
            generate_plots=generate_plots,
            save_model=save_model,
            save_path=save_path)
        self.latest_model = 'text_classification'
        clearLog()

    # summarization predict wrapper
    def get_summary(self, text, num_beams=4, no_repeat_ngram_size=2, num_return_sequences=1,
                    early_stopping=True):
        '''
        Calls the body of the summarizer which is located in the nlp_queries.py file
        :param text: set of text that you want to summarize (str).
        :param max_summary_length: Max generated summary length (int).
        :param early_stopping: Sets early stopping (bool).
        :param num_return_sequences: Sets the number of likely possibilities to output (int).
        :param no_repeat_ngram_size: Sets the number of unrepeated consecutive n-grams (int).
        :param num_beams: Sets number of possibilities to explore in beam search (int).
        :return: a summary of text inputted in the text field.
        '''
        clearLog()
        return get_summary(self=self, text=text, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
                           , num_return_sequences=num_return_sequences, early_stopping=early_stopping)

    # summarization query
    def summarization_query(self, instruction, label_column=None, preprocess=True,
                            drop=None,
                            epochs=5,
                            batch_size=32,
                            learning_rate=3e-5,
                            max_text_length=512,
                            test_size=0.2,
                            gpu=False,
                            random_state=49,
                            generate_plots=True,
                            save_model=False,
                            save_path=os.getcwd()):
        '''
        Calls the body of the summarization  query which is located in the nlp_queries.py file
        :param instruction: The objective that you want to model (str).
        :param preprocess: Preprocess the data (bool).
        :param drop: A list of the dataset's columns to drop.
        :param epochs: Number of epochs (int).
        :param batch_size: The batch size for the dataset (int).
        :param learning_rate: The learning rate of the model (float).
        :param max_text_length: The maximum length of the string of text (int).
        :param test_size: Size of the testing set (float).
        :param gpu: Use gpu for accelerated training (bool).
        :param random_state: Initialize a pseudo-random number generator (int).
        :param generate_plots: Generate plots for the model (bool).
        :param save_model: Save the model (bool).
        :param save_path: Filepath of where to save the model (str).


        :return: an updated model and history stored in the models dictionary
        '''

        self.models["summarization"] = summarization_query(
            self=self, instruction=instruction, preprocess=preprocess, label_column=label_column,
            drop=drop,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_text_length=max_text_length,
            test_size=test_size,
            gpu=gpu,
            random_state=random_state,
            generate_plots=generate_plots,
            save_model=save_model,
            save_path=save_path)

        self.latest_model = 'summarization'
        clearLog()

    # image_caption generator wrapper

    def generate_caption(self, image):
        '''
        Calls the body of the caption generator which is located in the nlp_queries.py file.
        :param image: the image that you want to generate a caption for.
        :return: a caption for the image inputted in the image field.
        '''
        caption = generate_caption(self=self, image=image)
        clearLog()
        return ' '.join(caption[:len(caption) - 1])

    # image_caption prediction query
    def image_caption_query(self, instruction, label_column=None,
                            drop=None,
                            epochs=10,
                            preprocess=True,
                            random_state=49,
                            test_size=0.2,
                            top_k=5000,
                            batch_size=32,
                            buffer_size=1000,
                            embedding_dim=256,
                            units=512,
                            gpu=False,
                            generate_plots=True,
                            save_model_decoder=False,
                            save_path_decoder=os.getcwd(),
                            save_model_encoder=False,
                            save_path_encoder=os.getcwd()):
        '''
        Calls the body of the image_caption query which is located in the nlp_queries.py file
        :param instruction: The objective that you want to model (str).
        :param drop: A list of the dataset's columns to drop.
        :param epochs: Number of epochs (int).
        :param preprocess: Preprocess the data (bool).
        :param random_state: Initialize a pseudo-random number generator (int).
        :param top_k:  Number of most frequent words in the vocab to be used in tokenization (int).
        :param batch_size: The batch size for the dataset (int).
        :param buffer_size: The maximum number of elements to buffer (int).
        :param embedding_dim: The dimension of the word embedding mapping (int).
        :param units: The recurrent units in the decoder (int).
        :param test_size: test size (int) .
        :param gpu: Choose to use gpu (bool).
        :param generate_plots: Generate plots for the model (bool).
        :param save_model_decoder: Save the decoder (bool).
        :param save_path_decoder: Filepath of where to save the decoder (str).
        :param save_model_encoder: Save the encoder (bool).
        :param save_path_encoder: Filepath of where to save the encoder (str).


        :return: an updated model and history stored in the models dictionary
        '''

        self.models["image_caption"] = image_caption_query(
            self, instruction=instruction, label_column=label_column,
            drop=drop,
            epochs=epochs,
            preprocess=preprocess,
            random_state=random_state,
            test_size=test_size,
            top_k=top_k,
            batch_size=batch_size,
            buffer_size=buffer_size,
            embedding_dim=embedding_dim,
            units=units,
            gpu=gpu,
            generate_plots=generate_plots,
            save_model_decoder=save_model_decoder,
            save_path_decoder=save_path_decoder,
            save_model_encoder=save_model_encoder,
            save_path_encoder=save_path_encoder)
        self.latest_model = 'image_caption'
        clearLog()

    def generate_text(self, file_data=True, prefix=None,
                      max_length=512,
                      top_k=50,
                      top_p=0.9,
                      temperature=0.3,
                      return_sequences=2):
        """
        :param file_data: using the data passed to the client instance or not (bool).
        :param prefix: a string that you want the generated text to begin with
        :param max_length: the length of desired text you want (int)
        :param top_k: number of most frequent words in the vocab to be used in tokenization (int).
        :param top_p: p value between 0 and 1 (float)
        :param temperature: temperature to make the next word probability distribution sharper (float).
        :param return_sequences: how many different text sequences you want returned
        :return: generated text
        """
        self.models['text_generation'] = generate_text(self=self,
                                                       file_data=file_data,
                                                       prefix=prefix,
                                                       max_length=max_length,
                                                       top_k=top_k,
                                                       top_p=top_p,
                                                       temperature=temperature,
                                                       return_sequences=return_sequences)

        self.latest_model = 'text_generation'
        clearLog()

    # name entity recognition query
    def named_entity_query(self, instruction):
        """
        function to identify name entities
        :param instruction: Used to get target column
        :return: dictionary object with detected name-entities
        """
        self.models["named_entity_recognition"] = get_ner(self, instruction=instruction)
        self.latest_model = "named_entity_recognition"
        clearLog()

    # shows the names of plots associated with a specific model
    def plot_names(self, model=None):
        '''
        Function to get names of plots given the name of the model you want
        :param model: the model that you want to get the plots for
        '''
        if model is None:
            model = self.latest_model
        print(self.models[model]['plots'].keys())

        # shows names of models in model dictionary

        clearLog()

    def model(self, model=None):
        '''
        Function that either returns the latest model or one specified and its information as a dictionary.
        :param model: is the model key that you want to use.
        '''
        if model is None:
            model = self.latest_model
        clearLog()
        return self.models[model]

    # shows the keys in the models dictionary

    def info(self, model=None):
        '''
        Function that retrieves the model_data; all the information in self.models for that model
        :param model: default to the latest model, but essentially the model key
        '''
        if model is None:
            model = self.latest_model
        return get_model_data(self, model)
        clearLog()

    # returns all operators applicable to the client's models dictionary
    def operators(self, model=None):
        '''
        Function that retrieves all of the operators; pipelines that were used to model the dataset
        :param model: default to the latest model, but essentially the model key
        '''
        if model is None:
            model = self.latest_model
        get_operators(self, model)
        clearLog()

    # show accuracy scores for client's model
    def accuracy(self, model=None):
        '''
        Function that retrieves all of the accuracies in the self.models dictionary for the key.
        :param model: default to the latest model, but essentially the model key
        '''
        if model is None:
            model = self.latest_model
        clearLog()
        return get_accuracy(self, model)

    # show losses for client's model
    def losses(self, model=None):
        '''
        Function that retrieves all of the losses in the self.models dictionary for the key.
        :param model: default to the latest model, but essentially the model key
        '''
        if model is None:
            model = self.latest_model
        clearLog()
        return get_losses(self, model)

    # return client model's target
    def target(self, model=None):
        '''
        Function that retrieves all of the targets in the self.models dictionary for the key.
        :param model: default to the latest model, but essentially the model key
        returns target variable of model used in client instance
        '''
        if model is None:
            model = self.latest_model
        clearLog()
        return get_target(self, model)

    # return NLP model's vocabulary
    def vocab(self, model=None):
        '''
        Function that retrieves the NLP models vocabulary.
        :param model: default to the latest model, but essentially the model key
        '''
        if model is None:
            model = self.latest_model
        clearLog()
        return get_vocab(self, model)

    # plotting for client
    def plots(self, model=None, plot=None, save=False):
        '''
        Function that retrieves all of plots in the self.models dictionary for the key.
        :param model: default to the latest model, but essentially the model key
        :param plot: plot specified during the client session to be procured
        :param save: option to save plots after client session is done (default is false, or
        '''
        if model is None:
            model = self.latest_model
        clearLog()
        get_plots(self, model, plot, save)

    # shows analysis of the model
    def analyze(self, model=None, save=True, save_model=False):
        '''
        Function that retrieves all of plots in the self.models dictionary for the key.
        :param model: default to the latest model, but essentailly the model key
        '''
        if model is None:
            model = self.latest_model
        clearLog()
        analyze(self, model, save, save_model)

    def dashboard(self):
        dash = edaDashboard(self.dataset)
        dash.dashboard()
