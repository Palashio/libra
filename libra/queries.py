from libra.query.nlp_queries import (image_caption_query,
                                     generate_caption, classify_text,
                                     text_classification_query, get_summary,
                                     summarization_query)
from libra.query.classification_models import (k_means_clustering,
                                               train_svm, nearest_neighbors,
                                               decision_tree)

from libra.query.supplementaries import tune_helper, get_model_data, get_operators, get_accuracy, get_losses, get_target, get_plots, get_vocab

from libra.query.feedforward_nn import (regression_ann,
                                        classification_ann,
                                        convolutional)
from libra.data_generation.grammartree import get_value_instruction
from libra.data_generation.dataset_labelmatcher import (get_similar_column,
                                                        get_similar_model)
from libra.plotting.generate_plots import analyze
from colorama import Fore, Style
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import warnings
import os
import nltk
import ssl
import numpy as np

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
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')



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

    # param modelKey: string representation of the model to make prediction
    # param data: dataframe version of desired prediction set
    def predict(self, data, modelKey=None):
        '''
        Uses a model from the self.models dictionary to make a prediction. Also fits it based on the operator stored in the models dictionary.
        :param data: is the data that you want to predict for using model
        :param modelKey: is the specific model you want to use to predict
        :return: a prediction, most likely an array
        '''
        if modelKey is None:
            modelKey = self.latest_model
        if modelKey == 'Text Classification':
            map_func = np.vectorize(lambda x: self.classify_text(x))
            predictions = map_func(data)
            return predictions
        else:
            modeldict = self.models[modelKey]
            if modeldict.get('preprocesser'):
                data = modeldict['preprocesser'].transform(data)
            predictions = modeldict['model'].predict(data)
        clearLog()
        return self.interpret(modelKey, predictions)

    def interpret(self, modelKey, predictions):
        '''
        Function to interpret predictions from a neural network for the creation for graphs / user understanding.
        :param modelKey: is the model in the self.models dictionary that you want to use to interpret
        :param predictions: the predictions that come out of the model
        :return: a prediction, most likely an array
        '''
        modeldict = self.models[modelKey]
        if modeldict.get('interpreter'):
            if isinstance(modeldict['interpreter'], dict):
                inverted_interpreter = dict(
                    map(reversed, modeldict['interpreter'].items()))
                toRet = []
                for each in predictions:
                    toRet.append(inverted_interpreter[each])
                predictions = toRet
            else:
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
                             save_path=os.getcwd()):
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
                    save_path=save_path)
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
                    save_path=save_path)
        clearLog()
        return self

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
            save_path=os.getcwd()):
        '''
        Calls the body of the regression_query__ code in the supplementaries.py file. Used for a regression feed forward neural network.
        :param many params: all used as hyperparameters for the algorithm.
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
            save_path=save_path)

        self.latest_model = 'regression_ANN'
        clearLog()
        return self

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
            save_path=os.getcwd()):
        '''
        Calls the body of the classification code in the supplementaries.py file. Used for a classification feed forward neural network.
        :param many params: all used as hyperparameters for the algorithm.
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
            save_path=save_path)

        self.latest_model = 'classification_ANN'
        clearLog()
        return self
        
    # query to perform k-means clustering

    def kmeans_clustering_query(self,
                                preprocess=True,
                                scatters=[],
                                generate_plots=True,
                                drop=None,
                                base_clusters=2,
                                verbose=0,
                                n_init=10,
                                max_iter=300,
                                random_state=42,
                                text=[]
                                ):
        '''
        Calls the body of the kmeans_clustering code in the supplementaries.py file. Can be used without any preprocessing and/or parameters.
        :param many params: all used as hyperparameters for the algorithm.
        :return: a model and information to along with it stored in the self.models dictionary.
        '''

        self.models['k_means_clustering'] = k_means_clustering(
            dataset=self.dataset,
            scatters=scatters,
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
        return self
        
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
        :param many params: all used as hyperparameters for the algorithm.
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
        return self
        
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
        :param many params: all used as hyperparameters for the algorithm.
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
        return self
      
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
        :param many params: all used as hyperparameters for the algorithm.
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
        return self

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
        :param many params: all used as hyperparameters for the algorithm.
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
        return self

    # query to build a convolutional neural network

    def convolutional_query(self,
                            instruction=None,
                            read_mode=None,
                            verbose=0,
                            preprocess=True,
                            new_folders=True,
                            image_column=None,
                            test_size=0.2,
                            augmentation=True,
                            epochs=10,
                            height=None,
                            width=None):
        '''
        Calls the body of the convolutional neural network query which is located in the feedforward.py file
        :param many params: all used as hyperparameters for the algorithm.
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
            training_ratio=1 - testing_ratio,
            augmentation=augmentation,
            epochs=epochs,
            height=height,
            width=width)

        self.latest_model = 'convolutional_NN'
        clearLog()
        return self

    # sentiment analysis prediction wrapper

    def classify_text(self, text):
        '''
        Calls the body of the text classification neural network query which is located in the nlp_queries.py file. This can only be called
        if text_classification_query has been called previously.
        :param many params: all used as hyperparameters for the algorithm.
        :return: a classification of text that you've provided
        '''
        clearLog()
        return classify_text(self=self, text=text)

    # sentiment analysis query
    def text_classification_query(self, instruction, drop=None,
                                  preprocess=True,
                                  test_size=0.2,
                                  random_state=49,
                                  learning_rate=1e-2,
                                  epochs=20,
                                  monitor="val_loss",
                                  batch_size=32,
                                  max_text_length=200,
                                  max_features=20000,
                                  generate_plots=True,
                                  save_model=False,
                                  save_path=os.getcwd()):
        '''
        Calls the body of the text classification query which is located in the nlp_queries.py file
        :param many params: all used as hyperparameters for the algorithm.
        :return: an updated model and history stored in the models dictionary
        '''

        # storing values the model dictionary
        self.models["Text Classification"] = text_classification_query(
            self=self, instruction=instruction, drop=drop,
            preprocess=preprocess,
            test_size=test_size,
            random_state=random_state,
            learning_rate=learning_rate,
            monitor=monitor,
            epochs=epochs,
            batch_size=batch_size,
            max_text_length=max_text_length,
            max_features=max_features,
            generate_plots=generate_plots,
            save_model=save_model,
            save_path=save_path)
        self.latest_model = 'Text Classification'
        clearLog()
        return self

    # document summarization predict wrapper

    def get_summary(self, text):
        '''
        Calls the body of the summarizer which is located in the nlp_queries.py file
        :param text: set of text that you want to summarize.
        :return: a summary of text inputted in the text field.
        '''
        clearLog()
        return get_summary(self=self, text=text)

    # text summarization query
    def summarization_query(self, instruction, preprocess=True,
                            drop=None,
                            epochs=10,
                            batch_size=32,
                            learning_rate=1e-4,
                            max_text_length=512,
                            max_summary_length=150,
                            test_size=0.2,
                            random_state=49,
                            gpu=False,
                            generate_plots=True,
                            save_model=False,
                            save_path=os.getcwd()):
        '''
        Calls the body of the summarization  query which is located in the nlp_queries.py file
        :param many params: all used as hyperparameters for the algorithm.
        :return: an updated model and history stored in the models dictionary
        '''

        self.models["Document Summarization"] = summarization_query(
            self=self, instruction=instruction, preprocess=preprocess,
            drop=drop,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_text_length=max_text_length,
            max_summary_length=max_summary_length,
            test_size=test_size,
            random_state=random_state,
            gpu=gpu,
            generate_plots=generate_plots,
            save_model=save_model,
            save_path=save_path)

        self.latest_model = 'Document Summarization'
        clearLog()
        return self

    # image caption generator wrapper

    def generate_caption(self, image):
        '''
        Calls the body of the caption generator which is located in the nlp_queries.py file.
        :param image: the image that you want to generate a caption for.
        :return: a caption for the image inputted in the image field.
        '''
        caption = generate_caption(self=self, image=image)
        clearLog()
        return ' '.join(caption[:len(caption) - 1])

    # image caption prediction query
    def image_caption_query(self, instruction,
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
        Calls the body of the image caption query which is located in the nlp_queries.py file
        :param many params: all used as hyperparameters for the algorithm.
        :return: an updated model and history stored in the models dictionary
        '''

        self.models["Image Caption"] = image_caption_query(
            self, instruction=instruction,
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
        self.latest_model = 'Image Caption'
        clearLog()
        return self

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
    def model_names(self):
        '''
        Function that shows names of models associated with the client
        '''
        models_avail = [key for key in self.models.keys()]
        print(models_avail)
        clearLog()
    # shows the keys in the models dictionary
    
    def model_data(self, model=None):
        '''
        Function that retrieves the model_data; all the information in self.models for that model
        :param model: default to the latest model, but essentially the model key
        '''
        if model is None:
            model = self.latest_model
        get_model_data(self, model)
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
    def plots(self, model="", plot="", save=False):
        '''
        Function that retrieves all of plots in the self.models dictionary for the key.
        :param model: default to the latest model, but essentially the model key
        :param plot: plot specified during the client session to be procured
        :param save: option to save plots after client session is done (default is false, or

        '''
        clearLog()
        get_plots(self, model, plot, save)

    # shows analysis of the model
    def analyze(self, model=None, save=True):
        '''
        Function that retrieves all of plots in the self.models dictionary for the key.
        :param model: default to the latest model, but essentailly the model key
        '''
        if model is None:
            model = self.latest_model
        clearLog()
        analyze(self, model, save)
