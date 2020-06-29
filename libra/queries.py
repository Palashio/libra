from libra.query.nlp_queries import (image_caption_query,
                                     generate_caption, classify_text,
                                     text_classification_query, get_summary,
                                     summarization_query)
from libra.query.classification_models import (k_means_clustering,
                                               train_svm, nearest_neighbors,
                                               decision_tree)
from libra.query.supplementaries import tune_helper, stats, get_model_data, get_operators, get_accuracy, get_losses, get_target, get_plots
from libra.query.feedforward_nn import (regression_ann,
                                        classification_ann,
                                        convolutional)
from libra.query.dimensionality_red_queries import dimensionality_reduc
from libra.data_generation.grammartree import get_value_instruction
from libra.data_generation.dataset_labelmatcher import (get_similar_column,
                                                        get_similar_model)
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import warnings
import os

# supressing warnings for cleaner dialogue box
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


# logging function that creates hierarchial display of the processes of
# different functions. Copied into different python files to maintain
# global variables.
def logger(instruction, found="", slash=''):
    global currLog
    global counter
    if counter == 0:
        currLog += (" " * 2 * counter) + str(instruction) + str(found)
    elif instruction == "->":
        counter = counter - 1
        currLog += (" " * 2 * counter) + str(instruction) + str(found)
    else:
        currLog += (" " * 2 * counter) + "|- " + str(instruction) + str(found)
        if instruction == "done...":
            currLog += "\n" + "\n"

    counter += 1
    print(currLog)
    currLog = ""


# class to store all query information. Currently, old_models is not being used.
class client:
    def __init__(self, data):
        logger("creating object...")
        self.dataset = data
        logger("Loading dataset...")
        self.models = {}
        self.old_models = {}
        self.latest_model = None
        logger("done...")
        clearLog()

    # returns models with a specific string - currently deprecated, should not be used. 
    def get_models(self, model_requested):
        logger("Getting model...")
        return get_similar_model(model_requested, self.models.keys())
        clearLog()

    # param modelKey: string representation of the model to make prediction
    # param data: dataframe version of desired prediction set
    def predict(self, data, modelKey=None):
        if modelKey == None:
            modelKey = self.latest_model
        modeldict = self.models[modelKey]
        data = modeldict['preprocesser'].transform(data)
        predictions = modeldict['model'].predict(data)
        if modeldict.get('interpreter'):
            predictions = modeldict['interpreter'].inverse_transform(
                predictions)
        return predictions


# query to create a neural network model for the client 
    # will build either a regression ANN or a classification ANN
    def neural_network_query(self,
                             instruction,
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

        data = pd.read_csv(self.dataset)

        if preprocess:

            remove = get_similar_column(
                get_value_instruction(instruction), data)

            if len(data) < 50:
                raise Exception("Only datasets larger then 50 rows are supported for neural networks")
            if len(data[remove].value_counts()) <= 50:
                callback_mode = 'max'
                maximizer = "val_accuracy"
                self.classification_query_ann(
                    instruction,
                    text=text,
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

    # single regression query using a feed-forward neural network
    # instruction should be the value of a column
    def regression_query_ann(
            self,
            instruction,
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

        self.models['regression_ANN'] = regression_ann(
            instruction=instruction,
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

    # query for multilabel classification query, does not work for
    # binaryclassification, fits to feed-forward neural network
    def classification_query_ann(
            self,
            instruction,
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

        self.models['classification_ANN'] = classification_ann(
            instruction=instruction,
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

    # query to perform k-means clustering
    def kmeans_clustering_query(self,
                                preprocess=True,
                                generate_plots=True,
                                drop=None,
                                base_clusters=2,
                                verbose=0,
                                n_init=10,
                                max_iter=300,
                                random_state=42,
                                text=[]
                                ):

        self.models['k_means_clustering'] = k_means_clustering(
            dataset=self.dataset,
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

    # query to create a support vector machine
    def svm_query(self,
                  instruction,
                  test_size=0.2,
                  text=[],
                  kernel='linear',
                  preprocess=True,
                  drop=None,
                  cross_val_size=0.3,
                  degree=3,
                  gamma='scale',
                  coef0=0.0,
                  max_iter=-1
                  ):

        self.models['svm'] = train_svm(instruction,
                                       dataset=self.dataset,
                                       text=text,
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
    
    # query to create a nearest neighbors model
    def nearest_neighbor_query(
            self,
            text=[],
            instruction=None,
            preprocess=True,
            drop=None,
            min_neighbors=3,
            max_neighbors=10,
            leaf_size=30,
            p=2,
            algorithm='auto'
    ):
        self.models['nearest_neighbor'] = nearest_neighbors(
            instruction=instruction,
            text=text,
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

        self.models['decision_tree'] = decision_tree(instruction=instruction,
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
                                                     min_weight_fraction_leaf =min_weight_fraction_leaf,
                                                     max_leaf_nodes=max_leaf_nodes,
                                                     min_impurity_decrease =min_impurity_decrease,
                                                     ccp_alpha=ccp_alpha)



        self.latest_model = 'decision_tree'

    # tunes a specific neural network based on the input model_to_tune
    def tune(self,
             model_to_tune=None,
             max_layers=10,
             min_layers=2,
             min_dense=32,
             max_dense=512,
             executions_per_trial=3,
             max_trials=1,
             activation='relu',
             loss='categorical_crossentropy',
             metrics='accuracy',
             epochs=10,
             objective='val_accuracy',
             seed=42,
             directory='my_dir',
             verbose=0,
             test_size=0.2
             ):

        if model_to_tune == None:
            model_to_tune = self.latest_model

        self.models = tune_helper(
            model_to_tune=model_to_tune,
            dataset=self.dataset,
            models=self.models,
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

    # returns metrics about your dataset including similarity information
    def stat_analysis(self, column_name="none", drop=None):
        stats(
            dataset=self.dataset,
            drop=drop,
            column_name=column_name
        )

        return

    # query to build a convolutional neural network
    def convolutional_query(self,
                            instruction=None,
                            read_mode=None,
                            preprocess=True,
                            new_folders=True,
                            image_column=None,
                            training_ratio=0.8,
                            augmentation=True,
                            epochs=10,
                            height=None,
                            width=None):

        # storing values the model dictionary
        self.models["convolutional_NN"] = convolutional(
            instruction=instruction,
            read_mode=read_mode,
            preprocess=preprocess,
            data_path=self.dataset,
            new_folders=new_folders,
            image_column=image_column,
            training_ratio=training_ratio,
            augmentation=augmentation,
            epochs=epochs,
            height=height,
            width=width)

        self.latest_model = 'convolutional_NN'

    # Sentiment analysis predict wrapper
    def classify_text(self, text):
        return classify_text(self=self, text=text)

    # sentiment analysis query
    def text_classification_query(self, instruction, drop=None,
                                  preprocess=True,
                                  test_size=0.2,
                                  validation_size=0.1,
                                  random_state=49,
                                  learning_rate=1e-2,
                                  epochs=20,
                                  maximizer="val_loss",
                                  batch_size=32,
                                  maxTextLength=200,
                                  generate_plots=True,
                                  save_model=False,
                                  save_path=os.getcwd()):

        # storing values the model dictionary
        self.models["Text Classification"] = text_classification_query(
            self=self, instruction=instruction, drop=drop,
            preprocess=preprocess,
            test_size=test_size,
            val_size=validation_size,
            random_state=random_state,
            learning_rate=learning_rate,
            maximizer=maximizer,
            epochs=epochs,
            batch_size=batch_size,
            maxTextLength=maxTextLength,
            generate_plots=generate_plots,
            save_model=save_model,
            save_path=save_path)
        self.latest_model = 'Text Classification'

    # Document summarization predict wrapper
    def get_summary(self, text):
        return get_summary(self=self, text=text)

    # text summarization query
    def summarization_query(self, instruction, preprocess=True,
                            drop=None,
                            epochs=10,
                            batch_size=64,
                            learning_rate=1e-4,
                            max_text_length=512,
                            max_summary_length=150,
                            test_size=0.2,
                            random_state=49,
                            generate_plots=True,
                            save_model=False,
                            save_path=os.getcwd()):

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
            generate_plots=generate_plots,
            save_model=save_model,
            save_path=save_path)

        self.latest_model = 'Document Summarization'

    # Image caption prediction
    def generate_caption(self, image):
        caption = generate_caption(self=self, image=image)
        return ' '.join(caption[:len(caption) - 1])

    # Image Caption query
    def image_caption_query(self, instruction,
                            drop=None,
                            epochs=10,
                            preprocess=True,
                            random_state=49,
                            top_k=5000,
                            batch_size=1,
                            buffer_size=1000,
                            embedding_dim=256,
                            units=512,
                            generate_plots=True,
                            save_model_decoder=False,
                            save_path_decoder=os.getcwd(),
                            save_model_encoder=False,
                            save_path_encoder=os.getcwd()):
        self.models["Image Caption"] = image_caption_query(
            self, instruction=instruction,
            drop=drop,
            epochs=epochs,
            preprocess=preprocess,
            random_state=random_state,
            top_k=top_k,
            batch_size=batch_size,
            buffer_size=buffer_size,
            embedding_dim=embedding_dim,
            units=units,
            generate_plots=generate_plots,
            save_model_decoder=save_model_decoder,
            save_path_decoder=save_path_decoder,
            save_model_encoder=save_model_encoder,
            save_path_encoder=save_path_encoder)
        self.latest_model = 'Image Caption'

    # performs dimensionality reduction on your dataset 
    # based on user instruction for target variable 
    def dimensionality_reducer(self, instruction):
        dimensionality_reduc(instruction, self.dataset)

    # shows the names of plots associated with a specific model
    def plot_names(self, model=None):
        if model == None:
            model = self.latest_model
        print(self.models[model]['plots'].keys())

   # shows names of models associated with the client
    def model_names(self):
        models_avail = [key for key in self.models.keys()]
        print(models_avail)

    # shows the keys in the models dictionary
    def model_data(self, model=None):
        if model == None:
            model = self.latest_model
        get_model_data(self,model)

    # returns all operators applicable to the client's models dictionary
    def operators(self, model=None):
        if model == None:
            model = self.latest_model
        get_operators(self, model)

    # show accuracy scores for client's model
    def accuracy(self, model=None):
        if model == None:
            model = self.latest_model
        return get_accuracy(self, model)

    # show losses for client's model
    def losses(self, model=None): 
        if model == None:
            model = self.latest_model
        return get_losses(self, model)
    
    # return client model's target
    def target(self, model=None):
        if model == None:
            model = self.latest_model
        return get_target(self,model)
    
    # return NLP model's vocabulary
    def vocab(self, model=None):
        if model == None:
            model = self.latest_model
        return get_vocab(self,model)
    
    # plotting for client
    def plots(self, model = "", plot = "", save = False):
        get_plots(self, model, plot, save)
    

# Easier to comment the one you don't want to run instead of typing them
# out every time

# newClient = client('/Users/palashshah/Desktop') newClient.convolutional_query() newClient.tune('convolutional_NN',
# epochs=1) newClient.neural_network_query("Model median house value") newClient = client(
# 'tools/data/structured_data/landslides_after_rainfall.csv').neural_network_query(instruction='Model distance',
# drop=['id', 'geolocation', 'source_link', 'source_name'])
# newClient = client('tools/data/structured_data/fake_job_postings.csv')
# newClient.neural_network_query(instruction='predict fraudulent',
#                                drop=['job_id'],
#                                text=['department', 'description', 'company_profile', 'requirements', 'benefits'])

# newClient = client('tools/data/structured_data/fake_job_postings.csv').neural_network_query(instruction='Classify
# fraudulent', drop=['job_id'], text=['department','description', 'company_profile','requirements', 'benefits'])

#newClient = client('tools/data/structured_data/fake_job_postings.csv').neural_network_query(instruction='Classify fraudulent',
#                                                                                            drop=['job_id'],
#                                                                                            text=['department','description', 'company_profile','requirements', 'benefits'])
newClient = client('../../tools/data/structured_data/housing.csv')
newClient.neural_network_query("Model median house value", epochs=3)
newClient.plots()

