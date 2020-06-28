from libra.query.nlp_queries import (image_caption_query,
                                     generate_caption, predict_text_sentiment,
                                     text_classification_query, get_summary,
                                     summarization_query)
from libra.query.classification_models import (k_means_clustering,
                                               train_svm, nearest_neighbors,
                                               decision_tree)
from libra.query.supplementaries import tune_helper, stats
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
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, plot_confusion_matrix, recall_score, precision_score, f1_score
from sklearn.preprocessing import label_binarize
from numpy import interp
from itertools import cycle


#supressing warnings for cleaner dialogue box
warnings.simplefilter(action='error', category=FutureWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# function imports from other files
currLog = ""
counter = 0



#clears log when needed - currently not being used
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
        return self.interpret(modelKey, predictions)

    def interpret(self, modelKey, predictions):
        modeldict = self.models[modelKey]
        if modeldict.get('interpreter'):
            if type(modeldict['interpreter']) == dict:
                inverted_interpreter = dict(map(reversed, modeldict['interpreter'].items()))
                toRet = []
                for each in predictions:
                    toRet.append(inverted_interpreter[each])
                predictions = toRet
            else:
                predictions = modeldict['interpreter'].inverse_transform(
                     predictions)
        return predictions


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
            if data[remove].dtype.name == 'object':
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

    def kmeans_clustering_query(self,
                                preprocess=True,
                                generate_plots=True,
                                drop=None,
                                base_clusters=2):

        self.models['k_means_clustering'] = k_means_clustering(
            dataset=self.dataset,
            preprocess=preprocess,
            generate_plots=generate_plots,
            drop=drop,
            base_clusters=base_clusters)

        self.latest_model = 'k_means_clustering'

    def svm_query(self,
                  instruction,
                  test_size=0.2,
                  text=[],
                  kernel='linear',
                  preprocess=True,
                  drop=None,
                  cross_val_size=0.3):

        self.models['svm'] = train_svm(instruction,
                                       dataset=self.dataset,
                                       text=text,
                                       test_size=test_size,
                                       kernel=kernel,
                                       preprocess=preprocess,
                                       drop=drop,
                                       cross_val_size=cross_val_size)

        self.latest_model = 'svm'

    def nearest_neighbor_query(
            self,
            instruction=None,
            text=[],
            preprocess=True,
            drop=None,
            min_neighbors=3,
            max_neighbors=10):
        self.models['nearest_neighbor'] = nearest_neighbors(
            instruction=instruction,
            text=text,
            dataset=self.dataset,
            preprocess=preprocess,
            drop=drop,
            min_neighbors=min_neighbors,
            max_neighbors=max_neighbors)

        self.latest_model = 'nearest_neighbor'

    def decision_tree_query(
            self,
            instruction,
            preprocess=True,
            test_size=0.2,
            drop=None):

        self.models['decision_tree'] = decision_tree(instruction,
                                                     text=[],
                                                     dataset=self.dataset,
                                                     preprocess=True,
                                                     test_size=0.2,
                                                     drop=None)

        self.latest_model = 'decision_tree'


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
            model_to_tune =  self.latest_model

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

    def stat_analysis(self, column_name="none", drop=None):
        stats(
            dataset=self.dataset,
            drop=drop,
            column_name=column_name
        )

        return

    def convolutional_query(self,
                            instruction=None,
                            read_mode=None,
                            new_folders=True,
                            image_column=None,
                            training_ratio=0.8):

        # storing values the model dictionary
        self.models["convolutional_NN"] = convolutional(
            instruction=instruction,
            read_mode=read_mode,
            data_path=self.dataset,
            new_folders=new_folders,
            image_column=image_column,
            training_ratio=training_ratio)

        self.latest_model = 'convolutional_NN'


    # Sentiment analysis predict wrapper
    def predict_text_sentiment(self, text):
        return predict_text_sentiment(self=self, text=text)

    # sentiment analysis query
    def text_classification_query(self, instruction):

        # storing values the model dictionary
        self.models["Text Classification LSTM"] = text_classification_query(
            self=self, instruction=instruction)
        self.latest_model = 'Text Classification LSTM'

    # Document summarization predict wrapper
    def get_summary(self, text):
        return get_summary(self=self, text=text)

    # text summarization query
    def summarization_query(self, instruction,
                            preprocess=True,
                            test_size=0.2,
                            random_state=49,
                            epochs=1,
                            generate_plots=True):

        self.models["Document Summarization"] = summarization_query(
            self=self, instruction=instruction)
        self.latest_model = 'Document Summarization'

    # Image caption prediction
    def generate_caption(self, image):
        caption = generate_caption(self=self, image=image)
        return ' '.join(caption[:len(caption) - 1])

    # Image Caption query
    def image_caption_query(self, instruction, epochs, random_state,
                            preprocess=True,
                            generate_plots=True):
        self.models["Image Caption"] = image_caption_query(
            self=self,
            epochs=epochs,
            instruction=instruction,
            random_state=random_state,
            preprocess=preprocess,
            generate_plots=generate_plots)
        self.latest_model = 'Image Caption'

    def dimensionality_reducer(self, instruction):
        dimensionality_reduc(instruction, self.dataset)

    def show_plots(self, model=None):
        if model == None:
            model = self.latest_model
        print(self.models[model]['plots'].keys())

    # shows the keys in the models dictionary
    def model_data(self, model=None):
        if model == None:
            model = self.latest_model

        if model in self.models:
            data = [key for key in self.models[model].keys()]
            print(data)
        else:
            raise Exception(
                "The requested model has not been applied to the client.")

    # returns all operators applicable to the client's models dictionary
    def operators(self, model=None):
        if model == None:
            model = self.latest_model
        defined = ['plots', 'accuracy', 'losses']
        operations = [
            func +
            "()" for func in self.models[model].keys() if func in defined]
        if len(operations) > 0:
            print(operations)
        else:
            raise Exception(
                "There are no built-in operators defined for this model." 
                " Please refer to the models dictionary.")

    # show accuracy scores for client's model

    def accuracy(self, model=None):
        if model == None:
            model = self.latest_model

        if 'accuracy' in self.models[model].keys():
            return self.models[model]['accuracy']
        elif 'cross_val_score' in self.models[model].keys():
            return {'cross_val_score': self.models[model]['cross_val_score']}
        else:
            raise Exception("Accuracy is not defined for {}".format(model))

    # show losses for client's model
    def losses(self, model=None):
        if model == None:
            model = self.latest_model

        if 'losses' in self.models[model].keys():
            return self.models[model]['losses']
        else:
            raise Exception("Losses are not defined for {}".format(model))

    # Analysis of model
    def analyze(self, model=None):
        if model == None:
            model = self.latest_model

        modeldict = self.models[model]
        if modeldict.get('plots'):
            for key in modeldict['plots']:
                modeldict['plots'][key]
                plt.show()

        if modeldict.get('test_data'):
            data = modeldict['test_data']['X']
            real = modeldict['test_data']['y']
            preds = modeldict['model'].predict(data)

        if model == 'regression_ANN':
            MSE = sklearn.metrics.mean_squared_error(real, preds)
            MAE = sklearn.metrics.mean_absolute_error(real, preds)
            print("Mean Squared Error on Test Set: " + str(MSE))
            print("Mean Absolute Error on Test Set: " + str(MAE))

        elif model in ['svm', 'nearest_neighbor', 'decision_tree', 'classification_ANN']: # classification models
            plot_mc_roc(real, preds, modeldict['interpreter'])

            if model in ['svm', 'nearest_neighbor', 'decision_tree']: #sklearn models ONLY
                labels = list(modeldict['interpreter'].keys())
                plot_confusion_matrix(modeldict['model'], data, real, display_labels=labels)
                plt.show()

                accuracy = modeldict['accuracy_score']
            else: #classification_ANN
                # TODO: find a prettier way to plot this
                confusion_matrix(real, preds)
                accuracy = modeldict['accuracy']['validation_accuracy']

            recall = recall_score(real, preds, average='micro')
            precision = precision_score(real, preds, average='micro')
            f1 = f1_score(real, preds, average='micro')

            print("Accuracy on Test Set: " + str(accuracy))
            print("Recall on Test Set: " + str(recall))
            print("Precision on Test Set: " + str(precision))
            print("F1 Score on Test Set: " + str(f1))
        elif model not in ['k_means_clustering', 'regression_ANN']:
            print("further analysis is not supported for {}".format(model))


def plot_mc_roc (y_test, y_score, interpreter = None):
    lw = 2
    n_classes = len(np.unique(y_test))
    classes = pd.unique(y_test)
    y_test= label_binarize(y_test, classes = classes)
    y_score = label_binarize(y_score, classes = classes)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    for i in range(n_classes):
        if type(interpreter) == dict:
            inverted_interpreter = dict(map(reversed, interpreter.items()))
            plt.plot(fpr[i], tpr[i], lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(inverted_interpreter[i], roc_auc[i]))
        else:
            plt.plot(fpr[i], tpr[i], lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(interpreter.inverse_transform([i])[0], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.show()

# Easier to comment the one you don't want to run instead of typing them
# out every time
# newClient = client('/Users/palashshah/Desktop')
# newClient.convolutional_query()
# newClient.tune('convolutional_NN', epochs=1)
# newClient.neural_network_query("Model median house value")
# newClient = client('tools/data/structured_data/landslides_after_rainfall.csv').neural_network_query(instruction='Model distance',
# drop=['id', 'geolocation', 'source_link', 'source_name'])
#newClient = client('tools/data/structured_data/fake_job_postings.csv').neural_network_query(instruction='Classify fraudulent',
#                                                                                            drop=['job_id'],
#                                                                                            text=['department','description', 'company_profile','requirements', 'benefits'])
newClient = client('/Users/ramyabhaskara/PycharmProjects/libra/tools/data/structured_data/housing.csv')
newClient.decision_tree_query("Model ocean proximity")
newClient.analyze()