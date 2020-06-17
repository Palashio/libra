from tuner import tuneReg, tuneClass, tuneCNN
import numpy as np
import os
from data_reader import DataReader
from tabulate import tabulate
from data_preprocesser import structured_preprocesser, initial_preprocesser
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cosine
import uuid

currLog = ""
counter = 0
number = 0
# # current_dir=os.getcw()

# # allows for all columns to be displayed when printing()
# pd.options.display.width = None


# # clears the log when new process is started up


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
        currLog += (" " * 2 * counter) + "|"
        currLog += "\n"
        currLog += (" " * 2 * counter) + "|- " + str(instruction) + str(found)
        if instruction == "done...":
            currLog += "\n"
            currLog += "\n"

    counter += 1
    if instruction == "->":
        print(currLog, end="")
    else:
        print(currLog)
    currLog = ""


def tune_helper(
        model_to_tune=None,
        dataset=None,
        models=None,
        max_layers=10,
        min_layers=2,
        min_dense=32,
        max_dense=512,
        executions_per_trial=3,
        max_trials=1,
        activation='relu',
        loss='categorical_crossentropy',
        metrics='accuracy'):
    logger("Getting target model for tuning...")

    # checks to see which requested model is in the self.models

    # processing for regression feed forward NN
    if model_to_tune == 'regression_ANN':
        logger("Tuning model hyperparameters")
        dataReader = DataReader(dataset)
        data = dataReader.data_generator()
        target = models['regression_ANN']['target']
        target_column = data[models['regression_ANN']['target']]
        data = models['regression_ANN']['preprocesser'].transform(
            data.drop(target, axis=1))
        returned_model = tuneReg(
            data,
            target_column,
            max_layers=max_layers,
            min_layers=min_layers,
            min_dense=min_dense,
            max_dense=max_dense,
            executions_per_trial=executions_per_trial,
            max_trials=max_trials)
        models['regression_ANN'] = {'model': returned_model}
        return returned_model

        # processing for classification feed forward NN
    if model_to_tune == "classification_ANN":
        logger("Tuning model hyperparameters")
        dataReader = DataReader(dataset)
        data = dataReader.data_generator()
        target = models['classification_ANN']['target']
        target_column = data[models['classification_ANN']['target']]
        data = models['classification_ANN']['preprocesser'].transform(
            data.drop(target, axis=1))
        returned_model = tuneClass(
            data,
            target_column,
            models['classification_ANN']['num_classes'],
            max_layers=max_layers,
            min_layers=min_layers,
            min_dense=min_dense,
            max_dense=max_dense,
            executions_per_trial=executions_per_trial,
            max_trials=max_trials,
            activation=activation,
            loss=loss,
            metrics=metrics)
        models['classification_ANN'] = {'model': returned_model}
        return returned_model
        # processing for convolutional NN
    if model_to_tune == "convolutional_NN":
        logger("Tuning model hyperparameters")
        X = models['convolutional_NN']["X"]
        y = models['convolutional_NN']["y"]
        model = tuneCNN(
            np.asarray(X),
            np.asarray(y),
            models["convolutional_NN"]["num_classes"])
        models["convolutional_NN"]["model"] = model
    return models


def stats(dataset=None,
          drop=None,
          column_name=None):
    return


def save(model, save_model, save_path=os.getcwd()):
    model_json = model.to_json()
    with open(save_path + "/model" + str(number) + ".json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(save_path + "/weights" + str(number) + ".h5")
        logger("->", "Saved model to disk as model" + str(number))

def generate_id():
    return str(uuid.uuid4())