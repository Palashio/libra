from libra.modeling.tuner import (tuneReg,
                                  tuneClass,
                                  tuneCNN)
import os
from libra.preprocessing.data_reader import DataReader
from keras.preprocessing.image import ImageDataGenerator
from libra.preprocessing.image_preprocesser import (setwise_preprocessing,
                                                    set_distinguisher)

import uuid
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

currLog = ""
counter = 0
number = 0


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


def logger(instruction, found=""):
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
        metrics='accuracy',
        seed=42,
        objective='val_accuracy',
        directory='my_dir',
        epochs=10,
        step=32,
        verbose=0,
        test_size=0.2
):
    '''
    Helper function that calls the appropriate tuning function
    :param instruction: the objective that you want to reduce dimensions to maximize
    :return the updated models dictionary
    '''
    logger("Getting target model for tuning...")

    # checks to see which requested model is in the self.models

    # processing for regression feed forward NN
    if model_to_tune == 'regression_ANN':
        logger("Tuning model hyperparameters...")
        dataReader = DataReader(dataset)
        data = dataReader.data_generator()
        target = models['regression_ANN']['target']
        target_column = data[models['regression_ANN']['target']]
        data = models['regression_ANN']['preprocesser'].transform(
            data.drop(target, axis=1))
        returned_model, returned_pms, history = tuneReg(
            data,
            target_column,
            max_layers=max_layers,
            min_layers=min_layers,
            min_dense=min_dense,
            max_dense=max_dense,
            executions_per_trial=executions_per_trial,
            max_trials=max_trials,
            epochs=epochs,
            activation=activation,
            step=step,
            verbose=verbose,
            test_size=test_size
        )
        models['regression_ANN'] = {
            'model': returned_model,
            'hyperparametes': returned_pms,
            'losses': {
                'training_loss': history.history['loss'],
                'val_loss': history.history['val_loss']}
        }

        # processing for classification feed forward NN
    elif model_to_tune == "classification_ANN":
        logger("Tuning model hyperparameters...")
        dataReader = DataReader(dataset)
        data = dataReader.data_generator()
        target = models['classification_ANN']['target']
        target_column = data[models['classification_ANN']['target']]
        data = models['classification_ANN']['preprocesser'].transform(
            data.drop(target, axis=1))
        returned_model, returned_pms, history = tuneClass(
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
            metrics=metrics,
            epochs=epochs,
            step=step,
            verbose=verbose,
            test_size=test_size
        )
        models['classification_ANN'] = {
            'model': returned_model,
            'hyperparametes': returned_pms,
            'losses': {
                'training_loss': history.history['loss'],
                'val_loss': history.history['val_loss']}
        }

    elif model_to_tune == "convolutional_NN":
        logger("Tuning model hyperparameters...")
        X_train, X_test, height, width, num_classes = get_image_data(dataset)
        model, returned_pms, history = tuneCNN(
            X_train,
            X_test,
            height,
            width,
            num_classes,
            executions_per_trial=executions_per_trial,
            max_trials=max_trials,
            seed=seed,
            objective=objective,
            directory=directory,
            epochs=epochs,
            verbose=verbose,
            test_size=test_size
        )
        models["convolutional_NN"]["model"] = model
        models["convolutional_NN"]["hyperparametes"] = returned_pms,
        models["convolutional_NN"]["losses"] = {
            'training_loss': history.history['loss'],
            'val_loss': history.history['val_loss']}

    return models


def save(model, save_model, save_path=os.getcwd()):
    '''
    function used to save a model with its .h5 file and .json instance
    :param modeL; the models dictionary used
    :param save_model, whether you want to save the model
    :param save_path: location where you want to save the model
    '''
    global number
    model_json = model.to_json()
    with open(save_path + "/model" + str(number) + ".json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(save_path + "/weights" + str(number) + ".h5")
        logger("->", "Saved model to disk as model" + str(number))
    number = number + 1


def generate_id():
    '''
    function to generate a unique id.
    '''
    return str(uuid.uuid4())


def get_image_data(data_path, read_mode=None, training_ratio=0.8):
    '''
    function to get image data from a certain folder specifically for CNN tuning. Assumes CNN query has already been called.
    :param data_path: represents the location of the two training/testing folders.
    :param read_mode: represents the type of reading it does: setwise, pathwise, or classwise
    :param training_ratio: represents the size of the training / testing set.
    '''
    training_path = "/proc_training_set"
    testing_path = "/proc_testing_set"

    read_type = set_distinguisher(data_path, read_mode)['read_mode']

    process_info = setwise_preprocessing(data_path, True)

    input_shape = (process_info["height"], process_info["width"], 3)
    input_single = (process_info["height"], process_info["width"])
    num_classes = process_info["num_categories"]
    loss_func = ""

    if num_classes > 2:
        loss_func = "categorical_crossentropy"
    elif num_classes == 2:
        loss_func = "binary_crossentropy"

    train_data = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
    test_data = ImageDataGenerator(rescale=1. / 255)

    X_train = train_data.flow_from_directory(data_path + training_path,
                                             target_size=input_single,
                                             color_mode='rgb',
                                             batch_size=(32 if process_info["train_size"] >= 32 else 1),
                                             class_mode=loss_func[:loss_func.find("_")])
    X_test = test_data.flow_from_directory(data_path + testing_path,
                                           target_size=input_single,
                                           color_mode='rgb',
                                           batch_size=(32 if process_info["test_size"] >= 32 else 1),
                                           class_mode=loss_func[:loss_func.find("_")])

    return X_train, X_test, process_info['height'], process_info['width'], num_classes


def get_model_data(self, model):
    '''
    function to print out model dictionary information
    :param modeL: represents the specific model key that you're using to index dictionary
    '''
    if model in self.models:
        data = [key for key in self.models[model].keys()]
        print(data)
    else:
        raise Exception("The requested model has not been applied to the client.")


def get_operators(self, model):
    '''
    gets the operators that were used to preprocess the dataset for prediction.
    :param modeL; the key in the models dictionary.
    '''
    defined = ['plots', 'accuracy', 'losses']
    if model in self.models:
        operations = [func + "()" for func in self.models[model].keys() if func in defined]
        if len(operations) > 0:
            print(operations)
        else:
            raise Exception(
                "There are no built-in operators defined for this model. Please refer to the models dictionary.")
    else:
        raise Exception("The requested model has not been applied to the client.")


def get_accuracy(self, model):
    '''
    gets the accuracies for a specific model
    :param modeL; the key in the models dictionary.
    '''
    if model in self.models:
        if 'accuracy' in self.models[model].keys():
            return self.models[model]['accuracy']
        elif 'cross_val_score' in self.models[model].keys():
            return {'cross_val_score': self.models[model]['cross_val_score']}
        else:
            raise Exception("Accuracy is not defined for {}".format(model))
    else:
        raise Exception("The requested model has not been applied to the client.")


def get_losses(self, model):
    '''
    gets the losses for that specific model
    :param modeL; the key in the models dictionary.
    '''
    if model in self.models:
        if 'losses' in self.models[model].keys():
            return self.models[model]['losses']
        else:
            raise Exception("Losses are not defined for {}".format(model))
    else:
        raise Exception("The requested model has not been applied to the client.")


def get_target(self, model):
    '''
    gets the target for that specific model if it exists.
    :param modeL; the key in the models dictionary.
    '''
    if model in self.models:
        if 'target' in self.models[model].keys():
            return self.models[model]['target']
        else:
            raise Exception("Target is not defined for {}".format(model))
    else:
        raise Exception("The requested model has not been applied to the client.")


def get_vocab(self, model):
    '''
    gets the vocab that was used in the nlp queries
    :param modeL; the key in the models dictionary.
    '''
    if model in self.models:
        if 'vocabulary' in self.models[model].keys():
            return self.models[model]['vocabulary']
        else:
            raise Exception("Vocabulary is not defined for {}".format(model))
    else:
        raise Exception("The requested model has not been applied to the client.")



def get_plots(self, model="", plot="", save=False):
    '''
    function to get plots and then save them if appropriate
    :param modeL; the key in the models dictionary.
    :param plot: specific plot to get if applicable
    :param save: whether to save the file as a .png
    '''
    # no model or plot specified so plot all
    if model == "" and plot == "":
        for each_model in self.models:
            for each_plot in self.models[each_model]["plots"]:
                save_and_plot(self, each_model, each_plot, save)
    # show plots for specified model
    elif model != "" and plot == "":
        if "plots" in self.models[model].keys():
            for each_plot in self.models[model]['plots']:
                save_and_plot(self, model, each_plot, save)
        else:
            raise Exception("{} does not have plots".format(model))
    # show specified plot for specified model
    elif model != "" and plot != "":
        if plot in self.models[model]['plots'].keys():
            theplot = self.models[model]['plots'][plot]
            save_and_plot(self, model, theplot, save)
        else:
            raise Exception("{} is not available for {}".format(plot, model))
    else:
        raise Exception("Invalid plotting input")


# function to save and plot
def save_and_plot(self, modelname, plotname, save):
    '''
    helper function to save the model as a .png
    :param modelname: represents the model from which to save
    :param plotname: specific plot to save
    :param save: whether to save it or not
    '''
    figure.savefig('{}_{}.png'.format(model, plot))
    img = self.models[modelname]['plots'][plotname]
    path = "{}_{}.png".format(modelname, plotname)
    img.savefig(path)
    saved = Image.open(path)
    saved.show()
    if (save == False):
        currpath = os.getcwd()
        os.remove(currpath + '\\' + path)


def get_standard_training_output_keras(epochs, history):
    '''
    helper output for logger
    :param epochs: is the number of epochs model was running for
    :param history: the keras history object
    :return string of the output
    '''
    global counter
    col_name=[["Epochs","| Training Loss ","| Validation Loss "]]
    col_width = max(len(word) for row in col_name for word in row) + 2
    for row in col_name:
        print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
                                                    for word in row)) + " |")
    
    for i, j, k in zip(range(epochs), history.history["loss"], history.history["val_loss"]):
        values = []
        values.append(str(i))
        values.append("| " + str(j))
        values.append( "| " + str(k))
        datax = []
        datax.append(values)
        for row in datax:
            print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
                                                    for word in row)) + " |")


def get_standard_training_output_generic(epochs, loss, val_loss):
    '''
    helper output for logger
    :param epochs: is the number of epochs model was running for
    :param loss: is the amount of loss in the training instance
    :param val_loss: just validation loss
    :return string of the output
    '''
    global counter
    col_name=[["Epochs","| Training Loss ","| Validation Loss "]]
    col_width = max(len(word) for row in col_name for word in row) + 2
    for row in col_name:
        print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
                                                    for word in row)) + " |")

    for i, j, k in zip(range(epochs), loss, val_loss):
        values = []
        values.append(str(i))
        values.append("| " + str(j))
        values.append( "| " + str(k))
        datax = []
        datax.append(values)
        for row in datax:
            print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
                                                    for word in row)) + " |")

