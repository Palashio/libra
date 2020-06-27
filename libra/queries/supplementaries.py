from libra.modeling.tuner import (tuneReg, 
                                  tuneClass, 
                                  tuneCNN)
import os
from libra.preprocessing.data_reader import DataReader
from keras.preprocessing.image import ImageDataGenerator
from libra.preprocessing.image_preprocesser import (setwise_preprocessing,
                                                    set_distinguisher)

import uuid

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
               'hyperparametes' : returned_pms,
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
               'hyperparametes' : returned_pms,
               'losses': {
                          'training_loss': history.history['loss'],
                           'val_loss': history.history['val_loss']}
               }
        # processing for convolutional NN
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


def stats(dataset=None,
          drop=None,
          column_name=None):
    return


def save(model, save_model, save_path=os.getcwd()):
    global number
    model_json = model.to_json()
    with open(save_path + "/model" + str(number) + ".json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(save_path + "/weights" + str(number) + ".h5")
        logger("->", "Saved model to disk as model" + str(number))
    number= number+1


def generate_id():
    return str(uuid.uuid4())


def get_image_data(data_path, read_mode=None, training_ratio=0.8):
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
