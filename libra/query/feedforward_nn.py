from colorama import Fore, Style
from tensorflow.keras.callbacks import EarlyStopping
import os
from libra.preprocessing.image_preprocesser import (setwise_preprocessing,
                                                    csv_preprocessing,
                                                    classwise_preprocessing,
                                                    set_distinguisher,
                                                    already_processed)
from libra.preprocessing.data_reader import DataReader
from keras.models import Sequential
from keras.layers import (Dense, Conv2D, Flatten, MaxPooling2D, )
import pandas as pd
from libra.query.supplementaries import save, generate_id
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from libra.plotting.generate_plots import (generate_regression_plots,
                                           generate_classification_plots)
from libra.preprocessing.data_preprocesser import initial_preprocesser
from libra.modeling.prediction_model_creation import get_keras_model_reg, get_keras_model_class
from sklearn.preprocessing import StandardScaler
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


def regression_ann(
        instruction,
        callback=False,
        ca_threshold=None,
        text=[],
        dataset=None,
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
    Body of the regression function used that is called in the neural network query
    if the data is numerical.
    :param many parameters: used to preprocess, tune, plot generation, and parameterizing the neural network trained.
    :return dictionary that holds all the information for the finished model.
    '''

    logger("Reading in dataset...")

    dataReader = DataReader(dataset)
    data = dataReader.data_generator()
    # data = pd.read_csv(self.dataset)

    if drop is not None:
        data.drop(drop, axis=1, inplace=True)
    data, y, target, full_pipeline = initial_preprocesser(
        data, instruction, preprocess, ca_threshold, text, test_size=test_size, random_state=random_state)
    logger("->", "Target column found: {}".format(target))

    X_train = data['train']
    X_test = data['test']

    # Target scaling
    target_scaler = StandardScaler()

    y_train = target_scaler.fit_transform(np.array(y['train']).reshape(-1, 1))
    y_test = target_scaler.transform(np.array(y['test']).reshape(-1, 1))

    logger("Establishing callback function")

    models = []
    losses = []
    model_data = []

    # callback function to store lowest loss value
    es = EarlyStopping(
        monitor=maximizer,
        mode=callback_mode,
        verbose=0,
        patience=5)

    callback_value = None
    if callback is not False:
        callback_value = [es]

    i = 0

    # get the first 3 layer model
    model = get_keras_model_reg(data, i)

    logger("Training initial model")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(
            X_test,
            y_test),
        callbacks=callback_value,
        verbose=0)
    models.append(history)
    model_data.append(model)

    col_name = [["Initial number of layers ",
                 "| Training Loss ", "| Test Loss "]]
    col_width = max(len(word) for row in col_name for word in row) + 2
    for row in col_name:
        print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
                                                    for word in row)) + " |")
    values = []
    values.append(str(len(model.layers)))
    values.append(
        "| " + str(history.history['loss'][len(history.history['val_loss']) - 1]))
    values.append(
        "| " + str(history.history['val_loss'][len(history.history['val_loss']) - 1]))
    datax = []
    datax.append(values)
    for row in datax:
        print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
                                                    for word in row)) + " |")

    losses.append(history.history[maximizer]
                  [len(history.history[maximizer]) - 1])

    # keeps running model and fit functions until the validation loss stops
    # decreasing
    logger("Testing number of layers...")
    col_name = [["Current number of layers", "| Training Loss", "| Test Loss"]]
    col_width = max(len(word) for row in col_name for word in row) + 2
    for row in col_name:
        print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
                                                    for word in row)) + " |")
    datax = []
    while all(x > y for x, y in zip(losses, losses[1:])):
        model = get_keras_model_reg(data, i)
        history = model.fit(
            X_train,
            y_train,
            callbacks=callback_value,
            epochs=epochs,
            validation_data=(
                X_test,
                y_test), verbose=0)
        model_data.append(model)
        models.append(history)

        values = []
        datax = []
        values.append(str(len(model.layers)))
        values.append(
            "| " + str(history.history['loss'][len(history.history['val_loss']) - 1]))
        values.append(
            "| " + str(history.history['val_loss'][len(history.history['val_loss']) - 1]))
        datax.append(values)
        for row in datax:
            print((" " * 2 * counter) + "| " +
                  ("".join(word.ljust(col_width) for word in row)) + " |")
        del values, datax
        losses.append(history.history[maximizer]
                      [len(history.history[maximizer]) - 1])
        i += 1
    # print((" " * 2 * counter)+ tabulate(datax, headers=col_name, tablefmt='orgtbl'))
    final_model = model_data[losses.index(min(losses))]
    final_hist = models[losses.index(min(losses))]
    print("")
    logger('->', "Best number of layers found: " +
           str(len(final_model.layers)))

    logger('->', "Training Loss: " + str(final_hist.history['loss']
                                         [len(final_hist.history['val_loss']) - 1]))
    logger('->', "Test Loss: " + str(final_hist.history['val_loss']
                                     [len(final_hist.history['val_loss']) - 1]))

    # calls function to generate plots in plot generation
    plots = {}
    if generate_plots:
        init_plots, plot_names = generate_regression_plots(
            models[len(models) - 1], data, y)
        for x in range(len(plot_names)):
            plots[str(plot_names[x])] = init_plots[x]

    if save_model:
        save(final_model, save_model)
    # stores values in the client object models dictionary field
    print("")
    logger("Stored model under 'regression_ANN' key")
    clearLog()
    return {
        'id': generate_id(),
        'model': final_model,
        "target": target,
        "plots": plots,
        "preprocesser": full_pipeline,
        "interpreter": target_scaler,
        'test_data': {'X': X_test, 'y': y_test},
        'losses': {
            'training_loss': final_hist.history['loss'],
            'val_loss': final_hist.history['val_loss']}}


def classification_ann(instruction,
                       callback=False,
                       dataset=None,
                       text=[],
                       ca_threshold=None,
                       preprocess=True,
                       callback_mode='min',
                       drop=None,
                       random_state=49,
                       test_size=0.2,
                       epochs=50,
                       generate_plots=True,
                       maximizer="val_accuracy",
                       save_model=False,
                       save_path=os.getcwd()):
    '''
    Body of the classification function used that is called in the neural network query
    if the data is categorical.
    :param many parameters: used to preprocess, tune, plot generation, and parameterizing the neural network trained.
    :return dictionary that holds all the information for the finished model.
    '''
    logger("Reading in dataset...")

    dataReader = DataReader(dataset)
    data = dataReader.data_generator()

    if drop is not None:
        data.drop(drop, axis=1, inplace=True)

    data, y, remove, full_pipeline = initial_preprocesser(
        data, instruction, preprocess, ca_threshold, text, test_size=test_size, random_state=random_state)
    logger("->", "Target column found: {}".format(remove))

    # Needed to make a custom label encoder due to train test split changes
    # Can still be inverse transformed, just a bit of extra work
    y = pd.concat([y['train'], y['test']], axis=0)

    num_classes = len(np.unique(y))

    X_train = data['train']
    X_test = data['test']

    # ANN needs target one hot encoded for classification
    one_hot_encoder = OneHotEncoder()

    y = pd.DataFrame(
        one_hot_encoder.fit_transform(
            np.reshape(
                y.values,
                (-1,
                 1))).toarray(),
        columns=one_hot_encoder.get_feature_names())

    y_train = y.iloc[:len(X_train)]
    y_test = y.iloc[len(X_train):]

    models = []
    losses = []
    accuracies = []
    model_data = []

    logger("Establishing callback function...")

    # early stopping callback
    es = EarlyStopping(
        monitor=maximizer,
        mode='max',
        verbose=0,
        patience=5)

    callback_value = None
    if callback is not False:
        callback_value = [es]

    i = 0
    model = get_keras_model_class(data, i, num_classes)
    logger("Training initial model...")

    history = model.fit(
        X_train,
        y_train,
        callbacks=callback_value,
        epochs=epochs,
        validation_data=(
            X_test,
            y_test),
        verbose=0)

    model_data.append(model)
    models.append(history)
    col_name = [["Initial number of layers ",
                 "| Training Accuracy ", "| Test Accuracy "]]
    col_width = max(len(word) for row in col_name for word in row) + 2
    for row in col_name:
        print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
                                                    for word in row)) + " |")
    values = []
    values.append(str(len(model.layers)))
    values.append(
        "| " + str(history.history['accuracy'][len(history.history['val_accuracy']) - 1]))
    values.append(
        "| " + str(history.history['val_accuracy'][len(history.history['val_accuracy']) - 1]))
    datax = []
    datax.append(values)
    for row in datax:
        print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
                                                    for word in row)) + " |")
    # print((" " * 2 * counter)+ tabulate(datax, headers=col_name, tablefmt='orgtbl'))
    losses.append(history.history[maximizer]
                  [len(history.history[maximizer]) - 1])
    # keeps running model and fit functions until the validation loss stops
    # decreasing

    logger("Testing number of layers...")
    col_name = [["Current number of layers",
                 "| Training Accuracy", "| Test Accuracy"]]
    col_width = max(len(word) for row in col_name for word in row) + 2

    for row in col_name:
        print((" " * 2 * counter) + "| " + ("".join(word.ljust(col_width)
                                                    for word in row)) + " |")
    datax = []
    while all(x < y for x, y in zip(accuracies, accuracies[1:])):
        model = get_keras_model_class(data, i, num_classes)
        history = model.fit(
            X_train,
            y_train,
            callbacks=callback_value,
            epochs=epochs,
            validation_data=(
                X_test,
                y_test),
            verbose=0)

        values = []
        datax = []
        values.append(str(len(model.layers)))
        values.append(
            "| " + str(history.history['accuracy'][len(history.history['val_accuracy']) - 1]))
        values.append(
            "| " + str(history.history['accuracy'][len(history.history['val_accuracy']) - 1]))
        datax.append(values)
        for row in datax:
            print((" " * 2 * counter) + "| " +
                  ("".join(word.ljust(col_width) for word in row)) + " |")
        del values, datax
        losses.append(history.history[maximizer]
                      [len(history.history[maximizer]) - 1])
        accuracies.append(history.history['val_accuracy']
                          [len(history.history['val_accuracy']) - 1])
        models.append(history)
        model_data.append(model)

        i += 1
    # print((" " * 2 * counter)+ tabulate(datax, headers=col_name, tablefmt='orgtbl'))
    # del values, datax

    final_model = model_data[accuracies.index(max(accuracies))]
    final_hist = models[accuracies.index(max(accuracies))]

    print("")
    logger('->', "Best number of layers found: " +
           str(len(final_model.layers)))
    logger('->', "Training Accuracy: " + str(final_hist.history['accuracy']
                                             [len(final_hist.history['val_accuracy']) - 1]))
    logger('->', "Test Accuracy: " + str(final_hist.history['val_accuracy'][
        len(final_hist.history['val_accuracy']) - 1]))

    # genreates appropriate classification plots by feeding all information
    plots = {}
    if generate_plots:
        plots = generate_classification_plots(
            models[len(models) - 1], data, y, model, X_test, y_test)

    if save_model:
        save(final_model, save_model)

    print("")
    logger("Stored model under 'classification_ANN' key")
    clearLog()
    # stores the values and plots into the object dictionary
    return {
        'id': generate_id(),
        "model": final_model,
        'num_classes': num_classes,
        "plots": plots,
        "target": remove,
        "preprocesser": full_pipeline,
        "interpreter": one_hot_encoder,
        'test_data': {'X': X_test, 'y': y_test},
        'losses': {
            'training_loss': final_hist.history['loss'],
            'val_loss': final_hist.history['val_loss']},
        'accuracy': {
            'training_accuracy': final_hist.history['accuracy'],
            'validation_accuracy': final_hist.history['val_accuracy']}}


def convolutional(instruction=None,
                  read_mode=None,
                  preprocess=True,
                  verbose=0,
                  data_path=os.getcwd(),
                  new_folders=True,
                  image_column=None,
                  training_ratio=0.8,
                  augmentation=True,
                  epochs=10,
                  height=None,
                  width=None):
    '''
    Body of the convolutional function used that is called in the neural network query
    if the data is presented in images.
    :param many parameters: used to preprocess, tune, plot generation, and parameterizing the convolutional neural network trained.
    :return dictionary that holds all the information for the finished model.
    '''

    logger("Generating datasets for classes...")

    if preprocess:
        read_mode_info = set_distinguisher(data_path, read_mode)
        read_mode = read_mode_info["read_mode"]

        training_path = "/proc_training_set"
        testing_path = "/proc_testing_set"

        if read_mode == "setwise":
            processInfo = setwise_preprocessing(
                data_path, new_folders, height, width)
            if not new_folders:
                training_path = "/training_set"
                testing_path = "/testing_set"

        # if image dataset in form of csv
        elif read_mode == "csvwise":
            if training_ratio <= 0 or training_ratio >= 1:
                raise BaseException(f"Test ratio must be between 0 and 1.")
            processInfo = csv_preprocessing(read_mode_info["csv_path"],
                                            data_path,
                                            instruction,
                                            image_column,
                                            training_ratio,
                                            height,
                                            width)

        # if image dataset in form of one folder containing class folders
        elif read_mode == "classwise":
            if training_ratio <= 0 or training_ratio >= 1:
                raise BaseException(f"Test ratio must be between 0 and 1.")
            processInfo = classwise_preprocessing(
                data_path, training_ratio, height, width)

    else:
        training_path = "/training_set"
        testing_path = "/testing_set"
        processInfo = already_processed(data_path)

    input_shape = (processInfo["height"], processInfo["width"], 3)
    input_single = (processInfo["height"], processInfo["width"])
    num_classes = processInfo["num_categories"]
    loss_func = ""

    if num_classes > 2:
        loss_func = "categorical_crossentropy"
    elif num_classes == 2:
        loss_func = "binary_crossentropy"

    logger("Creating convolutional neural network dynamically...")
    # Convolutional Neural Network
    model = Sequential()
    model.add(
        Conv2D(
            64,
            kernel_size=3,
            activation="relu",
            input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss=loss_func,
        metrics=['accuracy'])
    logger("Located image data")
    if augmentation:
        train_data = ImageDataGenerator(rescale=1. / 255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)
        test_data = ImageDataGenerator(rescale=1. / 255)

        logger('Dataset augmented through zoom, shear, flip, and rescale')
    else:
        train_data = ImageDataGenerator()
        test_data = ImageDataGenerator()

    logger("->", "Optimal image size identified: {}".format(input_shape))
    X_train = train_data.flow_from_directory(data_path + training_path,
                                             target_size=input_single,
                                             color_mode='rgb',
                                             batch_size=(32 if processInfo["train_size"] >= 32 else 1),
                                             class_mode=loss_func[:loss_func.find("_")])
    X_test = test_data.flow_from_directory(data_path + testing_path,
                                           target_size=input_single,
                                           color_mode='rgb',
                                           batch_size=(32 if processInfo["test_size"] >= 32 else 1),
                                           class_mode=loss_func[:loss_func.find("_")])


    if epochs < 0:
        raise BaseException("Number of epochs has to be greater than 0.")
    logger('Training image model')
    history = model.fit(
        X_train,
        steps_per_epoch=X_train.n //
        X_train.batch_size,
        validation_data=X_test,
        validation_steps=X_test.n //
        X_test.batch_size,
        epochs=epochs,
        verbose=verbose)

    logger('->', 'Final training accuracy: {}'.format(history.history['accuracy'][len(history.history['accuracy']) - 1]))
    logger('->', 'Final validation accuracy: {}'.format(history.history['val_accuracy'][len(history.history['val_accuracy']) - 1]))
    # storing values the model dictionary

    logger("Stored model under 'convolutional_NN' key")
    clearLog()
    return {
        'id': generate_id(),
        'data_type': read_mode,
        'data_path': data_path,
        'data': {'train': X_train, 'test': X_test},
        'shape': input_shape,
        "model": model,
        'losses': {
            'training_loss': history.history['loss'],
            'val_loss': history.history['val_loss']},
        'accuracy': {
            'training_accuracy': history.history['accuracy'],
            'validation_accuracy': history.history['val_accuracy']},
        'num_classes': (2 if num_classes == 1 else num_classes),
        'data_sizes': {'train_size': processInfo['train_size'], 'test_size': processInfo['test_size']}}

