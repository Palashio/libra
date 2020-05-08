from regression_split_functions import (
    initializer,
    preprocesser,
    instruction_identifier,
    set_splitter,
    modeler,
    plotter)

#to initialize a testing pipeline, just create a dictionary with both the instruction you want to test and the path to the dataset.
init_params = {
    'instruction': "Predict median house value",
    'path_to_set': './data/housing.csv',
}

reg_pipeline = [initializer,
                mypreprocessor,
                instruction_identifier,
                set_splitter,
                modeler,
                plotter]




[func(init_params) for func in reg_pipeline]


def mypreprocessor():
    return
