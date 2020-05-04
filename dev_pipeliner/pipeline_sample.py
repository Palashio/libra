from regression_split_functions import (
    initializer,
    preprocesser,
    instruction_identifier,
    set_splitter,
    modeler)

init_params = {
    # instruction you want to test
    'instruction': "Predict median house value",
    # dataset you want to test on
    'dataset': './data/housing.csv',
}

reg_pipeline = [(initializer, (init_params,)),
                (preprocesser, (init_params,)),
                (instruction_identifier, (init_params,)),
                (set_splitter, (init_params,)),
                (modeler, (init_params,))]


for func, args in reg_pipeline:
    func(*args)
