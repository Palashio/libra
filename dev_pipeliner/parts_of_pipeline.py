# Making functions in other directories accesible to this file by
# inserting into sis path
from regression_split_functions import (
    initializer,
    preprocesser,
    instruction_identifier,
    set_splitter,
    modeler)

init_params = {
    'instruction': "Predict median house value",
    'dataset': './data/housing.csv',
}

reg_pipeline = [(initializer, (init_params,)),
            (preprocesser, (init_params,)),
            (instruction_identifier, (init_params,)),
            (set_splitter, (init_params,)),
            (modeler, (init_params,))]


for func, args in reg_pipeline:
    func(*args)
