import string
import Levenshtein

# generates mask for sequence to sequence processinng


def produceMask(instruction_label):
    # creates a mask for a string with representation [1, 0, 2] based on
    # character frequencies

    tokenized_string = [char for char in instruction_label]
    ascii_list = list(string.ascii_lowercase)
    res = {i: instruction_label.count(i) for i in set(instruction_label)}

    mask = [0] * len(ascii_list)

    # stores the masks in an object with the base mask and returns
    for x in range(len(ascii_list)):
        if ascii_list[x] in res.keys():
            mask[x] = res[str(ascii_list[x])]

    return mask


def get_similar_column(instruction, dataset):
    # instruction = produceMask(instruction_label)
    distances = []

    for element in dataset.columns:
        value = Levenshtein.distance(instruction, element)
        distances.append(value)

    val, idx = min((val, idx) for (idx, val) in enumerate(distances))

    # returning the column with the index of most similarity
    return dataset.columns[idx]


# exact exame to get_similar_column(). Adapted to allow for small changes
# for model similarity identification
def get_similar_model(model_requested, model_keys):
    distances = []

    for element in model_keys:
        distances.append(Levenshtein.distance(model_requested, element))

    val, idx = min((val, idx) for (idx, val) in enumerate(distances))

    return model_keys[idx]
