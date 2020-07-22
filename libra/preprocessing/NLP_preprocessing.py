import re
import tensorflow as tf
import spacy
from spacy.lang.en import English
from nltk.corpus import stopwords
from libra.data_generation.dataset_labelmatcher import get_similar_column
from libra.data_generation.grammartree import get_value_instruction

"""
Returns the target X, Y, and label of Y column from the instruction
"""


def get_target_values(data, instruction, yLabel):
    # labels
    label = get_similar_column(get_value_instruction(yLabel), data)
    Y = data[label]
    del data[label]
    # Get target columns
    target = get_similar_column(get_value_instruction(instruction), data)
    X = data[target]
    return X, Y, label


"""
Takes a list of text values and returns a lemmatized version of this text.
"""


def lemmatize_text(dataset):
    result = []
    nlp = English()
    for text in range(len(dataset)):
        word = ""
        doc = nlp(dataset[text])
        for token in doc:
            if word == "":
                word = token.lemma_
            else:
                word = word + " " + token.lemma_
        result.append(word)
    return result


"""
Takes a list of text values and returns a tokenized version of this text, using spacy.
"""


# Tokenize text
def tokenize_text(dataset):
    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    for i in range(len(dataset)):
        dataset[i] = tokenizer(dataset[i])
    return dataset


"""
Cleans up text data by removing unnecessary characters (links,
punctuation, uppercase letters, numbers, whitespace)
"""


def text_clean_up(dataset):
    newDataset = []
    for text in dataset:
        clean_text = re.sub(r'http\S+', '', text)
        punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
        clean_text = ''.join(
            ch for ch in clean_text if ch not in set(punctuation))
        clean_text = clean_text.lower()
        clean_text = re.sub(r'\d', ' ', clean_text)
        clean_text = ' '.join(clean_text.split())
        clean_text = clean_text.split()
        stops = set(stopwords.words("english"))
        clean_text = [w for w in clean_text if w not in stops]
        clean_text = " ".join(clean_text)
        newDataset.append(fix_slang(clean_text))

    return newDataset


"""
Cleans up text data by changing slang like concatenations into more meaningful words
"""


def fix_slang(text):
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text


"""
Takes a dataset and text and encodes the given text based on the vocabulary in the dataset
"""


def encode_text(dataset, text):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
        split=' ', char_level=False, oov_token=None, document_count=0)
    tokenizer.fit_on_texts(dataset)
    result = tokenizer.texts_to_sequences(text)
    return result
