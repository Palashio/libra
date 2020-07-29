import os
import re
import sys

import tensorflow as tf
from nltk.corpus import stopwords
from spacy.lang.en import English

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


"""
Tokenizes sentences and returns input ids
"""


def tokenize_for_input_ids(sentences, tokenizer, max_length):
    input_ids, input_masks, input_segments = [], [], []
    for sentence in sentences:
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_length, pad_to_max_length=True,
                                       return_attention_mask=True, return_token_type_ids=True, truncation=True)
        input_ids.append(inputs['input_ids'])

    return input_ids


"""
Add a specific prefix to all text data
"""


def add_prefix(dataset, prefix):
    for i in range(len(dataset)):
        dataset[i] = prefix + dataset[i]
    return dataset


"""
Used to suppress HuggingFace model loading output
"""


class NoStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, 'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush();
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush();
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()
