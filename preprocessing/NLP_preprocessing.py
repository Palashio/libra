# Lemmatizer for text
import re
import spacy
from spacy.lang.en import English


def lemmatize_text(dataset):
    result = []
    nlp = spacy.load('en')
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


# Tokenize text

def tokenize_text(dataset):
    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    for i in range(len(dataset)):
        dataset[i] = tokenizer(dataset[i])
    return dataset


# Cleans up text data by removing unnecessary characters (links,
# punctuation, uppercase letters, numbers, whitespace)

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
        newDataset.append(clean_text)

    return newDataset
