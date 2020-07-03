# Lemmatizer for text
import re
import tensorflow as tf
import spacy
from spacy.lang.en import English
from nltk.corpus import stopwords
from libra.data_generation.dataset_labelmatcher import get_similar_column
from libra.data_generation.grammartree import get_value_instruction


def get_target_values(data, instruction, yLabel):
    # Get target columns
    target = get_similar_column(get_value_instruction(instruction), data)
    X = data[target]
    del data[target]
    # labels
    Y = data[get_similar_column(get_value_instruction(yLabel), data)]
    return X, Y, get_similar_column(get_value_instruction(yLabel), data)


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
        clean_text = clean_text.split()
        stops = set(stopwords.words("english"))
        clean_text = [w for w in clean_text if w not in stops]
        clean_text = " ".join(clean_text)
        newDataset.append(fix_slang(clean_text))

    return newDataset


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


# text encoder
def encode_text(dataset, text):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
        split=' ', char_level=False, oov_token=None, document_count=0)
    tokenizer.fit_on_texts(dataset)
    result = tokenizer.texts_to_sequences(text)
    return result

# def decode_sequence(input_seq, encoder_model, target_word_index, decoder_model, reverse_target_word_index,
#                     max_len_summary=50):
#     e_out, e_h, e_c = encoder_model.predict(input_seq)
#
#     target_seq = np.zeros((1, 1))
#
#     target_seq[0, 0] = target_word_index.get('sostok')
#
#     stop_condition = False
#     decoded_sentence = ''
#     while not stop_condition:
#         output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
#
#         # Sample a token
#         sampled_token_index = np.argmax(output_tokens[0, -1, :]) + 1
#         sampled_token = reverse_target_word_index.get(sampled_token_index)
#
#         if sampled_token != 'eostok':
#             print(decoded_sentence)
#             decoded_sentence += ' ' + sampled_token
#
#             # Exit condition: either hit max length or find stop word.
#         if sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_len_summary - 1):
#            stop_condition = True
#
#         # # Update the target sequence (of length 1).
#         # target_seq = np.zeros((1, 1))
#         # target_seq[0, 0] = sampled_token_index
#
#         # Update internal states
#         e_h, e_c = h, c
#
#     return decoded_sentence
#
#
# def seq2summary(input_seq, target_word_index, reverse_target_word_index):
#     newString = ''
#     for i in input_seq:
#         if (i != 0 and i != target_word_index['sostok']) and i != target_word_index['eostok']:
#             newString = newString + reverse_target_word_index[i] + ' '
#     return newString
#
#
# def seq2text(input_seq, reverse_source_word_index):
#     newString = ''
#     for i in input_seq:
#         if i != 0:
#             newString = newString + reverse_source_word_index[i] + ' '
#     return newString
