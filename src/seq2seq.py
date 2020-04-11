from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

data_path = "./data/instructions.txt"

with open(data_path, 'r') as f:
    lines = f.read().split('\n')

input_texts = [] 
target_texts = [] 

for x in range(len(lines)):
    input_texts.append(lines[x].rsplit('\t', 1)[0].strip())
    target_texts.append(lines[x].rsplit('\t', 1)[1].strip())

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 8


input_characters = set()
target_characters = set()

for line in input_texts:
    for char in line:
         if char not in input_characters:
            input_characters.add(char)

for line in target_texts:
    for char in line:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)