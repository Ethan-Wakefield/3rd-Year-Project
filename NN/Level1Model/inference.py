import json 
import re                                   # 're' Replication of text.
import numpy as np     
import pickle                     
import pandas as pd                         # 'pandas' to manipulate the dataset.
import tensorflow as tf
from keras.models import Sequential                # 'Sequential' model will be used for training.
from sklearn.model_selection import train_test_split          # 'train_test_split' for splitting the data into train and test data. 
from keras.preprocessing.text import Tokenizer       
from keras.preprocessing.sequence import pad_sequences       # 'pad_sequences' for having same dimmension for each sequence.
from keras.layers import Embedding, LSTM, Flatten, Dense, Bidirectional, Input, TimeDistributed
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.utils import plot_model


model = tf.keras.models.load_model("/Users/ethanwakefield/Documents/3rdYearProject/3rd-Year-Project/NN/Level1Model/model.keras", compile = True)

index = None




with open('/Users/ethanwakefield/Documents/3rdYearProject/3rd-Year-Project/NN/Level1Model/sent_tokenizer.pkl', 'rb') as tokenizer1_file:
    sent_tokenizer = pickle.load(tokenizer1_file)
with open('/Users/ethanwakefield/Documents/3rdYearProject/3rd-Year-Project/NN/Level1Model/quest_tokenizer.pkl', 'rb') as tokenizer2_file:
    quest_tokenizer = pickle.load(tokenizer2_file)

reverse_target_word_index = quest_tokenizer.index_word
reverse_source_word_index = sent_tokenizer.index_word
target_word_index = quest_tokenizer.word_index



encoder_inputs = model.input[0] 
encoder_outputs, state_h, state_c = model.layers[6].output 
encoder_model = Model(inputs=encoder_inputs, outputs=[state_h, state_c])

decoder_inputs = model.input[1]  # input_2
decoder_embedding = model.layers[5]
decoder_embeds = decoder_embedding(decoder_inputs)
decoder_state_input_h = Input(shape=(128,), name="state_h")
decoder_state_input_c = Input(shape=(128,), name="state_c")
decoder_hidden_state_input = Input(shape=(200, 128), name="state_hidden")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[7]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_embeds, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[8]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

def decode_sequence(input_seq):

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        (output_tokens, h, c) = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        print(sampled_token)
        if sampled_token != 'endtok':
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find the stop word.
        if sampled_token == 'endtok' or len(decoded_sentence.split()) \
            >= 30 - 1:
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        states_value = [h, c]

    return decoded_sentence

# To convert sequence to summary
def seq2summary(input_seq):
    newString = ''
    for i in input_seq:
        if i != 0 and i != target_word_index['sostok'] and i \
            != target_word_index['eostok']:
            newString = newString + reverse_target_word_index[i] + ' '

    return newString


# To convert sequence to text
def seq2text(input_seq):
    newString = ''
    for i in input_seq:
        if i != 0:
            newString = newString + reverse_source_word_index[i] + ' '

    return newString

f = open('/Users/ethanwakefield/Documents/3rdYearProject/3rd-Year-Project/NN/dataset/Level1CQA.json')
data = json.load(f)
sentences = data['sentences']
sentences = np.array(sentences)
plop = sent_tokenizer.texts_to_sequences(sentences) 
sent = pad_sequences(plop,  maxlen=200, padding='post')

for i in range (0, 19):
    print("Input: " + sentences[i])
    print("Output: " + decode_sequence(sent[i].reshape(1, 200)))



