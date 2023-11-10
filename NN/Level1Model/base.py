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

#===========================================================================================================================================================
#Load Level1CQA into a python dictionary
#===========================================================================================================================================================
f = open('C:/3rdYearProject/3rd-Year-Project/NN/dataset/Level1CQA.json')
data = json.load(f)

#===========================================================================================================================================================
#Tokenize the sentences and pad them to max length, store for repeated use. Format into np array.
#===========================================================================================================================================================
sentences = data['sentences']
questions = data['questions']
tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(sentences)

sentences = np.array(sentences)
questions = np.array(questions)
both = pd.DataFrame({'sent': sentences,'quest': questions})
print(both.head(10))

with open('C:/3rdYearProject/3rd-Year-Project/NN/Level1Model/tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

#with open('C:/3rdYearProject/3rd-Year-Project/NN/Level1Model/tokenizer.pkl', 'rb') as tokenizer_file:
#    tokenizer = pickle.load(tokenizer_file)
sentences = tokenizer.texts_to_sequences(sentences)
vocab_size = len(tokenizer.word_index) + 1
max_len = 200
sentences = pad_sequences(sentences, padding = 'post', maxlen = max_len)

#===========================================================================================================================================================
#Establish glove embeddings
#===========================================================================================================================================================
embeddings_dictionary = dict()
glove_file = open('C:/3rdYearProject/3rd-Year-Project/NNTest/SentAnalysisTest/glove/glove.6B.50d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = np.zeros((vocab_size, 50))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

#===========================================================================================================================================================
#Establish Encoder & Decoder Training Models
#===========================================================================================================================================================
#ENCODER
#Take inputs and embed using glove
encoder_input = Input(shape=(max_len, ))
encoder_embedding = Embedding(vocab_size, 50,  weights=[embedding_matrix], input_length=max_len, trainable=False)(encoder_input)

#First LSTM
encoder_LSTM1 = LSTM(128, return_sequences=True, return_state=True)
(encoder_output_LSTM1, h_state_1, c_state_1) = encoder_LSTM1(encoder_embedding)

#Second LSTM
encoder_LSTM1 = LSTM(128, return_sequences=True, return_state=True)
(encoder_output_LSTM2, h_state_2, c_state_2) = encoder_LSTM1(encoder_output_LSTM1)

#Third LSTM
encoder_LSTM1 = LSTM(128, return_sequences=True, return_state=True)
(encoder_output_LSTM3, h_state_3, c_state_3) = encoder_LSTM1(encoder_output_LSTM2)

#DECODER
decoder_input = Input(shape=(None, ))  
decoder_embedding = Embedding(vocab_size, 50,  weights=[embedding_matrix], input_length=max_len, trainable=False)(decoder_input)

#Only LSTM
decoder_LSTM = LSTM(128, return_sequences=True, return_state=True)
(decoder_output, h_state_dec, c_state_dec) = decoder_LSTM(decoder_embedding, initial_state=[h_state_3, c_state_3])

#Sample over every time step
decoder_dense = TimeDistributed(Dense(vocab_size, activation='softmax'))
decoder_output = decoder_dense(decoder_output)

#Summary
model = Model([encoder_input, decoder_input], decoder_output)
print(model.summary())

#Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

#Save model
model.save("C:/3rdYearProject/3rd-Year-Project/NN/Level1Model/model.keras")