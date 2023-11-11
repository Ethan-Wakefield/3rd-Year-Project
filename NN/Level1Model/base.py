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
#sent_tokenizer = Tokenizer()
#quest_tokenizer = Tokenizer()

sentences = np.array(sentences)
questions = np.array(questions)
questions = ['_START_ '+ str(line) + ' _END_' for line in questions]
both = pd.DataFrame({'sent': sentences,'quest': questions})


with open('C:/3rdYearProject/3rd-Year-Project/NN/Level1Model/sent_tokenizer.pkl', 'rb') as tokenizer1_file:
    sent_tokenizer = pickle.load(tokenizer1_file)
with open('C:/3rdYearProject/3rd-Year-Project/NN/Level1Model/quest_tokenizer.pkl', 'rb') as tokenizer2_file:
    quest_tokenizer = pickle.load(tokenizer2_file)

train_sent, test_sent, train_quest, test_quest = train_test_split(
    np.array(both["sent"]),
    np.array(both["quest"]),
    test_size=0.1,
    random_state=0,
    shuffle=True,
)
max_len = 200

#Comment out if loading tokenizer
#sent_tokenizer.fit_on_texts(list(train_sent))
#quest_tokenizer.fit_on_texts(list(train_quest))

# Convert text sequences to integer sequences 
train_sent_seq = sent_tokenizer.texts_to_sequences(train_sent) 
train_quest_seq = quest_tokenizer.texts_to_sequences(train_quest)
test_sent_seq = sent_tokenizer.texts_to_sequences(test_sent) 
test_quest_seq = quest_tokenizer.texts_to_sequences(test_quest)

# Pad zero upto maximum length
train_sent = pad_sequences(train_sent_seq,  maxlen=max_len, padding='post')
train_quest = pad_sequences(train_quest_seq, maxlen=max_len, padding='post')
test_sent = pad_sequences(test_sent_seq,  maxlen=max_len, padding='post')
test_quest = pad_sequences(test_quest_seq, maxlen=max_len, padding='post')

# Size of vocabulary (+1 for padding token)
vocab_size = len(sent_tokenizer.word_index) + 1
questvocab_size = len(quest_tokenizer.word_index) + 1
print(vocab_size)
print(questvocab_size)

'''
with open('C:/3rdYearProject/3rd-Year-Project/NN/Level1Model/sent_tokenizer.pkl', 'wb') as tokenizer1_file:
    pickle.dump(sent_tokenizer, tokenizer1_file)
with open('C:/3rdYearProject/3rd-Year-Project/NN/Level1Model/quest_tokenizer.pkl', 'wb') as tokenizer2_file:
    pickle.dump(quest_tokenizer, tokenizer2_file)
'''

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
for word, index in sent_tokenizer.word_index.items():
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
#model.save("C:/3rdYearProject/3rd-Year-Project/NN/Level1Model/model.keras")


history = model.fit(
    [train_sent, train_quest[:, :-1]],
    train_quest.reshape(train_quest.shape[0], train_quest.shape[1], 1)[:, 1:],
    epochs=2,
    callbacks=[es],
    batch_size=64,
    validation_data=([test_sent, test_quest[:, :-1]],
                     test_quest.reshape(test_quest.shape[0], test_quest.shape[1], 1)[:
                     , 1:]),
    verbose=1
    )