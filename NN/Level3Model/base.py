import re                                   # 're' Replication of text.
import json
import numpy as np     
from scipy.sparse import dok_matrix
import pickle                     
import pandas as pd                         # 'pandas' to manipulate the dataset.
import tensorflow as tf
from keras.models import Sequential                # 'Sequential' model will be used for training.
from sklearn.model_selection import train_test_split          # 'train_test_split' for splitting the data into train and test data. 
from keras.preprocessing.text import Tokenizer       
from keras.preprocessing.sequence import pad_sequences       # 'pad_sequences' for having same dimmension for each sequence.
from keras.layers import Embedding, LSTM, Flatten, Dense , Dropout, GRU, TimeDistributed
import spektral
from spektral.layers import GCNConv, GlobalSumPool, GatedGraphConv, GlobalMaxPool
from spektral.data import Dataset, DisjointLoader
from keras.models import Model
from full_model import My_Dataset, Encoder_GGNN, Decoder, Model_GGNN, Loss


def summon_matrix(mode, vocab_input_size):
    #Calculate and save the embedding matrix and dictionary
    if mode == "save":
        glove_file = open('NN/glove/glove.6B.50d.txt', encoding="utf8")
        embeddings_dictionary = dict()
        for line in glove_file:
                    records = line.split()
                    word = records[0]
                    vector_dimensions = np.asarray(records[1:], dtype='float32')
                    embeddings_dictionary[word] = vector_dimensions
        glove_file.close()

        embedding_matrix = np.zeros((vocab_input_size, 50))
        for word, index in sent_tokenizer.word_index.items():
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        with open('NN/Level3Model/matrix/embedding_matrix', 'wb') as embedding_matrix_file:
            pickle.dump(embedding_matrix, embedding_matrix_file)
        with open('NN/Level3Model/matrix/embeddings_dictionary', 'wb') as embeddings_dictionary_file:
            pickle.dump(embeddings_dictionary, embeddings_dictionary_file)
        
        return embedding_matrix, embeddings_dictionary
    
    #Load the embedding matrix and dictionary
    else:
        with open('NN/Level3Model/matrix/embedding_matrix', 'rb') as embedding_matrix_file:
            embedding_matrix = pickle.load(embedding_matrix_file)
        with open('NN/Level3Model/matrix/embeddings_dictionary', 'rb') as embeddings_dictionary_file:
            embeddings_dictionary = pickle.load(embeddings_dictionary_file)
        return embedding_matrix, embeddings_dictionary

#===========================================================================================================================================================
#Prepare Data
#===========================================================================================================================================================

f = open('NN/dataset/Level3CQA.json')
data = json.load(f)

with open('NN/tokenizers/sent_tokenizer.pkl', 'rb') as tokenizer1_file:
    sent_tokenizer = pickle.load(tokenizer1_file)
with open('NN/tokenizers/quest_tokenizer.pkl', 'rb') as tokenizer2_file:
    quest_tokenizer = pickle.load(tokenizer2_file)

vocab_input_size = len(sent_tokenizer.word_index) + 1
vocab_target_size = len(quest_tokenizer.word_index) + 1

embedding_matrix, embeddings_dictionary = summon_matrix("save", vocab_input_size)
dataset = My_Dataset(embeddings_dictionary)
loader = DisjointLoader(dataset, batch_size=1)

batch = loader.__next__()
# inputs, target = batch
x, a, i = batch
print("X")
print(x)
print("A")
print(a)

#===========================================================================================================================================================
#Build Model
#===========================================================================================================================================================
units = 256
embedding_dimension = 50
layers = 3

model = Model_GGNN(layers, units, vocab_target_size, embedding_dimension, embedding_matrix)
optimizer = tf.keras.optimizers.legacy.Adam()

#===========================================================================================================================================================
#Train Model
#===========================================================================================================================================================
# loss_object = Loss()

# @tf.function(input_signature=loader.tf_signature())  # Specify signature here
# def train_step(inputs, target):
#     with tf.GradientTape() as tape:
#         prediction = model(inputs)
#         loss = loss_object.loss_function(target, prediction)
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))