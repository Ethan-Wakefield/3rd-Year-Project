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

with open('NN/tokenizers/sent_tokenizer.pkl', 'rb') as tokenizer1_file:
    sent_tokenizer = pickle.load(tokenizer1_file)
with open('NN/tokenizers/quest_tokenizer.pkl', 'rb') as tokenizer2_file:
    quest_tokenizer = pickle.load(tokenizer2_file)

def summon_matrix(mode, vocab_input_size):
    #Calculate and save the embedding matrix and dictionary
    if mode == "save":
        glove_file = open('NN/glove/glove.840B.300d.txt', encoding="utf8")
        embeddings_dictionary = dict()
        for line in glove_file:
            try:
                    records = line.split()
                    word = records[0]
                    vector_dimensions = np.asarray(records[1:], dtype='float32')
                    embeddings_dictionary[word] = vector_dimensions
            except (ValueError, IndexError):
                continue
        glove_file.close()

        embedding_matrix = np.zeros((vocab_input_size, 300))
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

f = open('NN/dataset/Level3CQA.json')
data = json.load(f)

vocab_input_size = len(sent_tokenizer.word_index) + 1
vocab_target_size = len(quest_tokenizer.word_index) + 1

embedding_matrix, embeddings_dictionary = summon_matrix("load", vocab_input_size)
dataset = My_Dataset(embeddings_dictionary, quest_tokenizer)

indexes = np.random.permutation(len(dataset))
test_split = int(0.9 * len(dataset))
index_train, index_test = np.split(indexes, [test_split])
train_data = dataset[index_train]
test_data = dataset[index_test]

train_loader = DisjointLoader(train_data, batch_size=1, epochs=3, node_level=False)
test_loader = DisjointLoader(test_data, batch_size=1, node_level=False)