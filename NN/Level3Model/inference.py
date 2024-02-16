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
from load_data import test_loader as loader
from load_data import vocab_input_size, vocab_target_size, quest_tokenizer, embedding_matrix

units = 600
embedding_dimension = 300
layers = 2

encoder = Encoder_GGNN(layers)
decoder = Decoder(units, embedding_dimension, vocab_input_size, vocab_target_size, embedding_matrix)
optimizer = tf.keras.optimizers.legacy.Adam()
model = Model_GGNN(encoder, decoder, optimizer, quest_tokenizer)

batch = loader.__next__()
A, B = batch
A = A[:-1]
output = model(A)
model.load_weights('/Users/ethanwakefield/Documents/3rdYearProject/3rd-Year-Project/NN/Level3Model/graph_600_2/21epochs/21epochs').expect_partial()

cnt = 0
for batch in loader:
    A, B = batch
    A = A[:-1]
    # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    # print(A)
    # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    B_words = []
    for i in B[0]:
        if i != 0:
            B_words.append(quest_tokenizer.index_word[i])
    print("TARGET:::::", B_words)
    cnt = cnt+1
    output = model(A)
    print("PREDICTED:::::", output)
    print("=====================================================================================================")
    if cnt == 50:
        break