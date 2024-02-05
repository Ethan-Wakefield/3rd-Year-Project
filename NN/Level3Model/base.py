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
from load_data import train_loader as loader
from load_data import vocab_input_size, vocab_target_size, quest_tokenizer, embedding_matrix

#TODO
#GRAPH the LOSS


# print(dataset.n_graphs)
# print(dataset[13].a)
# print(dataset[13].x)
# # print(dataset[13].e)
# print(dataset[13].y)
# print("=====================================================================================================")
# print(dataset[13])
     
# inputs, target = batch
# x, a, i = batch
# print("X")
# print(x)
# print("A")
# print(a)

#===========================================================================================================================================================
#Build Model
#===========================================================================================================================================================
units = 600
embedding_dimension = 300
layers = 3

encoder = Encoder_GGNN(layers)
optimizer = tf.keras.optimizers.legacy.Adam()
model = Model_GGNN(layers, units, vocab_input_size, vocab_target_size, optimizer, embedding_dimension, embedding_matrix, quest_tokenizer)


#===========================================================================================================================================================
#Train Model
#===========================================================================================================================================================
loss_object = Loss()

cnt = 1

for batch in loader:
    A, B = batch
    A = A[:-1]
    # print(len(A))
    # print(A[0])
    # print("-----------------------------------")
    # print(A[1])
    # print("-----------------------------------")
    # print(B)
    encoder_input = A
    # decoder_input = B[:, :-1]
    # print(decoder_input)
    # decoder_target = B[:, 1:]
    decoder_target = B
    # Need to pad the decoder input questions, and also the target questions. Just get stuff given to the decoder (B here) in a good form
    # For decoder input do question except last token, for target to question except first token 
    model_loss, model_output = model.train_step(encoder_input, decoder_target, loss_object)

    print(f"Iter: {cnt}   ", model_loss)
    # print("=====================================================================================================")
    

    # train_step((encoder_input, decoder_input), decoder_target)
    # print(cnt)
    cnt += 1
    
    # print("INPUTS")
    # print(inputs)
    # print("TARGET")
    # print(target)
    # print("BLEH")
    # print(bleh)
    # print("LABEL")
    # print(label)
    # print(batch)
    # print("=====================================================================================================")

model.save_weights('/Users/ethanwakefield/Documents/3rdYearProject/3rd-Year-Project/NN/Level3Model/graph_600/3epochs')