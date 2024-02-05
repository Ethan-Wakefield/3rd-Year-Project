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

#TODO
#GRAPH the LOSS

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
dataset = My_Dataset(embeddings_dictionary, quest_tokenizer)
loader = DisjointLoader(dataset, batch_size=1, epochs=6, node_level=False)

print(dataset.n_graphs)
print(dataset[13].a)
print(dataset[13].x)
# print(dataset[13].e)
print(dataset[13].y)
print("=====================================================================================================")
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

model.save_weights('/Users/ethanwakefield/Documents/3rdYearProject/3rd-Year-Project/NN/Level3Model/graph_600/6epochs')