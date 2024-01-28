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

#===========================================================================================================================================================
#Dataset Class
#===========================================================================================================================================================
class My_Dataset(Dataset):
    def __init__(self, embeddings_dictionary, quest_tokenizer, **kwargs):
        self.embeddings_dictionary = embeddings_dictionary
        self.quest_tokenizer = quest_tokenizer
        super().__init__(**kwargs)

    #Necessary override method to generate the dataset
    def read(self):
        output = []
        f = open('/Users/ethanwakefield/Documents/3rdYearProject/3rd-Year-Project/NN/dataset/Level3CQA.json')
        data = json.load(f)
        first = data["Super_Bowl_50"]
        for item in first:
            kg = item[0]
            questions = item[1]
            if (len(questions) == 0):
                continue

            question = questions[0]
            question = self.clean(question)
            question = 'sostok ' + question + ' endtok'
            question = self.quest_tokenizer.texts_to_sequences([question])
            question = pad_sequences(question,  maxlen=20, padding='post')[0]

            a, node_features = self.levi_graph(kg)
            node_feature_vectors = self.features_to_vectors(node_features)
            output.append(spektral.data.Graph(a=a, x=node_feature_vectors, y=question))
        return output
    
    def clean(self, text):
        sentence = text.lower()
        sentence = sentence.replace("\u2013", "-")
        sentence = sentence.replace("?", '')
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence

    #Create Levi graphs from existing edge labelled graphs
    def levi_graph(self, edge_list):
        nodes = set()
        for triple in edge_list:
            nodes.add(triple["head"])
            nodes.add(triple["tail"])

        #Create a mapping from nodes to indices
        node_to_index = {node: i for i, node in enumerate(nodes)}

        #Initialize a sparse matrix
        num_nodes = len(nodes)
        adjacency_matrix = dok_matrix((num_nodes, num_nodes), dtype=np.int8)
        node_features = []
        for node, i in node_to_index.items():
            node_features.append([i, node])
        edge_features_dict = dict()

        #Populate the adjacency matrix
        num_edges = len(edge_list)
        for triple in edge_list:
            head_index = node_to_index[triple["head"]]
            tail_index = node_to_index[triple["tail"]]
            adjacency_matrix[head_index, tail_index] = 1
            edge_features_dict[(head_index, tail_index)] = triple["relation"]

        #Convert to Levi graph
        incrementer = num_nodes
        levi_adjacency_matrix = dok_matrix((num_nodes + num_edges, num_nodes + num_edges), dtype=np.int8)
        for i in range(0, num_nodes):
            for j in range(0, num_nodes):
                if adjacency_matrix[i,j] == 1:
                    #Todo get the edge labels
                    levi_adjacency_matrix[i,incrementer] = 1
                    levi_adjacency_matrix[incrementer,j] = 1
                    node_features.append([incrementer, edge_features_dict.get((i,j))])
                    incrementer = incrementer + 1         

        #Convert to a compressed sparse row (CSR) matrix
        levi_adjacency_matrix_csr = levi_adjacency_matrix.tocsr()
        # print("\n")
        # print(edge_list)
        # print("\n")
        # print(node_features)
        # print("\n")
        # print(levi_adjacency_matrix_csr)
        # print("\n")
        # print("====================")
        return levi_adjacency_matrix_csr, node_features
    
    def features_to_vectors(self, node_features):
        embedding_size = 50
        averaged_embeddings = []

        for node in node_features:
            feature_words = node[1].split()
            node_embedding = np.zeros(embedding_size)
            valid_words_count = 0
            for word in feature_words:
                embedding_vector = self.embeddings_dictionary.get(word)
                if embedding_vector is not None:
                    node_embedding += embedding_vector
                    valid_words_count += 1
            if valid_words_count > 0:
                node_embedding /= valid_words_count

            averaged_embeddings.append(node_embedding)

        averaged_embeddings = np.array(averaged_embeddings)
        return averaged_embeddings

#===========================================================================================================================================================
#Define Encoder
#===========================================================================================================================================================
class Encoder_GGNN(Model):
    def __init__(self, n_layers):
        super().__init__()
        self.gated_graph_conv = GatedGraphConv(channels=50, n_layers=n_layers, name='encoder')
        
    def call(self, inputs):
        out = self.gated_graph_conv(inputs)
        return out

#===========================================================================================================================================================
#Define Decoder
#===========================================================================================================================================================
class Decoder(Model):
    def __init__(self, units, emb_dimension, vocab_input_size, vocab_target_size, embedding_matrix):
        super(Decoder, self).__init__()
        self.units = units
        # GRU needs input (batch_size, time_steps, features). Batch size is 1. This is why our B vector in base.py looks like that.
        self.gru = GRU(units, return_sequences=True, return_state=True)
        self.decoder_dense = Dense(vocab_target_size, activation='softmax')
        self.decoder_embedding = Embedding(vocab_input_size, emb_dimension,  weights=[embedding_matrix], trainable=False)

    def call(self, input, decoder_hidden_state):
        input = self.decoder_embedding(input)
        (output, state) = self.gru(input, initial_state=decoder_hidden_state)
        output = tf.reshape(output, (-1, output.shape[2]))
        output = self.decoder_dense(output)
        return output, state

#===========================================================================================================================================================
#Define Loss
#===========================================================================================================================================================
class Loss():
    def __init__(self):
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.optimizer = tf.keras.optimizers.legacy.Adam()

    def loss_function(self, real, pred):
        #Masking
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        #Average over the batch
        return tf.reduce_mean(loss_)

#===========================================================================================================================================================
#Define Model
#===========================================================================================================================================================
class Model_GGNN(Model):
    def __init__(self, n_layers, units, vocab_input_size, vocab_target_size, optimizer, emb_dimension, embedding_matrix):
        super().__init__()
        self.encoder = Encoder_GGNN(n_layers)
        self.max_pool = GlobalMaxPool()
        self.decoder = Decoder(units, emb_dimension, vocab_input_size, vocab_target_size, embedding_matrix)
        self.optimizer = optimizer

    def call(self, encoder_input, decoder_input):
        pass
    
    def train_step(self, encoder_input, decoder_input, target, loss_object):
        
        with tf.GradientTape() as tape:
            intermediate_representation= self.encoder(encoder_input)
            intermediate_representation = self.max_pool(intermediate_representation)
            prediction, _ = self.decoder(decoder_input, intermediate_representation)
            loss = loss_object.loss_function(target, prediction)
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return loss, prediction
    
    def train(self):
        pass
    
    