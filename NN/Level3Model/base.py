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

#If we have a finite number of relstions, we can one hot encode them. 
#OR (bi)LSTM the nodes & edges to get our fixed size vector representation a-la Subgraph... paper.


#===========================================================================================================================================================
#Establish glove embeddings and word tokenization
#===========================================================================================================================================================
f = open('/Users/ethanwakefield/Documents/3rdYearProject/3rd-Year-Project/NN/dataset/Level3CQA.json')
data = json.load(f)

with open('/Users/ethanwakefield/Documents/3rdYearProject/3rd-Year-Project/NN/Level1Model/sent_tokenizer.pkl', 'rb') as tokenizer1_file:
    sent_tokenizer = pickle.load(tokenizer1_file)
with open('/Users/ethanwakefield/Documents/3rdYearProject/3rd-Year-Project/NN/Level1Model/quest_tokenizer.pkl', 'rb') as tokenizer2_file:
    quest_tokenizer = pickle.load(tokenizer2_file)

vocab_input_size = len(sent_tokenizer.word_index) + 1
vocab_target_size = len(quest_tokenizer.word_index) + 1

glove_file = open('/Users/ethanwakefield/Documents/3rdYearProject/3rd-Year-Project/NNTest/SentAnalysisTest/glove.6B.50d.txt', encoding="utf8")
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



#===========================================================================================================================================================
#Spektral Dataset 
#===========================================================================================================================================================
class MyDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    #Necessary override method to generate the dataset
    def read(self):
        output = []
        f = open('/Users/ethanwakefield/Documents/3rdYearProject/3rd-Year-Project/NN/dataset/Level3CQA.json')
        data = json.load(f)
        first = data["Super_Bowl_50"]
        for item in first:
            kg = item[0]
            a, node_features = self.levi_graph(kg)
            node_feature_vectors = self.features_to_vectors(node_features)
            output.append(spektral.data.Graph(a=a, x=node_feature_vectors))
        return output

    #Create Levi graphs from existing edge labelled graphs
    def levi_graph(self, edge_list):
        nodes = set()
        for triple in edge_list:
            nodes.add(triple["head"])
            nodes.add(triple["tail"])

        # Create a mapping from nodes to indices
        node_to_index = {node: i for i, node in enumerate(nodes)}

        # Initialize a sparse matrix
        num_nodes = len(nodes)
        adjacency_matrix = dok_matrix((num_nodes, num_nodes), dtype=np.int8)
        node_features = []
        for node, i in node_to_index.items():
            node_features.append([i, node])
        edge_features_dict = dict()

        # Populate the adjacency matrix
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

        # Convert to a compressed sparse row (CSR) matrix
        levi_adjacency_matrix_csr = levi_adjacency_matrix.tocsr()
        print("\n")
        print(edge_list)
        print("\n")
        print(node_features)
        print("\n")
        print(levi_adjacency_matrix_csr)
        print("\n")
        print("====================")
        return levi_adjacency_matrix_csr, node_features
    
    def features_to_vectors(self, node_features):
        embedding_size = 50
        averaged_embeddings = []

        for node in node_features:
            feature_words = node[1].split()
            node_embedding = np.zeros(embedding_size)
            valid_words_count = 0
            for word in feature_words:
                embedding_vector = embeddings_dictionary.get(word)
                if embedding_vector is not None:
                    node_embedding += embedding_vector
                    valid_words_count += 1
            if valid_words_count > 0:
                node_embedding /= valid_words_count

            averaged_embeddings.append(node_embedding)

        averaged_embeddings = np.array(averaged_embeddings)
        return averaged_embeddings

    
dataset = MyDataset()
print(dataset[-1])


#===========================================================================================================================================================
#Define Encoder
#===========================================================================================================================================================
class Encoder_GGNN(Model):
    def __init__(self, n_layers):
        super().__init__()
        self.gated_graph_conv = GatedGraphConv(channels=50, n_layers=n_layers, name='encoder', )
        
    def call(self, inputs):
        out = self.gated_graph_conv(inputs)
        return out
    
#===========================================================================================================================================================
#Define Decoder
#===========================================================================================================================================================
class Decoder(Model):
    def __init__(self):
        super().__init__()
        self.decoder = GRU(50, return_sequences=True, return_state=True)
        self.decoder_dense = TimeDistributed(Dense(vocab_target_size, activation='softmax'))
    
    def call(self, input, hidden_state):
        out, hidden_state = self.decoder(input, initial_state=hidden_state)
        out = self.decoder_dense(out)
        return out, hidden_state
    
#===========================================================================================================================================================
#Define Model
#===========================================================================================================================================================
class ModelGGNN(Model):
    def __init__(self, n_layers):
        super().__init__()
        self.encoder = Encoder_GGNN(n_layers)
        self.max_pool = GlobalMaxPool()
        #Decoder here, GRU for now but will be LSTM
        self.decoder = GRU()
        
    def call(self, inputs):
        intermediate_representation= self.encoder(inputs)
        intermediate_representation = self.max_pool(out)
        out = self.decoder(intermediate_representation)
        return out
    
#===========================================================================================================================================================
#Set up training
#===========================================================================================================================================================

loader = DisjointLoader(dataset, batch_size=1)
batch = loader.__next__()
print(batch)
# inputs, target = batch
x, a, i = batch
print("X")
print(x)
print("A")
print(a)