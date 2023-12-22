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
from keras.layers import Embedding, LSTM, Flatten, Dense , Dropout 
import spektral
from spektral.layers import GCNConv, GlobalSumPool
from keras.models import Model

#If we have a finite number of relstions, we can one hot encode them. 
#OR (bi)LSTM the nodes & edges to get our fixed size vector representation a-la Subgraph... paper.

f = open('C:/3rdYearProject/3rd-Year-Project/NN/dataset/Level3CQA.json')
data = json.load(f)

# class MyFirstGNN(Model):

#     def __init__(self, n_hidden, n_labels):
#         super().__init__()
#         self.graph_conv = GCNConv(n_hidden)
#         self.pool = GlobalSumPool()
#         self.dropout = Dropout(0.5)
#         self.dense = Dense(n_labels, 'softmax')

#     def call(self, inputs):
#         out = self.graph_conv(inputs)
#         out = self.dropout(out)
#         out = self.pool(out)
#         out = self.dense(out)

#         return out

def my_graph(edge_list):
    nodes = set()
    for triple in edge_list:
        nodes.add(triple["head"])
        nodes.add(triple["tail"])
        

    # Create a mapping from nodes to indices
    node_to_index = {node: i for i, node in enumerate(nodes)}

    # Initialize a sparse matrix
    num_nodes = len(nodes)
    adjacency_matrix = dok_matrix((num_nodes, num_nodes), dtype=np.int8)

    # Populate the adjacency matrix
    num_edges = len(edge_list)
    for triple in edge_list:
        head_index = node_to_index[triple["head"]]
        tail_index = node_to_index[triple["tail"]]
        adjacency_matrix[head_index, tail_index] = 1

    #Convert to Levi graph
    incrementer = num_nodes
    levi_adjacency_matrix = dok_matrix((num_nodes + num_edges, num_nodes + num_edges), dtype=np.int8)
    for i in range(0, num_nodes):
        for j in range(0, num_nodes):
            if adjacency_matrix[i,j] == 1:
                levi_adjacency_matrix[i,incrementer] = 1
                levi_adjacency_matrix[incrementer,j] = 1
                incrementer = incrementer + 1


    # Convert to a compressed sparse row (CSR) matrix
    print("OG")
    adjacency_matrix_csr = adjacency_matrix.tocsr()
    print(adjacency_matrix_csr)
    print("\n")
    print("LEVI")
    levi_adjacency_matrix_csr = levi_adjacency_matrix.tocsr()
    print(levi_adjacency_matrix_csr)
    return adjacency_matrix_csr


first = data["Super_Bowl_50"]
for item in first:
    kg = item[0]
    graph = spektral.data.Graph(a=my_graph(kg))
    #print(graph.n_nodes)
    #print(graph.n_edges)

