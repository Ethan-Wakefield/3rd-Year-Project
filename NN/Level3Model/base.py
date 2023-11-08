import re                                   # 're' Replication of text.
import numpy as np     
import pickle                     
import pandas as pd                         # 'pandas' to manipulate the dataset.
import tensorflow as tf
from keras.models import Sequential                # 'Sequential' model will be used for training.
from sklearn.model_selection import train_test_split          # 'train_test_split' for splitting the data into train and test data. 
from keras.preprocessing.text import Tokenizer       
from keras.preprocessing.sequence import pad_sequences       # 'pad_sequences' for having same dimmension for each sequence.
from keras.layers import Embedding, LSTM, Flatten, Dense  

#Form graph out of RDF triples developed by the model
class di_graph():
    def __init__(self):
        self.adjList = defaultdict(list)

    def add_relation(self, rel_dict):
        source = rel_dict['head']
        
        relation = rel_dict['relation']
       
        sink = rel_dict['tail']
        self.adjList[source].append((relation, sink))
        
class levi_graph():
    def __init__(self):
        self.adjList = defaultdict(list)

    def convert_to_levi_graph(self, digraph):
        return None
        #convert to levi graph

print("bleh")