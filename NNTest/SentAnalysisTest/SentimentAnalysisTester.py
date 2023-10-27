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


with open('NNTest/SentAnalysisTest/tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

loaded_model = tf.keras.models.load_model("NNTest/SentAnalysisTest/model.keras", compile = False)
sentence = "This film was very interesting."
instance = tokenizer.texts_to_sequences(sentence)

flat_list = []

for sublist in instance :
    for item in sublist :
        flat_list.append(item)
        
flat_list = [flat_list]
print(flat_list)

flat_list = pad_sequences(flat_list, padding = 'post', maxlen = 100)



if loaded_model.predict(flat_list) > 0.5 :
    
    print("POSITIVE")
    
else :
    print("NEGATIVE")
