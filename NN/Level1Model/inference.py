import json 
import re                                   # 're' Replication of text.
import numpy as np     
import pickle                     
import pandas as pd                         # 'pandas' to manipulate the dataset.
import tensorflow as tf
from keras.models import Sequential                # 'Sequential' model will be used for training.
from sklearn.model_selection import train_test_split          # 'train_test_split' for splitting the data into train and test data. 
from keras.preprocessing.text import Tokenizer       
from keras.preprocessing.sequence import pad_sequences       # 'pad_sequences' for having same dimmension for each sequence.
from keras.layers import Embedding, LSTM, Flatten, Dense, Bidirectional, Input, TimeDistributed
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.utils import plot_model


model = tf.keras.models.load_model("C:/3rdYearProject/3rd-Year-Project/NN/Level1Model/model.keras", compile = True)
print(model.summary())


index = None
for idx, layer in enumerate(model.layers):
    print(idx)
    print(layer)
    print("\n")

plot_model(model, to_file='modelsummary.png', show_shapes=True, show_layer_names=True)

'''
encoder_inputs = model.input[0] 
encoder_outputs, state_h, state_c = model.layers[2].output 


encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs,
                      state_h, state_c])
'''