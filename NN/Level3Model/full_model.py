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
import os

folder_path = '/Users/ethanwakefield/Documents/3rdYearProject/3rd-Year-Project/NN/dataset2'
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
        all_output = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                output = self.read_one(file_path)
                all_output.extend(output)
        return all_output

    
    def read_one(self, file_path):
        output = []
        f = open(file_path)
        data = json.load(f)
        #Need to change this - for every question, get it's corresponding answer. Highlight that node's vector. Set target to question + answer
        for item in data:
            kg = item[0]
            questions = item[1]
            if (len(questions) == 0):
                continue
            a, node_features = self.levi_graph(kg)
            answers = item[2]
            for i in range(len(questions)):
                question = questions[i]
                answer = answers[i]
                question = self.clean(question)
                question = 'sostok ' + question + ' endtok'
                question = self.quest_tokenizer.texts_to_sequences([question])
                question = pad_sequences(question,  maxlen=20, padding='post')[0]
                node_feature_vectors = self.features_to_vectors(node_features, answer)
                output.append(spektral.data.Graph(a=a, x=node_feature_vectors, y=question))
        f.close()
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
    
    def features_to_vectors(self, node_features, answer):
        embedding_size = 600
        averaged_embeddings = []

        for node in node_features:
            highlight = False
            sent = node[1]
            if sent == answer:
                highlight = True

            sent = self.clean(sent)
            feature_words = sent.split()
            node_embedding = np.zeros(embedding_size)
            valid_words_count = 0

            for i, word in enumerate(feature_words):
                embedding_vector = self.embeddings_dictionary.get(word)
                if embedding_vector is not None:
                    node_embedding[:300] += embedding_vector[:300]  # First 50 dimensions
                    valid_words_count += 1

            if valid_words_count > 0:
                node_embedding[:300] /= valid_words_count

            # Set the last 50 dimensions
            if highlight:
                node_embedding[300:] = node_embedding[:300]
            else:
                node_embedding[300:] = 0

            averaged_embeddings.append(node_embedding)

        averaged_embeddings = np.array(averaged_embeddings)
        return averaged_embeddings

#===========================================================================================================================================================
#Define Encoder
#===========================================================================================================================================================
class Encoder_GGNN(Model):
    def __init__(self, n_layers):
        super().__init__()
        self.gated_graph_conv = GatedGraphConv(channels=600, n_layers=n_layers, name='encoder')
        
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
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
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
    def __init__(self, encoder, decoder, optimizer, quest_tokenizer):
        super().__init__()
        self.encoder = encoder
        self.max_pool = GlobalMaxPool()
        self.decoder = decoder
        self.optimizer = optimizer
        self.quest_tokenizer = quest_tokenizer
        self.reverse_target_word_index = quest_tokenizer.index_word


    def call(self, graph_input):
        intermediate_representation = self.encoder(graph_input)
        intermediate_representation = self.max_pool(intermediate_representation)
        decoder_input = tf.expand_dims([self.quest_tokenizer.word_index['sostok']], 1)
        state = intermediate_representation
        output = ''

        for i in range(30):
            prediction, state = self.decoder(decoder_input, state)
            predicted_token = tf.argmax(prediction[0])
            current_word = self.reverse_target_word_index[predicted_token.numpy()]
            if current_word == 'endtok':
                return output
            
            output = output + current_word + ' '
            decoder_input = tf.expand_dims([predicted_token], 0)


    def inference_pre_graph(self):
        pass


    def inference_post_graph(self, graph_input):
        pass
    

    def train_step(self, encoder_input, target, loss_object):
        with tf.GradientTape() as tape:
            intermediate_representation= self.encoder(encoder_input)
            intermediate_representation = self.max_pool(intermediate_representation)
            decoder_input = tf.expand_dims([self.quest_tokenizer.word_index['sostok']], 1)
            state = intermediate_representation
            loss = 0

            for i in range(1, len(target[0])):
                prediction, state = self.decoder(decoder_input, state)
                loss += loss_object.loss_function(target[:, i], prediction)
                decoder_input = tf.expand_dims(target[:, i], 1)

            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            loss = (loss / int(len(target[0])))
            return loss, prediction
    
    def train(self, loader, loss_object):
                
        cnt = 1
        total_loss = 0


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
            model_loss, model_output = self.train_step(encoder_input, decoder_target, loss_object)
            total_loss += model_loss
            print(f"Iter: {cnt}   ", model_loss)
            # print("=====================================================================================================")
            

            # train_step((encoder_input, decoder_input), decoder_target)
            # print(cnt)
            cnt += 1

            if cnt == 2030:
                with open('/Users/ethanwakefield/Documents/3rdYearProject/3rd-Year-Project/NN/Level3Model/graph_600_1/9epochs/loss.txt', 'w') as f:
                    f.write(f"Total Loss: {total_loss}\n")
                    f.write(f"Average Loss: {total_loss/2030}\n")
                cnt = 1
                total_loss = 0
                tf.keras.backend.clear_session()
            
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

            
            