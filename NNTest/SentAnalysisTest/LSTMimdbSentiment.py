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

#Read in data set
movie_reviews = pd.read_csv(r"C:\3rdYearProject\3rd-Year-Project\NNTest\SentAnalysisTest\input\IMDB Dataset.csv", encoding = 'latin1')
movie_reviews.isnull().values.any()



#############################################
#CLEAN DATA
#############################################
def preprocess_text(sen):
    # Removing html tags
    sentence = re.sub(r'<[^>]+>', '', sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

cleaned = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    cleaned.append(preprocess_text(sen))

#Convert sentiment word labels to 0 or 1 values for negative / positive
sentiments = np.array(list(map(lambda x : 1 if x == 'positive' else 0, list(movie_reviews['sentiment']))))

#Split into training and test data
Sentence_train, Sentence_test, Label_train, Label_test = train_test_split(sentences, sentiments, test_size = 0.2, random_state = 42) 



##############################################
#TOKENIZE AND PAD
##############################################
tokenizer = Tokenizer(num_words = 5000)

tokenizer.fit_on_texts(Sentence_train)

with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

Sentence_train = tokenizer.texts_to_sequences(Sentence_train)
Sentence_test = tokenizer.texts_to_sequences(Sentence_test)

vocab_size = len(tokenizer.word_index) + 1

max_len = 100

Sentence_train = pad_sequences(Sentence_train, padding = 'post', maxlen = max_len)
Sentence_test = pad_sequences(Sentence_test , padding = 'post', maxlen = max_len)



###############################################
#PREPARE GLOVE
###############################################
embeddings_dictionary = dict()
glove_file = open('C:/3rdYearProject/3rd-Year-Project/NNTest/SentAnalysisTest/glove/glove.6B.50d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = np.zeros((vocab_size, 50))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector



################################################
#DEFINE MODEL
################################################
model = Sequential([
    Embedding(vocab_size, 50,  weights=[embedding_matrix], input_length=max_len, trainable=False),
    LSTM(128),
    Dense(2, activation = 'sigmoid')
])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


#################################################
#TRAIN
#################################################
history = model.fit(Sentence_train, Label_train, batch_size = 128, epochs = 6, validation_split = 0.20, verbose = 1)
score = model.evaluate(Sentence_test, Label_test, verbose = 1)

model.save("model.keras")