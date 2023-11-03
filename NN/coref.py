import json

import spacy
import crosslingual_coreference


from transformers import logging
logging.set_verbosity_error()


f = open('C:/3rdYearProject/3rd-Year-Project/NN/dataset/dev-v2.0.json')
 
# returns JSON object as 
# a dictionary
data = json.load(f)

coref = spacy.load("en_core_web_sm", disable=['ner', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
coref.add_pipe("xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": -1})

for i in data['data']:
    for j in i['paragraphs']:
        print(coref(j['context'])._.resolved_text)
        print("\n")

