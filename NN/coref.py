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

fwrite = open("corefedsquad.txt", "w", encoding="utf-8")
cnt = 0

for i in data['data']:
    if i['title'] == "Normans":
        for j in i['paragraphs']:
            to_write = coref(j['context'])._.resolved_text
            fwrite.write(to_write)
            fwrite.write("\n")
            cnt = cnt + 1
            if cnt > 10:
                break



