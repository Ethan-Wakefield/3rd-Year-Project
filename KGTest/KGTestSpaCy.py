import spacy
import crosslingual_coreference

import requests
import re
import hashlib
from spacy import Language
from typing import List

from collections import defaultdict

from spacy.tokens import Doc, Span

from transformers import pipeline
from transformers import logging
logging.set_verbosity_error()


#Take output from REBEL model and parse it, to create a list of dictionaries with fields "head", "type" (relation) and "tail". 
def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets


#REBEL model, to be used as a SpaCy pipeline component
@Language.factory(
    "rebel",
    requires=["doc.sents"],
    assigns=["doc._.rel"],
    default_config={
        "model_name": "Babelscape/rebel-large",
        "device": -1,
    },
)
class RebelComponent:
    def __init__(
        self,
        nlp,
        name,
        model_name: str,
        device: int,
    ):
        assert model_name is not None, ""
        self.triplet_extractor = pipeline("text2text-generation", model=model_name, tokenizer=model_name, device=device)
        self.entity_mapping = {}
        # Register custom extension on the Doc
        if not Doc.has_extension("rel"):
          Doc.set_extension("rel", default={})
    
    def _generate_triplets(self, sent: Span) -> List[dict]:
          output_ids = self.triplet_extractor(sent.text, return_tensors=True, return_text=False)[0]["generated_token_ids"]["output_ids"]
          extracted_text = self.triplet_extractor.tokenizer.batch_decode(output_ids[0])
          extracted_triplets = extract_triplets(extracted_text[0])
          return extracted_triplets

    def set_annotations(self, doc: Doc, triplets: List[dict]):
        for triplet in triplets:
            if triplet['head'] != triplet['tail']:
                definer = triplet['head'] + triplet['type'] + triplet['tail']
                doc._.rel[hash(definer)] = {"head": triplet['head'], "relation": triplet["type"], "tail": triplet['tail']}

    def __call__(self, doc: Doc) -> Doc:
        for sent in doc.sents:
            sentence_triplets = self._generate_triplets(sent)
            self.set_annotations(doc, sentence_triplets)
        return doc
    

#Form graph out of RDF triples developed by the model
class DiGraph():
    def __init__(self):
        self.adjList = defaultdict(list)

    def addRelation(self, rel_dict):
        source = rel_dict['head']
        
        relation = rel_dict['relation']
       
        sink = rel_dict['tail']
        self.adjList[source].append((relation, sink))
        
        


DEVICE = -1 # Number of the GPU, -1 if want to use CPU

# Add coreference resolution model
coref = spacy.load("en_core_web_sm", disable=['ner', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
coref.add_pipe("xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": DEVICE})
# Define rel extraction model
rel_ext = spacy.load("en_core_web_sm", disable=['ner', 'lemmatizer', 'attribute_rules', 'tagger'])
rel_ext.add_pipe("rebel", config={'device':DEVICE, 'model_name':'Babelscape/rebel-large'})

#Test
input_text = "The plague disease, caused by Yersinia pestis, is enzootic (commonly present) in populations of fleas carried by ground rodents, including marmots, in various areas including Central Asia, Kurdistan, Western Asia, Northern India and Uganda. Nestorian graves dating to 1338-39 near Lake Issyk Kul in Kyrgyzstan have inscriptions referring to plague and are thought by many epidemiologists to mark the outbreak of the epidemic, from which it could easily have spread to China and India. In October 2010, medical geneticists suggested that all three of the great waves of the plague originated in China. In China, the 13th century Mongol conquest caused a decline in farming and trading. However, economic recovery had been observed at the beginning of the 14th century. In the 1330s a large number of natural disasters and plagues led to widespread famine, starting in 1331, with a deadly plague arriving soon after. Epidemics that may have included plague killed an estimated 25 million Chinese and other Asians during the 15 years before it reached Constantinople in 1347."

coref_text = coref(input_text)._.resolved_text
print(coref_text)
doc = rel_ext(coref_text)

graph = DiGraph()
for value, rel_dict in doc._.rel.items():
    print(f"{value}: {rel_dict}")
    graph.addRelation(rel_dict)

print("\n")
print("ADJLIST:")
print(graph.adjList.items())

