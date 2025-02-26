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

#split into independent clauses for more triples
input_text = "The secondary level includes schools offering years 7 through 12 (year twelve is known as lower sixth) and year 13 (upper sixth). This category includes university-preparatory schools or 'prep schools', boarding schools and day schools. Tuition at private secondary schools varies from school to school and depends on many factors, including the location of the school, the willingness of parents to pay, peer tuitions and the school's financial endowment. High tuition, schools claim, is used to pay higher salaries for the best teachers and also used to provide enriched learning environments, including a low student to teacher ratio, small class sizes and services, such as libraries, science laboratories and computers. Some private schools are boarding schools and many military academies are privately owned or operated as well."
input_text2 = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the 'golden anniversary' with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as 'Super Bowl L'), so that the logo could prominently feature the Arabic numerals 50."

coref_text = coref(input_text2)._.resolved_text
print(coref_text)
doc = rel_ext(coref_text)

graph = DiGraph()
for value, rel_dict in doc._.rel.items():
    print(f"{value}: {rel_dict}")
    graph.addRelation(rel_dict)

print("\n")
print("ADJLIST:")
print(graph.adjList.items())

