import urllib.request
from bs4 import BeautifulSoup
import spacy
import neuralcoref

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)
doc1 = nlp('The town has a hill to the left of it. There is a valley to its right. Lucy lives in the valley, but she does not like it.')
print(doc1._.coref_resolved)

#install spacy 2.3.5
#install python 3.7