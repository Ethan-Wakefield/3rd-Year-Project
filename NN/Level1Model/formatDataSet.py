import json
import nltk
import re


nltk.download('punkt')
f = open('C:/3rdYearProject/3rd-Year-Project/NN/dataset/SQuAD-v1.1.json')
data = json.load(f)

#===========================================================================================================================================================
#Remove things like multiple spaces and particular unicode characters
#===========================================================================================================================================================
def normalize_text(text):
    sentence = text.lower()
    sentence = sentence.replace("\u2013", "-")
    sentence = sentence.replace("?", '')
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

#===========================================================================================================================================================
#Establish dict of 3 arrays. Indeces correspond.
#===========================================================================================================================================================
dictionary = {
    'sentences': [],
    'questions': [],
    'answers': []
}
cnt = 0
for i in data['data']:
    for j in i['paragraphs']:
        cnt = cnt+1
        context = j['context']
        sentences = nltk.sent_tokenize(context)
        for qa in j['qas']:
            ans_list = qa['answers']
            answer = ans_list[0]['text']
            answer_offset = ans_list[0]['answer_start']
            question= qa['question']
            #now find the sentence containing the answer in the context para
            for sentence in sentences:
                start_index = context.find(sentence)
                end_index = start_index + len(sentence)
                if start_index <= answer_offset < end_index:
                    dictionary['sentences'].append(normalize_text(sentence))
                    dictionary['questions'].append(normalize_text(question))
                    dictionary['answers'].append(normalize_text(answer))
    print(cnt)

#===========================================================================================================================================================
#Write JSON object to file
#===========================================================================================================================================================
with open('C:/3rdYearProject/3rd-Year-Project/NN/dataset/Level1CQA.json', "w") as outfile:
    json.dump(dictionary, outfile)