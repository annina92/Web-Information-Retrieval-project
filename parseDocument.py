import nltk
import spacy
import numpy
import time
import json
import os
import re
import wikipedia
wikipedia.set_lang("en")
spacy_nlp = spacy.load('en_core_web_sm')

from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import MWETokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer 
ps = PorterStemmer() 
from nltk.corpus import words

wordList = words.words()

lemmatizer = WordNetLemmatizer() 


cat = [
    ['alt.atheism'],
    ['comp.graphics'],
    ['comp.os.ms-windows.misc'],
    ['comp.sys.ibm.pc.hardware'],
    ['comp.sys.mac.hardware'],
    ['comp.windows.x'],
    ['misc.forsale'],
    ['rec.autos'],
    ['rec.motorcycles'],
    ['rec.sport.baseball'],
    ['rec.sport.hockey'],
    ['sci.crypt'],
    ['sci.electronics'],
    ['sci.med'],
    ['sci.space'],
    ['soc.religion.christian'],
    ['talk.politics.guns'],
    ['talk.politics.mideast'],
    ['talk.politics.misc'],
    ['talk.religion.misc']
]

#clean dataset without useless elements
newsgroup_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))

dataset = []
topics =[]

nDocs = 2000

for i in range (nDocs):
    dataset.append(newsgroup_train.data[i])
    topics.append(cat[newsgroup_train.target[i]])

topics_dump = json.dumps(topics)
f = open("topics.json", "w+")
f.write(topics_dump)
f.close  

list_of_stemmed_docs=[]
list_of_tokenized_docs_for_wiki=[]
dictionary_word_to_stem = {}
stemmed_vocabulary = []

for i in range(nDocs):
    print(i)
    relevant_words = []
    relevant_words_broken = []
    document_plain = dataset[i]

    document_ner = spacy_nlp(document_plain)

    for element in document_ner.ents:
        # don't consider numbers
        if element.label_ not in "CARDINAL":
            relevant_words.append(element)

    mwetokenizer = MWETokenizer()

    #ogni frase rilevante va spezzata e messa in un elenco
    for word in relevant_words:
        token = str(word).split()
        move_data=[]
        for element in token:
            move_data.append(element)
        tup = tuple(move_data)
        mwetokenizer.add_mwe(tup)

    document_tokenized = word_tokenize(document_plain)
    document_retokenized = mwetokenizer.tokenize(document_tokenized)

    #document retokenized is the document to work on

    #if word contains _ it is a special word, no further job has to be done on it
    document_tagged = nltk.pos_tag(document_retokenized)

    document_tagged = [(x,y) for (x,y) in document_tagged if (y in ('VB', 'NN', 'NNS', 'NNP', 'NNPS','VBD', "VBG", "VBN", "VBP", "VBZ"))or ("_" in x) ]


    pattern = re.compile(".*/")
    pattern2 = re.compile(".*\.")
    pattern3 = re.compile(".*\*") 
    pattern4 =re.compile(".*\\\\")
    pattern5 =re.compile(".*=")
    pattern6 =re.compile(".*\+")
    pattern7 =re.compile(".*,")

    pattern8 = re.compile("([A-Za-z0-9]*_)+[A-Za-z0-9]*$")


    document_stemmed_tokens =[]
    document_tokens_for_wiki = []
    document_lemmatized_count={}

    for word, tag in document_tagged:
        if "_" not in word:
            word = word.lower()

            if word in ("be","have","can","must","do"):
                continue
            if len(word)<2:
                continue
            if pattern.match(str(word)) or pattern2.match(str(word)) or pattern3.match(str(word)) :
                continue
            if pattern4.match(str(word)) or pattern5.match(str(word)) or pattern6.match(str(word)) or pattern7.match(str(word)):
                continue
            if (word.startswith("\\") or word.startswith("°") or word.startswith(".") or word.startswith(",") or word.startswith("<") or word.startswith(">") or word.startswith("-") or word.startswith("\'") or word.startswith("/") or word.startswith("#")  or word.startswith("\*") or word.startswith("\+") or word.startswith("_")  or word.startswith("=") or word.startswith("[") or word.startswith("|") or word.startswith("~") or word.startswith("]") or word.startswith("^") or word.startswith(":")):
                continue
            #for nouns and verb, save lemmatized and stemmed word, and create dictionary between the two forms
            #use lemmatized word to create the new dataset with correct tokens
            if tag.startswith("NN"):
                lemmatized_word = lemmatizer.lemmatize(word, pos="n")
                dictionary_word_to_stem[lemmatized_word] = ps.stem(lemmatized_word)

                document_stemmed_tokens.append(ps.stem(lemmatized_word))
                document_tokens_for_wiki.append(lemmatized_word)
                
                if ps.stem(lemmatized_word) not in stemmed_vocabulary:
                    stemmed_vocabulary.append(ps.stem(lemmatized_word))

                if lemmatized_word in document_lemmatized_count :
                    document_lemmatized_count[lemmatized_word]+=1
                if lemmatized_word not in document_lemmatized_count:
                    document_lemmatized_count[lemmatized_word]=1



            if tag.startswith("VB"):
                lemmatized_word = lemmatizer.lemmatize(word, pos="v")
                dictionary_word_to_stem[lemmatized_word] = ps.stem(lemmatized_word)
                document_stemmed_tokens.append(ps.stem(lemmatized_word))
                document_tokens_for_wiki.append(lemmatized_word)

                if ps.stem(lemmatized_word) not in stemmed_vocabulary:
                    stemmed_vocabulary.append(ps.stem(lemmatized_word))

                if lemmatized_word in document_lemmatized_count:
                    document_lemmatized_count[lemmatized_word]+=1
                if lemmatized_word not in document_lemmatized_count:
                    document_lemmatized_count[lemmatized_word]=1    
        


        #if token with multiple word, do as above    
        if "_" in word:
            if (word.startswith("\\") or word.startswith("°") or word.startswith(".") or word.startswith(",") or word.startswith("<") or word.startswith(">") or word.startswith("-") or word.startswith("\'") or word.startswith("/") or word.startswith("#")  or word.startswith("\*") or word.startswith("\+") or word.startswith("_")  or word.startswith("=") or word.startswith("[") or word.startswith("|") or word.startswith("~") or word.startswith("]") or word.startswith("^") or word.startswith(":")):
                continue
            word = word.replace("/", "_")
            if "#" in word or "^" in word or "+" in word or "\\" in word or "__" in word or "=" in word or "*" in word:
                continue
            if len(word)<5:
                continue

            document_stemmed_tokens.append(word)
            document_tokens_for_wiki.append(word)

            if word in document_lemmatized_count:
                document_lemmatized_count[word]+1
            if word not in document_lemmatized_count:
                document_lemmatized_count[word] =1
            if word not in stemmed_vocabulary:
                stemmed_vocabulary.append(word)
                

    #create 2 dataset with stemmed words (for standard tfidf) and lemmatized words (for wikipedia query)
    list_of_stemmed_docs.append(document_stemmed_tokens)
    list_of_tokenized_docs_for_wiki.append(document_tokens_for_wiki)



#save data to files
f = open("list_of_stemmed_docs.json", "w+")
json_data = json.dumps(list_of_stemmed_docs)
f.write(json_data)
f.close()

f = open("list_of_tokenized_docs_for_wiki.json", "w+")
json_data = json.dumps(list_of_tokenized_docs_for_wiki)
f.write(json_data)
f.close()

f = open("dictionary_word_to_stem.json", "w+")
json_data = json.dumps(dictionary_word_to_stem)
f.write(json_data)
f.close()


f = open("stemmed_vocabulary.json", "w+")
json_data = json.dumps(stemmed_vocabulary)
f.write(json_data)
f.close()


