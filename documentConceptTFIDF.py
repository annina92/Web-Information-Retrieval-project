from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

import time
import json
import os
import sklearn

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import string 
import urllib

#vocabulary determined by wikipedia article

article_categories_dict={}
wiki_term_article_dict ={}
list_of_documents = []

category_articles_dict={}


#contains article with each category
with open("article_categories_dict"+str(2)+".json", 'r+') as myfile:
    data=myfile.read()
article_categories_dict = json.loads(data)


#contains wikipedia article titles
wikipedia_doctionary = list(article_categories_dict.keys())


#retrieve mapping from lemmatized word to wiki article titles
with open("wiki_term_article_dict"+str(2)+".json", 'r+') as myfile:
    data=myfile.read()
wiki_term_article_dict = json.loads(data)


#retrieve documents lemmatized tokenized
# read file
with open('list_of_tokenized_docs_for_wiki.json', 'r') as myfile:
    data=myfile.read()
obj = json.loads(data)

for element in obj:
    list_of_documents.append(element)

list_docs_documentConcept=[]

# scan each article: if its word belong to wiki dictionary, substitute words with wiki title, else remove the word
for i in range(0,1000):

    document = list_of_documents[i]

    newDocument = []
    for word in document:
        if word in wiki_term_article_dict:
            newDocument.append(wiki_term_article_dict[word])

    #update list of docs
    list_of_documents[i] = newDocument
    list_docs_documentConcept.append(newDocument)



#retrieve categorie-articles dictionary
for word in article_categories_dict:
    categories = article_categories_dict[word]
    for category in categories:
        if category in category_articles_dict:
            category_articles_dict[category].append(word)

        if category not in category_articles_dict:
            category_articles_dict[category] = []
            category_articles_dict[category].append(word)

print (category_articles_dict)
    

def dummy(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None)  

cv = CountVectorizer(
    tokenizer=dummy,
    preprocessor=dummy,
)  

#con fit_transform ho matrice sparsa e la posso stampare
tfidf_vectors = tfidf.fit_transform(list_docs_documentConcept)


tfidf_feature_names = tfidf.get_feature_names()
df = pd.DataFrame(tfidf_vectors.toarray(), columns= tfidf_feature_names)
print(df)

print(sklearn.metrics.pairwise.cosine_similarity(tfidf_vectors, Y=None, dense_output=True))



category_articles_dict_dump = json.dumps(category_articles_dict)
f = open("category_articles_dict.json", "w+")
f.write(category_articles_dict_dump)
f.close

list_docs_documentConcept_dump = json.dumps(list_docs_documentConcept)
f = open("list_docs_documentConcept.json", "w+")
f.write(list_docs_documentConcept_dump)
f.close