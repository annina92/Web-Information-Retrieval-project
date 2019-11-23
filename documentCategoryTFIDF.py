import time
import json
import os
import numpy as np
import pandas as pd
import re
import sklearn
import string 
import urllib

from nltk import wordpunct_tokenize, WordNetLemmatizer, sent_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


#vocabulary determined by wikipedia article

article_categories_dict={}
list_of_documents = []

list_docs_documentCategories = []
list_docs_documentCategoriesTEST=[]

#contains article with each category
with open("article_categories_dict"+str(2)+".json", 'r+') as myfile:
    data=myfile.read()
article_categories_dict = json.loads(data)



#retrieve documents with wikipedia titles rather than tokens
with open('list_docs_documentConcept.json', 'r') as myfile:
    data=myfile.read()
obj = json.loads(data)
for element in obj:
    list_of_documents.append(element)


category_articles_dict={}

#retrieve categorie-articles dictionary
for word in article_categories_dict:
    categories = article_categories_dict[word]
    for category in categories:
        if category in category_articles_dict:
            category_articles_dict[category].append(word)

        if category not in category_articles_dict:
            category_articles_dict[category] = []
            category_articles_dict[category].append(word)



for document in list_of_documents:
    newDocument =[]
    for word in document:
        #for each article, replace it with the corresponding categories
        #no need to count and group same categories since sklearn function takes as input a "raw" file and do the count to produce tfidf matrix
        categories = article_categories_dict[word]

        #remove article name
        for category in categories :
            newDocument.append(category)
    
    list_docs_documentCategories.append(newDocument)
    break


for document in list_of_documents:
    newDocument =[]
    for word in document:
        #for each article, replace it with the corresponding categories
        #no need to count and group same categories since sklearn function takes as input a "raw" file and do the count to produce tfidf matrix
        categories = article_categories_dict[word]
        if len(categories) ==1:
            newDocument.append(categories[0])

        if len(categories)>1:
            for category in categories:
                if len(category_articles_dict[category])>1:
                    newDocument.append(category)

    
    list_docs_documentCategoriesTEST.append(newDocument)
     
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
tfidf_doc_category_vectors = tfidf.fit_transform(list_docs_documentCategoriesTEST)


tfidf_feature_names = tfidf.get_feature_names()
df = pd.DataFrame(tfidf_doc_category_vectors.toarray(), columns= tfidf_feature_names)
print(df)

print(sklearn.metrics.pairwise.cosine_similarity(tfidf_doc_category_vectors, Y=None, dense_output=True))



