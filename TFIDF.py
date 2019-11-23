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
from sklearn.cluster import AgglomerativeClustering
import string 
import urllib



wiki_term_article_dict = {}
#retrieve mapping from lemmatized word to wiki article titles
with open("wiki_term_article_dict.json", 'r+') as myfile:
    data=myfile.read()
wiki_term_article_dict = json.loads(data)


article_categories_dict={}
#contains article with each category
with open("article_categories_dict.json", 'r+') as myfile:
    data=myfile.read()
article_categories_dict = json.loads(data)



def retrieve_category_article_dict(article_categories_dict):
    category_articles_dict ={}
    #retrieve categorie-articles dictionary
    for word in article_categories_dict:
        categories = article_categories_dict[word]
        for category in categories:
            if category in category_articles_dict:
                category_articles_dict[category].append(word)

            if category not in category_articles_dict:
                category_articles_dict[category] = []
                category_articles_dict[category].append(word)

    if not os.path.exists("category_articles_dics.json"):
        category_articles_dict_dump = json.dumps(category_articles_dict)
        f = open("category_articles_dics.json", "w+")
        f.write(category_articles_dict_dump)
        f.close

    return category_articles_dict



def hello(world):
    print(world)



def dummy(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy,
    preprocessor=dummy,
    token_pattern=None,
    use_idf=True,
    sublinear_tf =True)  


def tfIDF(list_docs):

    tfidf_vectors = tfidf.fit_transform(list_docs)
    tfidf_feature_names = tfidf.get_feature_names()
    df = pd.DataFrame(tfidf_vectors.toarray(), columns= tfidf_feature_names)
    #print(df)
    return tfidf_vectors


def cos_sim(list_docs):
    tfidf_vectors = tfidf.fit_transform(list_docs)

    tfidf_feature_names = tfidf.get_feature_names()
    df = pd.DataFrame(tfidf_vectors.toarray(), columns= tfidf_feature_names)
    #print(df)

    return sklearn.metrics.pairwise.cosine_similarity(tfidf_vectors, Y=None, dense_output=True)


def cos_distance(list_docs):
    tfidf_vectors = tfidf.fit_transform(list_docs)

    tfidf_feature_names = tfidf.get_feature_names()
    df = pd.DataFrame(tfidf_vectors.toarray(), columns= tfidf_feature_names)
    return sklearn.metrics.pairwise.cosine_distances(tfidf_vectors, Y=None)


def preparePlainDocs(list_docs):

    list_of_stemmed_docs =[]

    for document in list_docs:
        newDocument =[]
        for term in document:
            if not (term.startswith("\\") or term.startswith("<") or term.startswith(">") or term.startswith("-") or term.startswith("\'") or term.startswith("/") or term.startswith("&") or term.startswith("\*") or term.startswith("#")  or term.startswith("\*") or term.startswith("\+") or term.startswith("_")  or term.startswith("=") or term.startswith("[") or term.startswith("|") or term.startswith("~") or term.startswith("]") or term.startswith("^") or term.startswith(":")):
                newDocument.append(term)
        list_of_stemmed_docs.append(newDocument)

    return list_of_stemmed_docs


def preparedocConcDocs(list_docs):

    list_docs_documentConcept =[]
    # scan each article: if its word belong to wiki dictionary, substitute words with wiki title, else remove the word
    for document in list_docs:
        newDocument = []
        for word in document:
            if word in wiki_term_article_dict:
                newDocument.append(wiki_term_article_dict[word])
        #update list of docs
        list_docs_documentConcept.append(newDocument)

    if not os.path.exists("list_docs_wiki_articles.json"):
        list_docs_wiki_articles_dump = json.dumps(list_docs_documentConcept)
        f = open("list_docs_wiki_articles.json", "w+")
        f.write(list_docs_wiki_articles_dump)
        f.close


    return list_docs_documentConcept


def preparedocCatDocs(list_docs):
    list_docs_documentCategory = []


    category_articles_dict = retrieve_category_article_dict(article_categories_dict)

    for document in list_docs:
        newDocument =[]
        for word in document:
            #for each article, replace it with the corresponding categories
            #no need to count and group same categories since sklearn function takes as input a "raw" file and do the count to produce tfidf matrix
            categories = article_categories_dict[word]
            count =0
            if len(categories) ==0:
                continue
            if len(categories) ==1:
                newDocument.append(categories[0])
                count = count+1
            if len(categories)>1:
                for i in range(len(categories)):
                    category = categories[i]
                    if len(category_articles_dict[category])>1:
                        newDocument.append(category)
                        count=count+1
            if count==0:

                newDocument.append(categories[0])

        list_docs_documentCategory.append(newDocument)


    if not os.path.exists("list_docs_categories.json"):
        list_docs_categories_dump = json.dumps(list_docs_documentCategory)
        f = open("list_docs_categories.json", "w+")
        f.write(list_docs_categories_dump)
        f.close        

    return list_docs_documentCategory
   
#for agglomerative
def sum_cos_distance(plain_Matrix, doc_concept_matrix, doc_category_matrix, alpha, beta, gamma):
    #sim(d1,d2) = sim(d1,d2)(word) + alpha*sim(d1,d2)(concept) + beta*sim(d1,d2)(category)
    sim_matrix = gamma*plain_Matrix +doc_concept_matrix*alpha + doc_category_matrix*beta

    return sim_matrix

