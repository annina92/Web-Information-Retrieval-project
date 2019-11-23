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

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import string 
import urllib
import TFIDF
import scores
import cluster
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform


#########################
#clean dataset without useless elements
newsgroup_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))

dataset = []

for i in range (1000):
    dataset.append(newsgroup_train.data[i])

#########################
#for plain cos sim
list_of_stemmed_docs=[]

# retrieve stemmed token
with open('list_of_stemmed_docs.json', 'r') as myfile:
    data=myfile.read()
obj = json.loads(data)
for element in obj:
    list_of_stemmed_docs.append(element)


#########################
#for documentConcept cos sim
list_of_lemmatized_docs=[]

#retrieve documents lemmatized tokenized
with open('list_of_tokenized_docs_for_wiki.json', 'r') as myfile:
    data=myfile.read()
obj = json.loads(data)

for element in obj:
    list_of_lemmatized_docs.append(element)


#########################
#for documentCategory cos sim
list_of_wikiArticles_docs=[]

#retrieve documents with wikipedia titles rather than tokens
with open('list_docs_documentConcept.json', 'r') as myfile:
    data=myfile.read()
obj = json.loads(data)
for element in obj:
    list_of_wikiArticles_docs.append(element)

#########################
#for clustering
documentTopics = []

with open('topics.json', 'r') as myfile:
    data=myfile.read()
obj = json.loads(data)
for element in obj:
    documentTopics.append(element)


clustering_schemas = [
    'word',
    'concept',
    'category',
    'word_concept',
    'word_category',
    'concept_category',
    'word_concept_category'
]


clustering_data_dict = {}

#dictionary that, for each topic, return the number of docs of that topic
elements_per_topic = {}

for element in documentTopics:
    element = element[0]
    if element in elements_per_topic:
        elements_per_topic[element]= elements_per_topic[element]+1
    if element not in elements_per_topic:
        elements_per_topic[element] =1


#dictionary that cointains, for each topic, the id of the document
document_list_for_each_topic = {}
for i in range(1000):
    topic =documentTopics[i][0]
    if topic in document_list_for_each_topic:
        document_list_for_each_topic[topic].append(i)
    if topic not in document_list_for_each_topic:
        document_list_for_each_topic[topic] = []
        document_list_for_each_topic[topic].append(i)


#prepare documents wit correct vocabulary
plainDocs = TFIDF.preparePlainDocs(list_of_stemmed_docs)
documentConceptDocs = TFIDF.preparedocConcDocs(list_of_lemmatized_docs)
documentCatDocs = TFIDF.preparedocCatDocs(TFIDF.preparedocConcDocs(list_of_lemmatized_docs))

#cosin similarity computation for the three matrices
plainDocsCosSim = TFIDF.cos_sim(plainDocs)
documentConceptCosSim = TFIDF.cos_sim(documentConceptDocs)
documentCategoryCosSim = TFIDF.cos_sim(documentCatDocs)

#tfidf matrices
plainDocsTFIDF = TFIDF.tfIDF(plainDocs)
documentConceptTFIDF = TFIDF.tfIDF(documentConceptDocs)
documentCategoryTFIDF = TFIDF.tfIDF(documentCatDocs)

#cosine distance for agglomerative clustering
plainDocsDistanceMatrix = TFIDF.cos_distance(plainDocs)
documentConceptDistanceMatrix = TFIDF.cos_distance(documentConceptDocs)
documentCategoryDistanceMatrix = TFIDF.cos_distance(documentCatDocs)


clustering_data_dict["word"] = AgglomerativeClustering(affinity='precomputed', n_clusters=20, linkage='complete').fit(plainDocsDistanceMatrix)
clustering_data_dict["concept"] = AgglomerativeClustering(affinity='precomputed', n_clusters=20, linkage='complete').fit(documentConceptDistanceMatrix)
clustering_data_dict["category"] = AgglomerativeClustering(affinity='precomputed', n_clusters=20, linkage='complete').fit(documentCategoryDistanceMatrix)


#for word concept
word_concept_cluster_list = []
alpha =0.1
beta =0
for i in range (10):
    similarityMatrix = TFIDF.sum_cos_distance(plainDocsDistanceMatrix, documentConceptDistanceMatrix, documentCategoryDistanceMatrix, alpha, beta,1 )
    word_concept_cluster_list.append(AgglomerativeClustering(affinity='precomputed', n_clusters=20, linkage='complete').fit(similarityMatrix))
    alpha = alpha +0.1
    

#for word category
word_category_cluster_list=[]
alpha =0
beta =0.1
for i in range (10):
    similarityMatrix = TFIDF.sum_cos_distance(plainDocsDistanceMatrix, documentConceptDistanceMatrix, documentCategoryDistanceMatrix, alpha, beta,1 )
    word_category_cluster_list.append(AgglomerativeClustering(affinity='precomputed', n_clusters=20, linkage='complete').fit(similarityMatrix))
    beta = beta +0.1


#computing scores
fscore_word_concept = []
purity_word_concept= []
NMI_word_concept = []

fscore_word_category = []
purity_word_category= []
NMI_word_category= []


for i in range(10):
    fscore_word_concept.append(scores.FScore(word_concept_cluster_list[i].labels_))
    purity_word_concept.append(scores.purity(word_concept_cluster_list[i].labels_))
    NMI_word_concept.append(scores.NMI(word_concept_cluster_list[i].labels_))

    fscore_word_category.append(scores.FScore(word_category_cluster_list[i].labels_))
    purity_word_category.append(scores.purity(word_category_cluster_list[i].labels_))
    NMI_word_category.append(scores.NMI(word_category_cluster_list[i].labels_))


bestScore = scores.higherScore(fscore_word_concept, NMI_word_concept, purity_word_concept)
bestAlpha = bestScore*0.1+0.1
similarityMatrix = TFIDF.sum_cos_distance(plainDocsDistanceMatrix, documentConceptDistanceMatrix, documentCategoryDistanceMatrix, bestAlpha, 0,1 )
clustering_data_dict["word_concept"] = AgglomerativeClustering(affinity='precomputed', n_clusters=20, linkage='complete').fit(similarityMatrix)

bestScore = scores.higherScore(fscore_word_category, NMI_word_category, purity_word_category)
bestBeta = bestScore*0.1+0.1
similarityMatrix = TFIDF.sum_cos_distance(plainDocsDistanceMatrix, documentConceptDistanceMatrix, documentCategoryDistanceMatrix, 0, bestBeta+0.1,1 )
clustering_data_dict["word_category"] = AgglomerativeClustering(affinity='precomputed', n_clusters=20, linkage='complete').fit(similarityMatrix)


#for concept category
concept_category_cluster_list =[]
alpha =0.1
beta=0.1
for i in range(10):
    for j in range(10):
        similarityMatrix = TFIDF.sum_cos_distance(plainDocsDistanceMatrix, documentConceptDistanceMatrix, documentCategoryDistanceMatrix, alpha, beta,0)
        concept_category_cluster_list.append(AgglomerativeClustering(affinity='precomputed', n_clusters=20, linkage='complete').fit(similarityMatrix))
        beta=beta+0.1
    alpha = alpha+0.1


fscore_concept_category = []
purity_concept_category= []
NMI_concept_category= []

for i in range (100):
    fscore_concept_category.append(scores.FScore(concept_category_cluster_list[i].labels_))
    purity_concept_category.append(scores.purity(concept_category_cluster_list[i].labels_))
    NMI_concept_category.append(scores.NMI(concept_category_cluster_list[i].labels_))



bestScore = scores.higherScore(fscore_concept_category, NMI_concept_category, purity_concept_category)

#i take the 5th iteration's values
similarityMatrix = TFIDF.sum_cos_distance(plainDocsDistanceMatrix, documentConceptDistanceMatrix, documentCategoryDistanceMatrix, 0.1, 0.5,0 )
clustering_data_dict["concept_category"] = AgglomerativeClustering(affinity='precomputed', n_clusters=20, linkage='complete').fit(similarityMatrix)

#for word concept category
similarityMatrix = TFIDF.sum_cos_distance(plainDocsDistanceMatrix, documentConceptDistanceMatrix, documentCategoryDistanceMatrix, bestAlpha, bestBeta,1 )
clustering_data_dict["word_concept_category"] = AgglomerativeClustering(affinity='precomputed', n_clusters=20, linkage='complete').fit(similarityMatrix)

print("WORD")
print("fscore: "+str(scores.FScore(clustering_data_dict["word"].labels_)))
print("nmi: " + str(scores.NMI(clustering_data_dict["word"].labels_)) )
print("purity: "+ str(scores.purity(clustering_data_dict["word"].labels_)))
print("\n")

print("CONCEPT")
print("fscore: "+str(scores.FScore(clustering_data_dict["concept"].labels_)))
print("nmi: "+str(scores.NMI(clustering_data_dict["concept"].labels_)))
print("purity: "+str(scores.purity(clustering_data_dict["concept"].labels_)))
print("\n")


print("CATEGORY")
print("fscore: "+str(scores.FScore(clustering_data_dict["category"].labels_)))
print("nmi: "+str(scores.NMI(clustering_data_dict["category"].labels_)))
print("purity: "+str(scores.purity(clustering_data_dict["category"].labels_)))
print("\n")


print("WORD CONCEPT")
print("fscore: "+str(scores.FScore(clustering_data_dict["word_concept"].labels_)))
print("nmi: "+str(scores.NMI(clustering_data_dict["word_concept"].labels_)))
print("purity: "+str(scores.purity(clustering_data_dict["word_concept"].labels_)))
print("\n")

print("WORD CATEGORY")
print("fscore: "+str(scores.FScore(clustering_data_dict["word_category"].labels_)))
print("nmi: "+str(scores.NMI(clustering_data_dict["word_category"].labels_)))
print("purity: "+str(scores.purity(clustering_data_dict["word_category"].labels_)))
print("\n")


print("CONCEPT CATEGORY")
print("fscore: "+str(scores.FScore(clustering_data_dict["concept_category"].labels_)))
print("nmi: "+ str(scores.NMI(clustering_data_dict["concept_category"].labels_)))
print("purity: "+ str(scores.purity(clustering_data_dict["concept_category"].labels_)))
print("\n")


print("WORD CONCEPT CATEGORY")
print("fscore: "+str(scores.FScore(clustering_data_dict["word_concept_category"].labels_)))
print("nmi: "+str(scores.NMI(clustering_data_dict["word_concept_category"].labels_)))
print("purity: "+str(scores.purity(clustering_data_dict["word_concept_category"].labels_)))

