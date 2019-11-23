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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import fbeta_score
from sklearn.cluster import AgglomerativeClustering
import math


import scipy.special

# the two give the same results 
#scipy.special.binom(10, 5)


#########################
#clean dataset without useless elements
newsgroup_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))

dataset = []
topics=[]
nDocs = 2000
clusterSize=20

for i in range (nDocs):
    dataset.append(newsgroup_train.data[i])
    topics.append(newsgroup_train.target[i])


#########################
#for clustering
documentTopics = []

with open('topics.json', 'r') as myfile:
    data=myfile.read()
obj = json.loads(data)
for element in obj:
    documentTopics.append(element)


#dictionary that, for each topic, return the number of docs of that topic
elements_per_topic = {}

for element in documentTopics:
    element = element[0]
    if element in elements_per_topic:
        elements_per_topic[element]= elements_per_topic[element]+1
    if element not in elements_per_topic:
        elements_per_topic[element] =1


#for each topic id, how many elements are in it
topicID_nElements = {}
for i in range(nDocs):
    topic = topics[i]
    #i is document id, topic is topicID
    if topic in topicID_nElements:
        topicID_nElements[topic].append(i)
    if topic not in topicID_nElements:
        topicID_nElements[topic] = []
        topicID_nElements[topic].append(i)




#dictionary that cointains, for each topic, the id of the document
document_list_for_each_topic = {}
for i in range(nDocs):
    topic =documentTopics[i][0]
    if topic in document_list_for_each_topic:
        document_list_for_each_topic[topic].append(i)
    if topic not in document_list_for_each_topic:
        document_list_for_each_topic[topic] = []
        document_list_for_each_topic[topic].append(i)

def higherScore(Fscore, NMI, purity):
    #input: 3 lists of 10 values each
    #retrieve id with highest value
    higher_score_indices = []

    higher_score_indices.append(Fscore.index(max(Fscore)))

    higher_score_indices.append(NMI.index(max(NMI)))

    higher_score_indices.append(purity.index(max(purity)))
    result = max(set(higher_score_indices), key = higher_score_indices.count) 

    return result

def create_bucket_cluster_list(clusters):
    clusters_list = clusters.tolist()
    dict_single_cluster_elements_retrieved={}
    #each cluster is in a bucket
    for i in range(len(clusters_list)):
        if clusters_list[i] in dict_single_cluster_elements_retrieved:
            dict_single_cluster_elements_retrieved[clusters_list[i]].append(i)
        if clusters_list[i] not in dict_single_cluster_elements_retrieved:
            dict_single_cluster_elements_retrieved[clusters_list[i]] = []
            dict_single_cluster_elements_retrieved[clusters_list[i]].append(i)     
    return dict_single_cluster_elements_retrieved


def retrieve_real_label_clusters(dict_single_cluster_elements_retrieved):
    dict_clusters_real_labels = {}
    for i in range(len(dict_single_cluster_elements_retrieved)):
        for element in dict_single_cluster_elements_retrieved[i]:
            if i in dict_clusters_real_labels:
                dict_clusters_real_labels[i].append(topics[element])
            if i not in dict_clusters_real_labels:
                dict_clusters_real_labels[i] = []
                dict_clusters_real_labels[i].append(topics[element])
    return dict_clusters_real_labels


def NMI(clusters):
    #dictionary containing, for each i, the elements
    dict_single_cluster_elements_retrieved = create_bucket_cluster_list(clusters)
    dict_clusters_real_labels = retrieve_real_label_clusters(dict_single_cluster_elements_retrieved)
    ##########################

    P_omega = []
    P_classes = []

    for i in range(clusterSize):
        P_omega.append(len(dict_clusters_real_labels[i])/nDocs )
        P_classes.append(len(topicID_nElements[i])/nDocs)     

    H_omega = 0
    H_classes=0
    for i in range(clusterSize):
        H_omega = H_omega +P_omega[i]*math.log(P_omega[i],10)
        H_classes = H_classes+ P_classes[i]*math.log(P_classes[i], 10)
    
    H_classes = abs(H_classes)
    H_omega = abs(H_omega)


    I_omega_classes = 0

    for i in range(clusterSize):
        cluster = dict_clusters_real_labels[i]
        nLabelsPerCluster = {}
        for element in cluster:
            if element in nLabelsPerCluster:
                nLabelsPerCluster[element] = nLabelsPerCluster[element]+1

            if element not in nLabelsPerCluster:
                nLabelsPerCluster[element] =1
        
        for j in range(clusterSize):
            if j in nLabelsPerCluster:
                p_w_c = nLabelsPerCluster[j]/nDocs
                I_omega_classes = I_omega_classes + p_w_c * math.log(p_w_c/(P_classes[j]*P_omega[i]),10)
        
    NMI = I_omega_classes/((H_omega+H_classes)/2)
    return NMI

    

def purity(clusters):
    dict_single_cluster_elements_retrieved=create_bucket_cluster_list(clusters)
    dict_clusters_real_labels = retrieve_real_label_clusters(dict_single_cluster_elements_retrieved)

    max_for_each_cluster = []
    count_max_element = []
    for i in range(clusterSize):
        cluster = dict_clusters_real_labels[i]
        max_for_each_cluster.append(max(set(cluster), key = cluster.count))
        count_max_element.append(cluster.count(max_for_each_cluster[i]))


    result =0
    for i in range (clusterSize):
        result += count_max_element[i]

    purity = result/nDocs
    #print("Purity is: "+str(purity))
    return purity


#reference
#https://stats.stackexchange.com/questions/89030/rand-index-calculation
def FScore(clusters):
    dict_single_cluster_elements_retrieved= create_bucket_cluster_list(clusters)
    dict_clusters_real_labels = retrieve_real_label_clusters(dict_single_cluster_elements_retrieved)
    ###########

    #iterate over 20 clusters
    list_dict_label_frequency_in_cluster = []
    for i in range(clusterSize):
        dict_label_frequency_in_cluster = {}

        #for each cluster create a dictionary with label frequencies
        cluster = dict_clusters_real_labels[i]
        for element in cluster:
            if element in dict_label_frequency_in_cluster:
                dict_label_frequency_in_cluster[element] = dict_label_frequency_in_cluster[element]+1
            if element not in dict_label_frequency_in_cluster:
                dict_label_frequency_in_cluster[element] =1
        list_dict_label_frequency_in_cluster.append(dict_label_frequency_in_cluster)
 
    tp_fp =0
    for i in range(clusterSize):
        #sum of all binomial coefficient from each cluster
        cluster_size = len(dict_clusters_real_labels[i])
        tp_fp = tp_fp + scipy.special.binom(cluster_size, 2)

    tp=0

    for dict_cluster in list_dict_label_frequency_in_cluster:
    #for each cluster, get the binomial coefficient of each element 
        for i in range(clusterSize):
            if i in dict_cluster and dict_cluster[i]>1:
                tp = tp+ scipy.special.binom(dict_cluster[i], 2)

    fp = tp_fp-tp
    
    #total number of pairs N*(N-1)/2
    Npairs = len(dataset)*(len(dataset)-1)/2


    fn =0
    #no need to check last cluster
    for i in range(19):
        dict_cluster_i = list_dict_label_frequency_in_cluster[i]
        for j in range (i+1,clusterSize):
            dict_cluster_j = list_dict_label_frequency_in_cluster[j]
            for k in range(clusterSize):
                if k in dict_cluster_i and k in dict_cluster_j:
                    fn = fn + dict_cluster_i[k]*dict_cluster_j[k]

    tn_fn =0
    for i in range(19):
        clusterI = len(dict_clusters_real_labels[i])
        for j in range(i+1, clusterSize):
            clusterJ = len(dict_clusters_real_labels[j])
            tn_fn = tn_fn+ clusterI*clusterJ
    tn = tn_fn -fn

    #print("True Positive: "+ str(tp)+ "\nFalse Positive: "+str(fp)+ "\nTrue Negative: "+ str(tn)+ "\nFalse Negative: "+ str(fn))

    precision = tp/ (tp+fp)
    recall = tp/(tp+fn)

    #print("Precision: "+str(precision)+"\nRecall: "+str(recall))

    fscore = (2*precision*recall)/(precision+recall)
    #print("FScore is: ")
    #print(fscore)

    return fscore