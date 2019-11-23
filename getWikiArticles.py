import nltk
import string
import numpy as np
import time
import json
import os
import wikipedia
import urllib

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag


wikipedia.set_lang("en")

list_of_documents =[]

article_categories_dict={} #dict word - list

disambiguation_list = []

wiki_vocabulary_list = []

wiki_term_article_dict = {}

pageError_list=[]

# import list of disambiguation
with open("disambiguation.json", 'r+') as myfile:
    data=myfile.read()
obj = json.loads(data)
for element in obj:
    disambiguation_list.append(element)

#import list of wikipedia vocabulary
with open("wiki_vocabulary_list.json", 'r+') as myfile:
    data=myfile.read()
obj = json.loads(data)
for element in obj:
    wiki_vocabulary_list.append(element)

#import list of pageError
with open("pageError_list.json", 'r+') as myfile:
    data=myfile.read()
obj = json.loads(data)
for element in obj:
    pageError_list.append(element)

#import dict
with open("article_categories_dict.json", 'r+') as myfile:
    data=myfile.read()
article_categories_dict = json.loads(data)

#import wiki_term_article_dict
with open("wiki_term_article_dict.json", 'r+') as myfile:
    data=myfile.read()
wiki_term_article_dict = json.loads(data)

# read file
with open('list_of_tokenized_docs_for_wiki.json', 'r') as myfile:
    data=myfile.read()
obj = json.loads(data)

for element in obj:
    list_of_documents.append(element)

count =0



for i in range(2000):
    doc = list_of_documents[i]
    if i == 498 or i ==767 or i ==1492 or i ==1506:
        continue
    
    print(count)
    for term in doc:

        #if word does not start with special carachters (gives error with python wikipedia api)
        if not ("|" in term or term.startswith("\\") or term.startswith(".") or term.startswith("<") or term.startswith(">") or term.startswith("-") or term.startswith("\'") or term.startswith("/") or term.startswith("#")  or term.startswith("\*") or term.startswith("\+") or term.startswith("_")  or term.startswith("=") or term.startswith("[") or term.startswith("|") or term.startswith("~") or term.startswith("]") or term.startswith("^") or term.startswith(":")):
            try:   
                #if word has not been already searched for (and put into found, disambiguation or pageError) do the wiki lookup
                if (term not in wiki_term_article_dict) and (term not in disambiguation_list) and (term not in pageError_list):
                    print(term)
                    term= term.replace(":", "")
                    page = wikipedia.WikipediaPage(str(term))
                    #insert word in dictionary to match lemmatized term with wikipedia title
                    if term not in wiki_term_article_dict:
                        wiki_term_article_dict[term] = str(page.title)
                    #create a list for categories of each word
                    article_categories =[]
                    #add categories to word
                    for element in page.categories:
                        article_categories.append(element.lower())
                    #remove administrative categories
                    keyWordFilter = ('articles','wikipedia','needing','use' ,'confirmation','cs1','cs1:',"stubs","incomplete","lists","redirects","wiktionary",'british english',"dmy",'mdy','wikidata','category','webarchive', 'pages using timeline',"needing confirmation", 'links','reference', 'pages','error','disputes','ambiguous', 'engvarb','use american english', 'use british english' )

                    categories = [x for x in article_categories if not any(word in x.split(' ') for word in keyWordFilter)]
                    #add list of categories to article
                    article_categories_dict[page.title] = categories

                    wiki_vocabulary_list.append(term)


            except wikipedia.exceptions.PageError:
                print("pageError")
                pageError_list.append(term)


            except wikipedia.exceptions.DisambiguationError:
                print("disambiguation")
                disambiguation_list.append(term)
                continue

    count=count+1
    

disambiguation_dump = json.dumps(disambiguation_list)
f = open("disambiguation.json", "w+")
f.write(disambiguation_dump)
f.close

wiki_term_article_dict_dump = json.dumps(wiki_term_article_dict)
f = open("wiki_term_article_dict.json", "w+")
f.write(wiki_term_article_dict_dump)
f.close

article_categories_dict_dump = json.dumps(article_categories_dict)
f = open("article_categories_dict.json", "w+")
f.write(article_categories_dict_dump)
f.close

wiki_vocabulary_list_dump = json.dumps(wiki_vocabulary_list)
f = open("wiki_vocabulary_list.json", "w+")
f.write(wiki_vocabulary_list_dump)
f.close

pageError_list_dump = json.dumps(pageError_list)
f = open("pageError_list.json", "w+")
f.write(pageError_list_dump)
f.close        

#levo dai token dei documenti le parole che non sono presenti nel vocabolario di wiki?




