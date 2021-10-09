# To DO recall,
# precision,
# Matthews correlation coefficient (MCC),
# work saved over sampling (WSS) measures
# and the number of support vectors.
# tfidf
# most similar
import tempfile, csv
import csv
import json
import os
import re
from flask import Flask, jsonify, request, send_file, flash, redirect, render_template
import requests
import scipy.spatial as sp
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import nltk
import csv
import pandas as pd
# from empath import Empath
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

# TF-IDF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
from nltk.corpus import wordnet
import SLR_API.convert_JSON_to_csv as json_converter
import mysql.connector
from mysql.connector import errorcode
import mysql.connector.errors
import json
import SLR_API.def_controller as em

# Implementing the API
# from pandas.tests.io.excel.test_xlsxwriter import xlsxwriter
from SLR_API import db_controller

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

stopwords_local = set(stopwords.words('english'))


# insert string in string to create file names
def insert_dash(string, index, addin):
    return string[:index] + addin + string[index:]


# building just one dictionary of csv files
def csv_2_dict(variables_file):
    with open(variables_file, newline='') as pscfile:
        reader = csv.reader(pscfile)
        next(reader)
        results = dict(reader)
    return results


# building list of dictionaries of csv files
def csv_dict_list(variables_file):
    reader = csv.DictReader(open(variables_file, 'r'))
    dict_list = []
    for line in reader:
        dict_list.append(line)
    return dict_list


# convert received model from Swagger
def convert_model(input_model):
    if input_model == "Wikipedia":
        input_model = "WikiW2V"
    elif input_model == "WikipediaNews":
        input_model = "WikiNewsFast"
    elif input_model == "Crawlweb":
        input_model = "glove"
    elif input_model == "ConceptNet":
        input_model = "Numberbatch"
    elif input_model == "Universal":
        input_model = "Universal"
    elif input_model == "ELMo":
        input_model = "ELMo"

    return input_model


# creating a list of a given corpus
def csv2list(v_filename):
    corpus_list = []
    with open(v_filename, encoding='utf-8', errors='ignore') as corpus_set:
        row_reader = csv.reader(corpus_set)
        # abstracts.append(' '.join(list(row_reader)))
        corpus_list = [record for record in row_reader]
        corpus_list = [''.join(record) for record in corpus_list]  ## change the list within the list to string
    return corpus_list


# check the file validity
def __file_validity():
    # todo
    return True


# creating the vectors of abstracts and queries' keywords
def corpus_to_keywords(corpus):
    calling_endpoint = 'http://localhost:9000/?properties="annotators":"tokenize,pos,lemma","outputFormat":"json"'
    response = requests.post(calling_endpoint, data=corpus.replace('.', ' and ').encode('utf-8'))
    result = response.json()
    keywords_list = []
    for sentence in result['sentences']:
        for token in sentence['tokens']:
            if not token['word'].lower() in stopwords_local and not token['lemma'].lower() in stopwords_local:
                if token['pos'] in {'VB', 'VBN', 'VBD', 'VBG', 'VBP', 'VBZ', 'NNP', 'NNS', 'NNPS', 'NN', 'JJ'}:
                    keywords_list.append(token['lemma'])
    return keywords_list


# Enriching keywords set with Synonyms
def keywords_to_synonym(corpus):
    calling_endpoint = 'http://localhost:9000/?properties="annotators":"tokenize,pos,lemma","outputFormat":"json"'
    response = requests.post(calling_endpoint, data=corpus.replace('.', ' and '))
    result = response.json()
    synonym_list = []
    for sentence in result['sentences']:
        for token in sentence['tokens']:
            if not token['word'].lower() in stopwords_local and not token['lemma'].lower() in stopwords_local:
                if token['pos'] in {'VB', 'VBN', 'VBD', 'VBG', 'VBP', 'VBZ', 'NNP', 'NNS', 'NNPS', 'NN', 'JJ'}:
                    for syn in wordnet.synsets(token['lemma']):
                        for l in syn.lemmas():
                            synonym_list.append(l.name())
    return synonym_list


# creating the vector's value with word2Vec models
def __word2vec_model(word, embedding_model):
    embedding_model = convert_model(embedding_model)
    if embedding_model == 'WikiW2V':
        model_url = 'http://drstrange.cse.unsw.edu.au:5001/WikiW2V/vector?word={}'
    elif embedding_model == 'WikiNewsFast':
        model_url = 'http://drstrange.cse.unsw.edu.au:5001/WikiNewsFast/vector?word={}'
    elif embedding_model == 'Glove':  # Global Vectors for Word Representation  Stanford NLP Group
        model_url = 'http://drstrange.cse.unsw.edu.au:5001/glove/events/vector?word={}'
        # model_url='http: //localhost'
    elif embedding_model == 'Numberbatch':
        model_url = 'http://drstrange.cse.unsw.edu.au:5001/Numberbatch/vector?word={}'
    elif embedding_model == 'Universal':
        model_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'
    elif embedding_model == 'ELMo':
        model_url = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'
   
    response = requests.get(model_url.format(word))
    if response.status_code == 200:
        if 'vector' in json.loads(response.content):
            return json.loads(response.content)['vector']
        else:
            return None


# Most similarity with word2Vec embedding
def most_similar_model(word, embedding_model):
    embedding_model = convert_model(embedding_model)
    if embedding_model == 'WikiW2V':
        model_url = 'http://drstrange.cse.unsw.edu.au:5001/WikiW2V/most_similar?word={}'
    elif embedding_model == 'WikiNewsFast':
        model_url = 'http://drstrange.cse.unsw.edu.au:5001/WikiNewsFast/most_similar?word={}'
    elif embedding_model == 'glove':
        model_url = 'http://drstrange.cse.unsw.edu.au:5001/glove/events/most_similar?word={}'
    elif embedding_model == 'Numberbatch':
        model_url = 'http://drstrange.cse.unsw.edu.au:5001/Numberbatch/most_similar?word={}'
    response = requests.get(model_url.format(word))
    if response.status_code == 200:
        return json.loads(response.content)['similar_words']
    else:
        return None


# back up 20201105
# def calculating_similarity_word2vec(avg_abstract_vector: [], ave_question_vector: [], corpus_type, path) -> object:
#     list_of_dict = []
#     if corpus_type == 'seed':
#         for abstract_var in avg_abstract_vector:
#             similarity_list = {}
#             abst_key = list(abstract_var.keys())[0]
#             for seed in ave_question_vector:
#                 seed_key = list(seed.keys())[0]
#                 similarity_list['absId'] = '{}'.format(abstract_var[abst_key]['AbstractID'])
#                 similarity_list['seedId'] = '{}'.format(seed[seed_key]['AbstractID'])
#                 similarity_list['similarity'] = '{}'.format(
#                     1 - sp.distance.cosine(seed['vector'], abstract_var['vector']))
#                 # similarity_list['Include'] = '{}'.format(abstract_var[abst_key]['incl_excl'])
#                 list_of_dict.append(similarity_list.copy())
#
#     else:
#         for abstract_var in avg_abstract_vector:
#             similarity_list = {}
#             abst_key = list(abstract_var.keys())[0]
#             for question in ave_question_vector:
#                 question_key = list(question.keys())[0]
#                 similarity_list['absId'] = '{}'.format(abstract_var[abst_key]['AbstractID'])
#                 similarity_list['qesId'] = '{}'.format(question_key)
#                 similarity_list['similarity'] = '{}'.format(
#                     1 - sp.distance.cosine(question['vector'], abstract_var['vector']))
#                 # similarity_list['Include'] = '{}'.format(abstract_var[abst_key]['incl_excl'])
#                 list_of_dict.append(similarity_list.copy())
#
#     # write to a CSV
#     keys = set()
#     # myorder = ['absID', 'qesId', 'similarity']
#     # from collections import OrderedDict
#     # ordered = OrderedDict((k, list_of_dict[k]) for k in myorder)
#     for d in list_of_dict:
#         keys.update(d.keys())
#     # with open("C:\\Users\\maisieb01\\Desktop\\PHD\\Framework\\Data\\20082020\\test.csv", 'a') as output_file:
#     if path is not '':
#         try:
#             with open(path, 'a') as output_file:
#                 dict_writer = csv.DictWriter(output_file, fieldnames=keys, restval='-', delimiter=',')
#                 dict_writer.writeheader()
#                 dict_writer.writerows(list_of_dict)
#         except ValueError as e:
#             json.dumps("The path is not valid {}".format(e),
#                        content_type="application/json")
#         # sort based on similarity
#     return json.dumps(sorted(list_of_dict, key=lambda i: i['similarity'], reverse=True))
#     # return json.dumps(sorted(list_of_dict, key=lambda i: i['absId'], reverse=True))
#     # Get JSON Data


# calculating the average value of the vector of abstracts
def calculating_similarity_word2vec(avg_abstract_vector: [], ave_question_vector: [], corpus_type, path) -> object:
    list_of_dict = []
    list_of_dict_filter = []
    if corpus_type == 'query':
        avg_query_vector = avg_abstract_vector
        ave_slr_query_vector = ave_question_vector
        for query_var in avg_query_vector:
            similarity_list = {}
            query_key = list(query_var.keys())[0]
            if query_key != '':
                for slr in ave_slr_query_vector:
                    slr_key = list(slr.keys())[0]
                    if slr_key != '':
                        similarity_list['Query_Id'] = '{}'.format(query_key)
                        similarity_list['query'] = '{}'.format(query_var['{}'.format(query_key)])
                        similarity_list['SLRId'] = '{}'.format(slr_key)
                        similarity_list['slr_query'] = '{}'.format(slr['{}'.format(slr_key)])
                        similarity_list['similarity'] = '{}'.format(
                            1 - sp.distance.cosine(slr['vector'], query_var['vector']))
                        list_of_dict.append(similarity_list.copy())

    elif corpus_type == 'seed':
        for abstract_var in avg_abstract_vector:
            similarity_list = {}
            abst_key = list(abstract_var.keys())[0]
            for seed in ave_question_vector:
                seed_key = list(seed.keys())[0]
                similarity_list['absId'] = '{}'.format(abstract_var[abst_key]['AbstractID'])
                similarity_list['abstract'] = '{}'.format(abstract_var[abst_key]['Abstract'])
                similarity_list['seedId'] = '{}'.format(seed_key)
                similarity_list['seed'] = '{}'.format(seed[seed_key]['Abstract'])
                similarity_list['similarity'] = '{}'.format(
                    1 - sp.distance.cosine(seed['vector'], abstract_var['vector']))
                # similarity_list['Include'] = '{}'.format(abstract_var[abst_key]['incl_excl'])
                list_of_dict.append(similarity_list.copy())
    else:
        for abstract_var in avg_abstract_vector:
            similarity_list = {}
            abst_key = list(abstract_var.keys())[0]
            for question in ave_question_vector:
                question_key = list(question.keys())[0]
                similarity_list['absId'] = '{}'.format(abstract_var[abst_key]['AbstractID'])
                similarity_list['abstract'] = '{}'.format(abstract_var[abst_key]['Abstract'])
                similarity_list['qesId'] = '{}'.format(question_key)
                similarity_list['question'] = '{}'.format(question['{}'.format(question_key)])
                similarity_list['similarity'] = '{}'.format(
                    1 - sp.distance.cosine(question['vector'], abstract_var['vector']))
                # similarity_list['Include'] = '{}'.format(abstract_var[abst_key]['incl_excl'])
                list_of_dict.append(similarity_list.copy())
                # filter the list base on the threshold
                # if float(similarity_list['similarity']) > float(threshold):
                list_of_dict_filter.append(similarity_list.copy())

    # write to a CSV and excel
    keys = set()
    # headers = ['absID', 'qesId', 'similarity']
    # from collections import OrderedDict
    # ordered = OrderedDict((k, list_of_dict[k]) for k in myorder)
    for d in list_of_dict:
        keys.update(d.keys())
    if corpus_type != 'seed':
        if path is not '':
            try:
                # create csv file if it does not exist
                # if not os.path.isfile(path):
                #     with open(path, 'w')as csv_file:
                #         csv_file.writelines(', '.join(headers))
                # open the files and start the loop
                with open(path, 'a+') as output_file:
                    dict_writer = csv.DictWriter(output_file, fieldnames=keys, restval='-', delimiter=',')
                    dict_writer.writeheader()
                    # if list_of_dict[d.keys['similarity']]>= float(threshold):
                    dict_writer.writerows(list_of_dict)
                    # else:
                    #     dict_writer.writerows(list_of_dict)
            except ValueError as e:
                json.dumps("The path is not valid {}".format(e),
                           content_type="application/json")
    if corpus_type == 'seed':
        try:
            path_seed = insert_dash(path, len(path) - 4, '_seed')
            with open(path_seed, 'a+') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys, restval='-', delimiter=',')
                dict_writer.writeheader()
                dict_writer.writerows(list_of_dict)
        except ValueError as e:
            json.dumps("The path is not valid {}".format(e),
                       content_type="application/json")

    # sort based on similarity
    return json.dumps(sorted(list_of_dict, key=lambda i: i['similarity'], reverse=True))
    # return json.dumps(sorted(list_of_dict, key=lambda i: i['absId'], reverse=True))
    # Get JSON Data


def __vec_keyword_corpus(corpus, corpus_type):
    if corpus_type == "abstract":
        for corpus_item in corpus:
            author = list(corpus_item.keys())[0]
            corpus_item[author]['Abstract'] = corpus_item[author]['Abstract'].lower()
            corpus_item[author]['Abstract'] = re.sub(r"[#$%&()!*+-./:;<=>?@[\]^_`~,'{|}]", " ",
                                                     corpus_item[author]['Abstract'])
            corpus_item.update({'synonym': corpus_to_keywords(corpus_item[author]['Abstract'])})
            corpus_item.update({'keywords': corpus_to_keywords(corpus_item[author]['Abstract'])})
    return corpus


# create a clean , Stemmed list of given corpus ( abstracts , questions ,seeds, objectives)
def __vec_corpus(corpus, corpus_type):
    if corpus_type == "abstract":
        for corpus_item in corpus:
            author = list(corpus_item.keys())[0]
            corpus_item[author]['Abstract'] = corpus_item[author]['Abstract'].lower()
            corpus_item[author]['Abstract'] = re.sub(r"[#$%&()!*+-./:;<=>?@[\]^_`~,'{|}]", " ",
                                                     corpus_item[author]['Abstract'])
            corpus_item.update({'keywords': corpus_to_keywords(corpus_item[author]['Abstract'])})
    elif corpus_type == "seeds":
        for corpus_item in corpus:
            author = list(corpus_item.keys())[0]
            corpus_item[author]['seed'] = corpus_item[author]['Abstract'].lower()
            corpus_item[author]['seed'] = re.sub(r"[#$%&()!*+-./:;<=>?@[\]^_`~,'{|}]", " ", corpus_item[author]['seed'])
            corpus_item.update({'keywords': corpus_to_keywords(corpus_item[author]['seed'])})

    elif corpus_type == "questions":
        for corpus_item in corpus:
            question_id = list(corpus_item.keys())[0]
            corpus_item[question_id] = corpus_item[question_id].lower()
            corpus_item[question_id] = re.sub(r"[#$%&()!*+-./:;<=>?@[\]^_`~,'{|}]", " ", corpus_item[question_id])
            corpus_item.update({'keywords': corpus_to_keywords(corpus_item[question_id])})

            # clean_corpus.append({'abstract': corpus_item["abstract"], 'keywords': corpus_to_keywords(corpus_item["abstract"])})

    elif corpus_type == "objectives":
        for corpus_item in corpus:
            question_id = list(corpus_item.keys())[0]
            corpus_item[question_id] = corpus_item[question_id].lower()
            corpus_item[question_id] = re.sub(r"[#$%&()!*+-./:;<=>?@[\]^_`~,'{|}]", " ", corpus_item[question_id])
            corpus_item.update({'keywords': corpus_to_keywords(corpus_item[question_id])})
            # clean_corpus.append({'abstract': corpus_item["abstract"], 'keywords': corpus_to_keywords(corpus_item["abstract"])})
    elif corpus_type == "queries":
        for corpus_item in corpus:
            query_id = list(corpus_item.keys())[0]
            corpus_item[query_id] = corpus_item[query_id].lower()
            corpus_item[query_id] = re.sub(r"[#$%&()!*+-./:;<=>?@[\]^_`~,'{|}]", " ", corpus_item[query_id])
            corpus_item.update({'keywords': corpus_to_keywords(corpus_item[query_id])})
    elif corpus_type == "slr":
        for corpus_item in corpus:
            slr_id = list(corpus_item.keys())[0]
            corpus_item[slr_id] = corpus_item[slr_id].lower()
            corpus_item[slr_id] = re.sub(r"[#$%&()!*+-./:;<=>?@[\]^_`~,'{|}]", " ", corpus_item[slr_id])
            corpus_item.update({'keywords': corpus_to_keywords(corpus_item[slr_id])})
    else:
        for corpus_item in corpus:
            corp_id = list(corpus_item.keys())[0]
            corpus_item[corp_id] = corpus_item[corp_id].lower()
            corpus_item[corp_id] = re.sub(r"[#$%&()!*+-./:;<=>?@[\]^_`~,'{|}]", " ", corpus_item[corp_id])
            corpus_item.update({'keywords': corpus_to_keywords(corpus_item[corp_id])})
    return corpus


# SVM classifier
def SVM_classifier(abstract, path: str):
    # # --> cleaning abstracts
    # clean_abstracts = __vec_corpus(abstract["abstracts"], "abstract")
    # for item in clean_abstracts:
    #     abstract_words_vectors = []
    #     for keyword in item['keywords']:
    #         # --> embedding each keyword within the list of keywords
    #         word_vector = __word2vec_model(keyword, embedding_model)
    #         if word_vector is not None:
    #             abstract_words_vectors.append(word_vector)
    #         avg_abstract_vector = np.mean(abstract_words_vectors, axis=0)
    #     item.update({'vector': avg_abstract_vector})
    #     all_abstracts_avg_vectors.append(item)
    # # --> calling calculating_similarity to calculate the cosine similarity between abstracts and questions
    #
    # dataset = pd.read_csv('C:\\Users\\maisieb01\\Desktop\\PHD\\Framework\\Data\\SLR\\Clean\\EXP\\Asbtracts-Classifier.csv',
    #                       encoding='windows-1252', error_bad_lines=False)
    #
    # X = dataset.drop('Class', axis=1)
    # y = dataset['Class']
    #
    # from sklearn.model_selection import train_test_split
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    #
    # from sklearn.svm import SVC
    #
    # svclassifier = SVC(kernel='linear')
    # svclassifier.fit(X_train, y_train)
    #
    # y_pred = svclassifier.predict(X_test)
    #
    # from sklearn.metrics import classification_report, confusion_matrix
    #
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    return True


# def keywords2Vec():
#         clean_questions = __vec_corpus(abstract_question["questions"], "questions")
#         for item in clean_questions:
#             question_words_vectors = []
#             for keyword in item['keywords']:
#                 word_vector = __word2vec_model(keyword, embedding_model)
#                 if word_vector is not None:
#                     question_words_vectors.append(word_vector)
#                 avg_question_vector = np.mean(question_words_vectors, axis=0)
#             item.update({'vector': avg_question_vector})
#             all_questions_avg_vectors.append(item)
#
#     return True

# --> first Endpoint , getting Abstract and the questions and return the similarity
# --> from server @app.route("/similarity_questions_abstracts", methods=["POST"])
def question_abstract_embed(abstract_question, embedding_model: str, path: str):
    all_abstracts_avg_vectors = []
    all_questions_avg_vectors = []

    # clean the questions
    if "questions" in abstract_question:
        clean_questions = __vec_corpus(abstract_question["questions"], "questions")
        # print(clean_questions)
        for item in clean_questions:
            question_words_vectors = []
            for keyword in item['keywords']:
                word_vector = __word2vec_model(keyword, embedding_model)
                if word_vector is not None:
                    question_words_vectors.append(word_vector)
                avg_question_vector = np.mean(question_words_vectors, axis=0)
            item.update({'vector': avg_question_vector})
            all_questions_avg_vectors.append(item)
    # Clean the  abstracts
    if "abstracts" in abstract_question:
        # --> cleaning abstracts
        clean_abstracts = __vec_corpus(abstract_question["abstracts"], "abstract")
        for item in clean_abstracts:
            abstract_words_vectors = []
            for keyword in item['keywords']:
                # --> embedding each keyword within the list of keywords
                word_vector = __word2vec_model(keyword, embedding_model)
                if word_vector is not None:
                    abstract_words_vectors.append(word_vector)
                avg_abstract_vector = np.mean(abstract_words_vectors, axis=0)
            item.update({'vector': avg_abstract_vector})
            all_abstracts_avg_vectors.append(item)
    # --> calling calculating_similarity to calculate the cosine similarity between abstracts and questions
    return calculating_similarity_word2vec(all_abstracts_avg_vectors, all_questions_avg_vectors, 'abstracts', path
                                           )


# compare queries vectors
def query_embed(query, embedding_model: str, path: str):
    all_queries_avg_vectors = []
    all_queries_slr_avg_vectors = []
    # clean the questions
    if "query" in query:
        clean_queries = __vec_corpus(query["query"], "queries")
        for item in clean_queries:
            queries_words_vectors = []
            for keyword in item['keywords']:
                word_vector = __word2vec_model(keyword, embedding_model)
                if word_vector is not None:
                    queries_words_vectors.append(word_vector)
                avg_queries_vector = np.mean(queries_words_vectors, axis=0)
            item.update({'vector': avg_queries_vector})
            all_queries_avg_vectors.append(item)

    if "slr" in query:
        clean_queries_slr = __vec_corpus(query["slr"], "slr")
        for item in clean_queries_slr:
            slr_words_vectors = []
            for keyword in item['keywords']:
                word_vector = __word2vec_model(keyword, embedding_model)
                if word_vector is not None:
                    slr_words_vectors.append(word_vector)
                avg_slr_vector = np.mean(slr_words_vectors, axis=0)
            item.update({'vector': avg_slr_vector})
            all_queries_slr_avg_vectors.append(item)

    # --> calling calculating_similarity to calculate the cosine similarity between abstracts and questions
    return calculating_similarity_word2vec(all_queries_avg_vectors, all_queries_slr_avg_vectors, 'query', path)


# call DL API (IEEXPLORE)
def IEEE_Xplore(corpus, path: str, from_year: int, to_year: int):
    url = "http://ieeexploreapi.ieee.org/api/v1/search/articles?"
    # key = "&apikey=###################&format=json"
    key = "&apikey=Institute key&format=json&max_records=7000&start_record=1&sort_order=asc&sort_field=relevance&end_year={}".format(
        to_year) + "&start_year={}".format(from_year)
    querytext = 'querytext={}'.format(corpus)
    # querytext = "querytext=(Metadata%20OR%20%22predict%20OR%20%22software%20OR%20%22defect%20OR%20%22qulity%22)"
    # querytext = "querytext=(rfid%20NOT%20%22internet%20of%20things%22)"
    response = requests.get(url + querytext + key)
    json_response = response.json()
    print(json_response)
    print(querytext)
    print(url + querytext + key)
    title = json_response['articles'][0]['title']
    if path is not '':
        try:
            json_converter.convert_json_to_csv(json_response, path)
            format_result(path, insert_dash(path, len(path) - 4, '_format'))
        except ValueError as e:
            json.dumps("The path is not valid {}".format(e), content_type="application/json")


def format_result(path_in, path_out):
    if path_in is not '':
        try:
            col_names = ['publisher', 'rank', 'publication_year', 'abstract']
            col_names_out = 'SLR_Author', 'AbstractID', 'Abstract_Year', 'Abstract'
            df = pd.read_csv(path_in, header=0, usecols=col_names)
            df.reindex(columns=col_names_out)
            # df.rename(columns={'publisher': 'SLR_Author', 'rank': 'AbstractID', 'publication_year': 'Abstract_Year',
            #                    'abstract': 'Abstract'})
            df.to_csv(path_out, header=col_names_out, index=False)
        except ValueError as e:
            json.dumps("The path is not valid {}".format(e), content_type="application/json")
    return True


# After annotating , extract keywords with tag
# def form_search_query_annotate(corpus, embedding_model: str, path: str, call_type: str):
#     words_vectors = []
#     word_vector_Glove = {}
#     list_of_dict_Glove = []
#
#     # for item in corpus:
#     #     abstract_words_vectors = []
#     #     for keyword in item['keywords']:
#     #         word_vector = __word2vec_model(keyword, embedding_model)
#     #         if word_vector is not None:
#     #             question_words_vectors.append(word_vector)
#     #         avg_question_vector = np.mean(question_words_vectors, axis=0)
#     #     item.update({'vector': avg_question_vector})
#     #     all_questions_avg_vectors.append(item)
#
#     for abs in corpus:
#         clean_abstracts = __vec_corpus(corpus["abstracts"], "abstract")
#         for item in clean_abstracts:
#             for keyword in item['keywords']:
#                 words_vectors.append(keyword)
#                 if not any(d['keyword'] == keyword for d in list_of_dict_Glove):
#                     word_vector_Glove['keyword'] = '{}'.format(keyword)
#                     word_vector_Glove['vector'] = '{}'.format(__word2vec_model(keyword, 'Glove'))
#                     list_of_dict_Glove.append(word_vector_Glove.copy())
#
#     ## Add keywords
#     # total number of  words in document
#     total_occurrence = sum(Counter(words_vectors).values())
#     if path is not '':
#         try:
#             with open(path, 'w+', newline='') as output_file:
#                 writer = csv.DictWriter(output_file,
#                                         fieldnames=["Keyword", "Number", "TF", "Tag", "Reward", "Demote"])
#                 writer.writeheader()
#                 write = csv.writer(output_file)
#                 write.writerows(Counter(words_vectors).items())
#                 output_file.close()
#         except ValueError as e:
#             json.dumps("The path is not valid {}".format(e), content_type="application/json")
#
#         # Add TF
#         df = pd.read_csv(path)
#         df['TF'] = df['Number'] / total_occurrence  # adding TF
#         df = df.sort_values(['Number'], ascending=[False])  # sort list
#         df.to_csv(path, index=True)
#
#         # Add synonym for top IDF
#         df = pd.read_csv(path)
#         df.set_index("Keyword", drop=False)
#         for i, row in df.iterrows():
#             if float(df.values[i][3]) * 10 > 0.1:
#                 synonyms_list = []
#                 synonyms = ''
#                 for syn in wordnet.synsets(df.values[i][1]):
#                     for l in syn.lemmas():
#                         if l.name() not in synonyms_list:
#                             synonyms_list.append(l.name())
#                             if synonyms == '':
#                                 synonyms = l.name()
#                             else:
#                                 synonyms = synonyms + ',' + l.name()
#                 df.loc[i, 'Synonyms'] = synonyms
#                 if call_type == 'questions':
#                     df.loc[i, 'Reward'] = 1
#                     df.loc[i, 'Demote'] = 0
#                 elif call_type == 'seeds':
#                     df.loc[i, 'Reward'] = 1
#                     df.loc[i, 'Demote'] = 0
#         df = df.sort_values(['Number'], ascending=[False])  # sort list
#         df.to_csv(path, index=False)
#
#         # create alternative search query
#         df = pd.read_csv(path)
#         df.set_index("Keyword", drop=False)
#         search_query = ''
#         for i, row in df.iterrows():
#             if float(df.values[i][3]) * 10 > 0.1:
#                 if search_query == '':
#                     #  Adding synonym using OR
#                     # search_query = '( {}'.format(df.values[i][1]) + '{}'.format(df.values[i][4].replace(","," OR ")) + ' )'
#                     search_query = '( ' + '{}'.format(df.values[i][4].replace(",", " OR ")) + ' )'
#                     # for i in range(len(list_syn)):
#                     #     search_query = '{}'.format(search_query) +' OR '+'{}'.format(df.values[i][1])
#                 else:
#                     #  Adding synonym using OR
#                     search_query = '{}'.format(search_query) + ' AND ' + '( ' + '{}'.format(
#                         df.values[i][4].replace(",", " OR ")) + ' )'
#                 # list_syn = list(df.values[i][4].split(","))
#                 df.loc[i, 'search query'] = search_query
#         df = df.sort_values(['Number'], ascending=[False])  # sort list
#         df.to_csv(path, index=False)
#
#     return json.dumps(sorted(Counter(words_vectors)))


# Reward _Demote Schema
def Reward_Demote(previous_keywordset: str, new_keywordset: str, iteration: int, path: str):
#     new keyword set has the IR and R tagged
#     previous keyword set has the reward/ demote values
    for each t in new_keywordset
        if  t in previous_keywordset and t is tagged R
                  previous_keywordset [t].reward +1
        if t not in previous_keywordset and t is tagged R
                add t to previous_keywordset
                previous_keywordset[t].reward +1
         else  if  t in previous_keywordset and t is tagged IR
                 previous_keywordset [t].demote +1
        else  if t not in previous_keywordset and t is tagged IR
                   do not add
    if path is not '':
        try:
            with open(path, 'w+', newline='') as output_file:
                writer = csv.DictWriter(output_file, fieldnames=["Keyword", "Number", "TF", "Synonyms","Reward","Demote", "search query","vector"])
    return True


# calculating TF-IDF
def TF_IDF(corpus, review_name: str, iteration: int, corpus_type: str):
    sentences = list()
    if corpus_type == 'questions':
        _min_df = 0.1
        _max_df = 0.3
    else:
        _min_df = 0.1
        _max_df = 0.7
    for corpus_item in corpus:
        question_id = list(corpus_item.keys())[0]
        for l in re.split(r"\.\s|\?\s|\!\s|\n", corpus_item[question_id]):
            if l:
                sentences.append(l)
        cvec = CountVectorizer(stop_words='english', ngram_range=(1, 1), min_df=_min_df, max_df=_max_df,
                               max_features=100)
        sf = cvec.fit_transform(sentences)
        transformer = TfidfTransformer()
        transformed_weights = transformer.fit_transform(sf)
        weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
        weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
        weights_df = weights_df.sort_values(by='weight', ascending=False)

        # calling_endpoint = 'http://localhost:9000/?properties="annotators":"tokenize,pos,lemma","outputFormat":"json"'
        # response = requests.post(calling_endpoint, data=sentences.replace('.', ' and ').encode('utf-8'))
        # result = response.json()
        # keywords_list = []
        # for sentence in result['sentences']:
        #     for token in sentence['tokens']:
        #         if not token['word'].lower() in stopwords_local and not token['lemma'].lower() in stopwords_local:
        #             if token['pos'] in {'VB', 'VBN', 'VBD', 'VBG', 'VBP', 'VBZ', 'NNP', 'NNS', 'NNPS', 'NN', 'JJ'}:
        #                 keywords_list.append(token['lemma'])

        # weights_df.to_csv(insert_dash(path, len(path) - 4, '_idf'), index=True)
        for row in weights_df.iterrows():
            row_index, row_values = row
            db_controller.add_idf(row_values['term'], row_values['weight'], iteration, review_name, corpus_type)
            # query = ("UPDATE Regression_Data.Input SET FITTEDVALUES=" + (
            #     row_values['yhat'].__str__()) + " where timecount=" + (row_values['timecount'].__str__()) + ";")
    return weights_df


# building search query
def form_search_query(corpus, embedding_model: str, path: str, call_type: str):
    # lemmatizer = nltk.stem.WordNetLemmatizer()
    words_vectors = []
    # word_vector_Glove = {}
    # list_of_dict_Glove = []
    # lemma_list = []
    # tf_idf_dic=[]
    if call_type == 'questions':
        clean_questions = __vec_corpus(corpus["questions"], "questions")
        # tf_idf_dic = TF_IDF(corpus["questions"], "Wohono", 1, call_type)
        # tf_idf_dic = TF_IDF(corpus["questions"], "Radjenovic", 1, call_type)
        # tf_idf_dic = TF_IDF(corpus["questions"], "Hall", 1, call_type)
        for item in clean_questions:
            for keyword in item['keywords']:
                words_vectors.append(keyword)

        #         lemma = lemmatizer.lemmatize(keyword, 'v')
        #         if lemma == keyword:
        #             lemma = lemmatizer.lemmatize(keyword)
        #         lemma_list.append(lemma)
        # if not any(d['keyword'] == keyword for d in list_of_dict_Glove):
        #     word_vector_Glove['keyword'] = '{}'.format(keyword)
        #     word_vector_Glove['vector'] = '{}'.format(__word2vec_model(keyword, 'Glove'))
        #     list_of_dict_Glove.append(word_vector_Glove.copy())
    if call_type == 'abstract':
        clean_abstracts = __vec_corpus(corpus["abstracts"], "abstract")
        # clean_questions = __vec_corpus(corpus["questions"], "questions")
        for item in clean_abstracts:
            for keyword in item['keywords']:
                words_vectors.append(keyword)
                # lemma = lemmatizer.lemmatize(keyword, 'v')
                # if lemma == keyword:
                #     lemma = lemmatizer.lemmatize(keyword)
                # lemma_list.append(lemma)

                # if not any(d['keyword'] == keyword for d in list_of_dict_Glove):
                #     word_vector_Glove['keyword'] = '{}'.format(keyword)
                #     word_vector_Glove['vector'] = '{}'.format(__word2vec_model(keyword, 'Glove'))
                #     list_of_dict_Glove.append(word_vector_Glove.copy())

        # for item in clean_questions:
        #     for keyword in item['keywords']:
        #         for keyword in item['keywords']:
        #             words_vectors.append(keyword)
        #             if not any(d['keyword'] == keyword for d in list_of_dict_Glove):
        #                 word_vector_Glove['keyword'] = '{}'.format(keyword)
        #                 word_vector_Glove['vector'] = '{}'.format(__word2vec_model(keyword, 'Glove'))
        #                 list_of_dict_Glove.append(word_vector_Glove.copy())
    elif call_type == 'seeds':
        clean_abstracts = __vec_corpus(corpus["seed_abstracts"], "seeds")
        clean_questions = __vec_corpus(corpus["questions"], "questions")
        for item in clean_abstracts:
            for keyword in item['keywords']:
                words_vectors.append(keyword)
                # if not any(d['keyword'] == keyword for d in list_of_dict_Glove):
                #     word_vector_Glove['keyword'] = '{}'.format(keyword)
                #     word_vector_Glove['vector'] = '{}'.format(__word2vec_model(keyword, 'Glove'))
                #     list_of_dict_Glove.append(word_vector_Glove.copy())
        for item in clean_questions:
            for keyword in item['keywords']:
                words_vectors.append(keyword)
                # if not any(d['keyword'] == keyword for d in list_of_dict_Glove):
                #     word_vector_Glove['keyword'] = '{}'.format(keyword)
                #     word_vector_Glove['vector'] = '{}'.format(__word2vec_model(keyword, 'Glove'))
                #     list_of_dict_Glove.append(word_vector_Glove.copy())

    elif call_type == 'objective':
        clean_objective = __vec_corpus(corpus["objectives"], "objectives")
        clean_questions = __vec_corpus(corpus["questions"], "questions")
        for item in clean_objective:
            for keyword in item['keywords']:
                words_vectors.append(keyword)
        for item in clean_questions:
            for keyword in item['keywords']:
                words_vectors.append(keyword)
                # if not any(d['keyword'] == keyword for d in list_of_dict_Glove):
                #     word_vector_Glove['keyword'] = '{}'.format(keyword)
                #     word_vector_Glove['vector'] = '{}'.format(__word2vec_model(keyword, 'Glove'))
                #     list_of_dict_Glove.append(word_vector_Glove.copy())

    ## Add keywords
    # total number of  words in document
    total_occurrence = sum(Counter(words_vectors).values())
    if path is not '':
        try:
            with open(path, 'w+', newline='') as output_file:
                writer = csv.DictWriter(output_file,
                                        fieldnames=["Keyword", "Number", "TF_IDF", "Synonyms", "Reward", "Demote",
                                                    "search query"])
                writer.writeheader()
                write = csv.writer(output_file)
                write.writerows(Counter(words_vectors).items())
                output_file.close()
        except ValueError as e:
            json.dumps("The path is not valid {}".format(e), content_type="application/json")

        # Add TF_IDF
        df = pd.read_csv(path)
        df['TF_IDF'] = df['Number'] / total_occurrence  # adding TF
        df = df.sort_values(['Number'], ascending=[False])  # sort list
        df.to_csv(path, index=True)

        # Add synonym for top IDF
        remove_list = csv2list(
            "C:\\Users\\maisieb01\\Desktop\\PHD\\All Others\\Python All\\SLR_API\\SLR_API\\Remove_k.csv")

        df = pd.read_csv(path)
        df.set_index("Keyword", drop=False)

        # for i,row in df.iterrows():
        #     for i, row in
        if call_type == 'questions':
            # TFIDF_weight = 0.6
            TFIDF_weight = 0.5
        # -->change for short questions
        if call_type == 'seeds':
            TFIDF_weight = 0.5

        for i, row in df.iterrows():
            if float(df.values[i][3]) * 10 > TFIDF_weight:
                synonyms_list = []
                synonyms = ''
                for syn in wordnet.synsets(df.values[i][1]):
                    for l in syn.lemmas():
                        if l.name() not in synonyms_list and l.name().lower() not in remove_list:
                            synonyms_list.append(l.name())
                            if synonyms == '':
                                synonyms = l.name()
                            else:
                                synonyms = synonyms + ',' + l.name()
                df.loc[i, 'Synonyms'] = synonyms
                if call_type == 'questions':
                    df.loc[i, 'Reward'] = 1
                    df.loc[i, 'Demote'] = 0
                elif call_type == 'seeds':
                    df.loc[i, 'Reward'] = 1
                    df.loc[i, 'Demote'] = 0
        df = df.sort_values(['Number'], ascending=[False])  # sort list
        df.to_csv(path, index=False)

        # create alternative search query
        df = pd.read_csv(path)
        df.set_index("Keyword", drop=False)
        search_query = ''
        for i, row in df.iterrows():
            if float(df.values[i][3]) * 10 > TFIDF_weight:
                if search_query == '':
                    #  Adding synonym using OR
                    # search_query = '( {}'.format(df.values[i][1]) + '{}'.format(df.values[i][4].replace(","," OR ")) + ' )'
                    search_query = '( ' + '{}'.format(df.values[i][4].replace(",", " OR ")) + ' )'
                    # for i in range(len(list_syn)):
                    #     search_query = '{}'.format(search_query) +' OR '+'{}'.format(df.values[i][1])
                else:
                    #  Adding synonym using OR
                    if not df.isnull().values[i][4]:
                        search_query = '{}'.format(search_query) + ' AND ' + '( ' + '{}'.format(
                            df.values[i][4].replace(",", " OR ")) + ' )'
                    else:
                        search_query = '{}'.format(search_query) + ' AND ' + '( ' + '{}'.format(
                            df.values[i][1] + ' )')
                # list_syn = list(df.values[i][4].split(","))
                df.loc[i, 'search query'] = search_query
        df = df.sort_values(['Number'], ascending=[False])  # sort list

        for row in df.iterrows():
            row_index, row_values = row
            if float(row_values['TF_IDF']) * 10 > TFIDF_weight:
                db_controller.add_iteration_result(row_values['Keyword'], row_values['TF_IDF'], row_values['Synonyms'],
                                                   row_values['Reward'], row_values['Demote'],
                                                   row_values["search query"], "Radjenovic", call_type)
        df.to_csv(path, index=False)
    return json.dumps(sorted(Counter(words_vectors)))


def string_to_list(string):
    li = list(string.split(","))
    return li


# import embedded abstracts into DB
def import_abstract_embed(abstracts, embedding_model: str):
    import SLR_API.db_controller as db_controller
    # Clean abstracts
    if "abstracts" in abstracts:
        clean_abstracts = __vec_corpus(abstracts["abstracts"], "abstract")
        for item in clean_abstracts:
            abstract_words_vectors = []
            for keyword in item['keywords']:
                word_vector = __word2vec_model(keyword, embedding_model)
                if word_vector is not None:
                    abstract_words_vectors.append(word_vector)
                avg_abstract_vector = np.mean(abstract_words_vectors, axis=0)
            item.update({'vector': avg_abstract_vector.tolist()})
            # item["keywords"],item[""]
            item_keys = list(item.keys())
            # --> abstract_id,,abstract , slr_author, vec_keywords,vec_average,json_format,embedding_model,incl_excl
            db_controller.add_abstract(item[item_keys[0]]['AbstractID'], item[item_keys[0]]['Abstract_Year'],
                                       item[item_keys[0]]['Abstract'], item_keys[0],
                                       item['keywords'], item['vector'], json.dumps(item), embedding_model,
                                       item[item_keys[0]]['incl_excl'])
    return True


# create vectors of the given queries and fetch the similarity with the existing abstracts in the DB
def question_embed(questions, embedding_model: str):
    import SLR_API.db_controller as db_controller
    # pre-processing the questions
    if "questions" in questions:
        all_question_avg_vectors = []
        clean_questions = __vec_corpus(questions["queries"], "questions")
        for item in clean_questions:
            question_words_vectors = []
            for keyword in item['keywords']:
                word_vector = __word2vec_model(keyword, embedding_model)
                if word_vector is not None:
                    question_words_vectors.append(word_vector)
                avg_question_vector = np.mean(question_words_vectors, axis=0)
                item.update({'vector': avg_question_vector})
            all_question_avg_vectors.append(item)
        abstracts = db_controller.get_abstracts(embedding_model)
        return calculating_similarity_word2vec(abstracts, all_question_avg_vectors, "query", path="")


# enrich the given list of keywords
def enrich_keyword_set(keyword, embedding_model: str):
    word_vector = most_similar_model(keyword, embedding_model)
    return json.dumps(word_vector)


# abstracts vs seed abstracts
def seed_abstract_embed(seeds_abstracts, embedding_model: str, path: str):
    all_abstracts_avg_vectors = []
    all_seeds_avg_vectors = []
    # clean the seeds
    if "seeds" in seeds_abstracts:
        clean_seeds = __vec_corpus(seeds_abstracts["seeds"], "seeds")
        for item in clean_seeds:
            seeds_words_vectors = []
            for keyword in item['keywords']:
                word_vector = __word2vec_model(keyword, embedding_model)
                if word_vector is not None:
                    seeds_words_vectors.append(word_vector)
                avg_seeds_vector = np.mean(seeds_words_vectors, axis=0)
            item.update({'vector': avg_seeds_vector})
            all_seeds_avg_vectors.append(item)

    # Clean the abstracts
    if "abstracts" in seeds_abstracts:
        # --> cleaning abstracts
        clean_abstracts = __vec_corpus(seeds_abstracts["abstracts"], "abstract")
        for item in clean_abstracts:
            abstract_words_vectors = []
            for keyword in item['keywords']:
                # --> embedding each keyword within the list of keywords
                word_vector = __word2vec_model(keyword, embedding_model)
                if word_vector is not None:
                    abstract_words_vectors.append(word_vector)
                avg_abstract_vector = np.mean(abstract_words_vectors, axis=0)
            item.update({'vector': avg_abstract_vector})
            all_abstracts_avg_vectors.append(item)
    # --> calling calculating_similarity_word2vec to calculate the cosine similarity between abstracts and seeds
    return calculating_similarity_word2vec(all_abstracts_avg_vectors, all_seeds_avg_vectors, "seed", path)


def compute_tfidf(path: str):
    sentences = list()
    with open(path) as file:
        for line in file:
            for l in re.split(r"\.\s|\?\s|\!\s|\n", line):
                if l:
                    sentences.append(l)

    cvec = CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=0.1, max_df=0.7, max_features=100)
    sf = cvec.fit_transform(sentences)
    transformer = TfidfTransformer()
    transformed_weights = transformer.fit_transform(sf)
    weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
    print(weights_df.sort_values(by='weight', ascending=False).head(10))


# set up SVM kernel as
def svm_kernel(X, Y):
    return np.dot(X, Y.T)


@app.errorhandler(404)
def page_not_found(error):
    return 'This page does not exist', 404


if __name__ == '__main__':
    app.debug = True
    app.run()
