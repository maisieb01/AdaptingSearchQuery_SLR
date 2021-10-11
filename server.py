# **Server.Py** : This file includes the endpoints for swagger API ,

# @app.route("/searchquery", methods=["POST"]) : 
# # This endpoint is used for building search queries of given seeds and the result is saved into a csv file
# @app.route("/callingAPI/IEEEXPLORE", methods=["POST"]) :
# # This endpoint is used to execute boolean queries by calling digital libraries APIs and retrieves the query results into a csv file

# @app.route("/similarity/queries", methods=["POST"]) : 
# # This endpoint generates vectors for given search queries based on the selected embedding model and return the similarity result of the given queries(QPPs)
# @app.route("/similarity/query/abstracts", methods=["POST"]) : 
# # This endpoint generates vectors for given search queries and abstracts based on the selected embedding model and return the similarity result of the given inputs (QPPs)
# @app.route("/similarity/query/abstracts", methods=["POST"]) :
# # This endpoint generates vectors for given relevant asbtracts and retrievd abstracts based on the selected embedding model and return the similarity result of the given inputs(QPPs)
# @app.route("/import/abstracts", methods=["POST"])
# # This endpoint recieved labeled abstracts (relevant/irrelevant), to then extrct keywords from it and update the search terms rewards and demotes in Mysql database.

# there are additional endpoints that we used during our studies.



import csv
import json
from flask import Flask, jsonify, request, send_file, flash, redirect, render_template, Response
import requests
import SLR_API.convert_csv_to_JSON as csv_converter
import SLR_API.def_controller as def_controller
import SLR_API.service as slr_service
import flask_swagger_ui
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd
import re

# from swaggerui_blueprint import

app = Flask(__name__)
# app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
CORS(app)

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config["ALLOWED_FILE_EXTENSION"] = ["CSV", "TXT"]

SWAGGER_URL = '/docs'  # URL for exposing Swagger UI (without trailing '/')
API_URL = 'http://localhost:5000/doc'  # Our API url (can of course be a local resource)

swaggerui_blueprint = flask_swagger_ui.get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "SLR API"
    },
)

# Register blueprint at URL
# (URL must match the one given to factory function above)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


# creating a list of the abstracts
def csv2list(v_filename):
    corpus_list = []
    with open(v_filename, encoding='utf-8', errors='ignore') as corpus_set:
        row_reader = csv.reader(corpus_set)
        corpus_list = [record for record in row_reader]
        corpus_list = [''.join(record) for record in corpus_list]  ## change the list within the list to string
    return corpus_list


# checking the format of the given file
def allowed_file(filename):
    if "." not in filename:
        return False
    ext = filename.rsplit('.', 1)[1]
    if ext.upper() in app.config["ALLOWED_FILE_EXTENSION"]:
        return True
    else:
        return False


# TF-IDF weights
@app.route("/extract/TF_IDF", methods=["POST"])
def extract_TF_IDF():
    if request.method == "POST":
        if request.files:
            info_dict = dict()
            for filename, value in request.files.items():
                if value.filename == "":
                    return Response("The selected file is empty", 400, content_type="application/json")
                if not allowed_file(value.filename):
                    return Response("The selected file is not in the right format (.csv)", 400,
                                    content_type="application/json")
                else:
                    if filename == "abstracts":
                        json_value = csv_converter.convert(value, "abstracts")
                    elif filename == "questions":
                        json_value = csv_converter.convert(value, "questions")
    return Response(def_controller.compute_tfidf(request.form["path"]), content_type="application/json")


# extract keywords from RQ and objective and seeds building search query
@app.route("/searchquery", methods=["POST"])
def search_query():
    if request.method == "POST":
        if request.files:
            info_dict = dict()
            for filename, value in request.files.items():
                if value.filename == "":
                    return Response("The selected file is empty", 400, content_type="application/json")
                if not allowed_file(value.filename):
                    return Response("The selected file is not in the right format (.csv)", 400,
                                    content_type="application/json")
                else:
                    if filename == "abstracts":
                        json_value = csv_converter.convert(value, "abstracts")
                    elif filename == "questions":
                        # with open(value, newline='') as f_q:
                        #     reader = csv.reader(f_q)
                        #     next(reader, None)  # skip the headers
                        #     combine_RQs = ''
                        #     for row in reader:
                        #         combine_RQs = combine_RQs + ' ' + row[1]
                        json_value = csv_converter.convert(value, "questions")
                        # df=def_controller.TF_IDF(value)
                    elif filename == "objectives":
                        json_value = csv_converter.convert(value, "objectives")
                    elif filename == "seed_abstracts":
                        json_value = csv_converter.convert(value, "seeds")
                    info_dict.update({filename: json_value})

            call_type = ''
            if 'abstracts' in info_dict.keys() and 'questions' in info_dict.keys():
                call_type = 'abstract'

            elif 'objectives' in info_dict.keys() and 'questions' in info_dict.keys():
                call_type = 'objective'

            elif 'seed_abstracts' in info_dict.keys():
                call_type = 'seeds'

            elif 'questions' in info_dict.keys():
                call_type = 'questions'

            embedding_model = {"embedding_model": 'Wikipedia'}
            return Response(def_controller.form_search_query(info_dict, embedding_model, request.form["path"],
                                                             call_type), content_type="application/json")
        # return Response(def_controller.form_search_query(info_dict, values, request.form["path"],request.values["inclusion_year_from"],
        #                                                 request.values["inclusion_year_to"],request.values["content_type"]),


# uploading files for Questions and abstracts
@app.route("/callingAPI/IEEEXPLORE", methods=["POST"])
def call_IEEE_API():
    if request.method == "POST":
        if request.files:
            info_dict = dict()
            for filename, value in request.files.items():
                if value.filename == "":
                    return Response("The selected file is empty", 400, content_type="application/json")
                if not allowed_file(value.filename):
                    return Response("The selected file is not in the right format (.csv)", 400,
                                    content_type="application/json")
                else:
                    if filename == "search_query":
                        json_value = csv_converter.convert(value, "query")
                # info_dict.update({filename: json_value})

        return Response(
            def_controller.IEEE_Xplore(json_value[0]['1'], request.form["path"], request.values["inclusion_year_from"],
                                       request.values["inclusion_year_to"]), content_type="application/json")


# uploading files for Questions and abstracts
@app.route("/similarity/queries", methods=["POST"])
def compare_quries():
    if request.method == "POST":
        if request.files:
            info_dict = dict()
            for filename, value in request.files.items():
                if value.filename == "":
                    return Response("The selected file is empty", 400, content_type="application/json")
                if not allowed_file(value.filename) and request.form["type"] != "streamlit":
                    return Response("The selected file is not in the right format (.csv)", 400,
                                    content_type="application/json")
                else:
                    if filename == "query":
                        json_value = csv_converter.convert(value, "query")
                    else:
                        json_value = csv_converter.convert(value, "slr")
                    info_dict.update({filename: json_value})

            if 'query' in info_dict.keys():
                # TODO add function here
                Response(def_controller.query_embed(info_dict, request.form["embedding_model"],
                                                    request.form["path"]), content_type="application/json")

                return "Files are saved in the provided path."


# uploading files for Questions and abstracts
@app.route("/similarity/query/abstracts", methods=["POST"])
def upload_file():
    if request.method == "POST":
        if request.files:
            info_dict = dict()
            for filename, value in request.files.items():
                if value.filename == "":
                    return Response("The selected file is empty", 400, content_type="application/json")

                if not allowed_file(value.filename) and request.form["type"] != "streamlit":
                    return Response("The selected file is not in the right format (.csv)", 400,
                                    content_type="application/json")
                else:
                    if filename == "abstracts":
                        json_value = csv_converter.convert(value, "abstracts")
                    elif filename == "query":
                        json_value = csv_converter.convert(value, "questions")
                    elif filename == "seeds":
                        json_value = csv_converter.convert(value, "seeds")
                    info_dict.update({filename: json_value})

            if 'abstracts' in info_dict.keys() and 'questions' in info_dict.keys() and 'seeds' in info_dict.keys():
                # TODO add function here
                Response(def_controller.question_abstract_embed(info_dict, request.form["embedding_model"],
                                                                request.form["path"]), content_type="application/json")

                Response(def_controller.seed_abstract_embed(info_dict, request.form["embedding_model"],
                                                            request.form["path"]), content_type="application/json")
                return "Files are saved in the provided path."

            if 'abstracts' in info_dict.keys() and 'query' in info_dict.keys() and 'seeds' not in info_dict.keys():
                # TODO add function here
                return Response(def_controller.question_abstract_embed(info_dict, request.form["embedding_model"],
                                                                       request.form["path"]),
                                content_type="application/json")


# uploading files for Questions and abstracts

# This API generates vectors for given research questions(RQs) based on the selected embedding model and return the
# similarity result of the RQs embedded abstracts from database with the same embedding model.
# This service can be used when the search is done and the research question
# definition is changing  with using this service the best RQs would be define based on search keyword"
# endpoint to import the abstract with the embedded elements in the database
@app.route("/import/abstracts", methods=["POST"])
def import_abstract():
    if request.method == "POST":
        if request.files:
            info_dict = dict()
            for filename, value in request.files.items():
                if value.filename == "":
                    return Response("The selected file is empty", 400, content_type="application/json")
                if not allowed_file(value.filename):
                    return Response("The selected file is not in the right format (.csv)", 400,
                                    content_type="application/json")
                else:
                    if filename == "abstracts":
                        json_value = csv_converter.convert(value, "abstracts")
                    # file = value.read()
                    info_dict.update({filename: json_value})

            if 'abstracts' in info_dict.keys():
                return Response(def_controller.import_abstract_embed(info_dict, request.form["embedding_model"]),
                                content_type="application/json")


# endpoint to keep the embedded abstracts into API storage for future reference
@app.route("/similarity/questions", methods=["POST"])
def embedding_question():
    if request.method == "POST":
        if request.files:
            info_dict = dict()
            for filename, value in request.files.items():
                if value.filename == "":
                    return Response("The selected file is empty", 400, content_type="application/json")
                if not allowed_file(value.filename):
                    return Response("The selected file is not in the right format (.csv)", 400,
                                    content_type="application/json")
                else:
                    if filename == "questions":
                        json_value = csv_converter.convert(value, "questions")
                    # file = value.read()
                    info_dict.update({filename: json_value})
            if 'questions' in info_dict.keys():
                # TODO add function here
                # result = def_controller.question_embed(info_dict, request.form["embedding_model"])
                return Response(def_controller.question_embed(info_dict, request.form["embedding_model"]),
                                content_type="application/json")


# uploading similarity seeds and abstracts
@app.route("/similarity/seeds/abstracts", methods=["POST"])
def seed_abstract():
    if request.method == "POST":
        if request.files:
            info_dict = dict()
            for filename, value in request.files.items():
                if value.filename == "":
                    return Response("The selected file is empty", 400, content_type="application/json")
                if not allowed_file(value.filename):
                    return Response("The selected file is not in the right format (.csv)", 400,
                                    content_type="application/json")
                else:
                    if filename == "abstracts":
                        json_value = csv_converter.convert(value, "abstracts")
                    else:
                        json_value = csv_converter.convert(value, "seeds")
                    info_dict.update({filename: json_value})

            if 'abstracts' in info_dict.keys() and 'seeds' in info_dict.keys():
                # TODO add function here
                return Response(
                    def_controller.seed_abstract_embed(info_dict, request.form["embedding_model"],
                                                       request.form["path"]), content_type="application/json")


# jacard similarity
@app.route("/similarity/compare/models", methods=["POST"])
def j_similarity():
    if request.method == "POST":
        if request.files:
            for filename, value in request.files.items():
                if value.filename == "":
                    return Response("The selected file is empty", 400, content_type="application/json")
                if not allowed_file(value.filename):
                    return Response("The selected file is not in the right format", 400,
                                    content_type="application/json")
                else:
                    if filename == "query":
                        f = request.files['query']
                        q_txt = f.read()
                    else:
                        f = request.files['corpus']
                        c_txt = f.read()

        return Response(
            'jaccard :{} '.format(
                slr_service.__compute_jaccard_similarity(c_txt, q_txt)) + "\n"
                                                                          'Universal :{} '.format(
                slr_service.__compute_universal_cosine_similarity(c_txt.decode('ascii'), q_txt.decode('ascii'))) + "\n"
                                                                                                                   'Glove :{}'.format(
                slr_service.__compute_cosine_similarity(c_txt.decode('ascii'), q_txt.decode('ascii'))) + "\n"
                                                                                                         'tf-idf: {}'.format(
                slr_service.__compute_tfidf(c_txt.decode('ascii'), q_txt.decode('ascii'))),
            content_type="application/text")


# enrich the keywords
@app.route("/enrich/keywords", methods=["POST"])
def enrich_keywords():
    if request.method == "POST":
        if request.files:
            info_dict = dict()
            for filename, value in request.files.items():
                if value.filename == "":
                    return Response("The selected file is empty", 400, content_type="application/json")
                if not allowed_file(value.filename):
                    return Response("The selected file is not in the right format (.csv)", 400,
                                    content_type="application/json")
                else:
                    if filename.upper() == "KEYWORDS":
                        json_value = csv_converter.convert(value, "keywords")
                    info_dict.update({filename: json_value})
            # for item in request.form['keyword']:
            # info_dict.append(def_controller.most_similar_model(item, request.form["embedding_model"]))
    return Response(def_controller.enrich_keyword_set(info_dict, request.form["embedding_model"]),
                    content_type="application/json")


# A visual representation of most common words.
@app.route("/wordcloud", methods=["POST"])
def wordclouding():
    if request.method == "POST":
        if request.files:
            for filename, value in request.files.items():
                if value.filename == "":
                    return Response("The selected file is empty", 400, content_type="application/json")
                if not allowed_file(value.filename):
                    return Response("The selected file is not in the right format", 400,
                                    content_type="application/json")
                else:
                    if filename == "query":
                        f = request.files['query']
                        q_txt = f.read()
                    else:
                        f = request.files['corpus']
                        c_txt = f.read()

        return Response(slr_service.__generate_wordcloud(c_txt))


# classification
@app.route("/classification/abstracts", methods=["POST"])
def svm_class():
    if request.method == "POST":
        if request.files:
            info_dict = dict()
            for filename, value in request.files.items():
                if value.filename == "":
                    return Response("The selected file is empty", 400, content_type="application/json")
                if not allowed_file(value.filename):
                    return Response("The selected file is not in the right format (.csv)", 400,
                                    content_type="application/json")
                else:
                    if filename == "abstracts":
                        json_value = csv_converter.convert(value, "abstracts")
                    else:
                        json_value = csv_converter.convert(value, "questions")
                    info_dict.update({filename: json_value})

            if 'abstracts' in info_dict.keys() and 'questions' in info_dict.keys():
                # TODO add function here
                return Response(def_controller.question_abstract_embed(info_dict, request.form["embedding_model"]),
                                content_type="application/json")


# swagger
@app.route("/doc/", methods=["GET"])
def swagger():
    s_file = open("swaggers/swaggerspec_new.yaml", "r")
    s_doc = s_file.read()
    s_file.close()
    return s_doc


if __name__ == '__main__':
    app.debug = True
    app.run()
    # app.run(host='10.35.10.75')
