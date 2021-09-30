from flask import jsonify, request
from nltk import sent_tokenize
from numpy import dot
from numpy.linalg import norm
import math
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import SLR_API.def_controller as cont
import scipy.spatial as sp
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity as sim_cosine
from wordcloud import WordCloud



# generating wordcloud to see get a visual representation of most common words.
def __generate_wordcloud(corpus):
    _wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    _wordcloud.generate(corpus)
    return _wordcloud.to_image()


# extract sentences from abstracts
def __sent_token(text):
    sent_token = sent_tokenize(text)
    return sent_token


#
# TF = (Number of time the word occurs in the text) / (Total number of words in text)
# IDF = (Total number of documents / Number of documents with word t in it)
# TF-IDF = TF * IDF
# calculating tfidf
def _compute_tfidf(path):
    # sentences = list()
    # with open("C:\\Users\\maisieb01\\Desktop\\PHD\\Framework\\To GitHub\\TEST") as file:
    #     for line in file:
    #         for l in re.split(r"\.\s|\?\s|\!\s|\n", line):
    #             if l:
    #                 sentences.append(l)
    # cvec = CountVectorizer(stop_words='english', min_df=3, max_df=0.5, ngram_range=(1, 2))
    # sf = cvec.fit_transform(sentences)
    # transformer = TfidfTransformer()
    # transformed_weights = transformer.fit_transform(sf)
    # weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
    # weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
    # weights_df.sort_values(by='weight', ascending=False).head(10)

    sentences = list()
    with open(
            path) as file:
        for line in file:
            for l in re.split(r"\.\s|\?\s|\!\s|\n", line):
                if l:
                    sentences.append(l)

    # cvec = CountVectorizer(stop_words='english', min_df=1, max_df=0.6, ngram_range=(1,2))
    cvec = CountVectorizer(stop_words='english', ngram_range=(1, 3), min_df=0.1, max_df=0.7, max_features=100)
    sf = cvec.fit_transform(sentences)
    transformer = TfidfTransformer()
    transformed_weights = transformer.fit_transform(sf)
    weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
    print(weights_df.sort_values(by='weight', ascending=False).head(10))

    #
    # vectorizer = TfidfVectorizer(max_features=1500, stop_words=stopwords.words('english'))
    # # x = vectorizer.fit_transform([query])
    # # y = vectorizer.fit_transform([corpus])
    # vectors = vectorizer.fit_transform([query])  # , corpus])
    # feature_names = vectorizer.get_feature_names()
    # dense = vectors.todense()
    # denselist = dense.tolist()
    # df = pd.DataFrame(denselist, columns=feature_names)
    # # df.sort_values(ascending=False)
    # # corr_matrix = (vectors * vectors.T).A
    # corr = np.corrcoef(vectors)
    # weights_df.sort_values(by='weight', ascending=False).head(10)
    return weights_df.sort_values(by='weight', ascending=False).head(10)
    # __compute_cosine_similarity_tfidf(x,y)
    # df, corr_matrix


# cosine similarity for text comparisonusing tfidf
def __compute_cosine_similarity_tfidf(x, y):
    cosine_sim_matrix = sim_cosine(x, y)
    return cosine_sim_matrix


# cosine similarity for text comparison
def __compute_cosine_similarity(query, document):
    # Word2Vec
    avg_query_vector = []
    avg_document_vector = []
    query_vec = cont.corpus_to_keywords(query)
    document_vec = cont.corpus_to_keywords(document)
    document_words_vectors = []
    query_words_vectors = []
    for item in query_vec:
        word_vector = cont.__word2vec_model(item, 'glove')
        if word_vector is not None:
            query_words_vectors.append(word_vector)
            avg_query_vector = np.mean(query_words_vectors, axis=0)
    for item in document_vec:
        word_vector = cont.__word2vec_model(item, 'glove')
        if word_vector is not None:
            document_words_vectors.append(word_vector)
            avg_document_vector = np.mean(document_words_vectors, axis=0)
    cosine_sim = sp.distance.cosine(avg_query_vector, avg_document_vector)
    return cosine_sim


# universal cosine similarity for text format comparison
def __compute_universal_cosine_similarity(query, document):
    if type(query) is str:
        query = [query]
    if type(document) is str:
        document = [document]
    query_vec = __universal_embedding(query)
    document_vec = __universal_embedding(document)
    uni_cosine_sim = 1 - sp.distance.cosine(query_vec, document_vec)
    return uni_cosine_sim


# calculating Jacard similarity
# ToDo we can compare gold sentence from seed to given abstract sentences
def __compute_jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    if union != 0:
        return len(intersection) / len(union)
    else:
        return 0


# plotting heatmap similarity
def heatmap(x_labels, y_labels, values):
    fig, ax = plt.subplots()
    im = ax.imshow(values)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10,
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, "%.2f" % values[i, j],
                           ha="center", va="center", color="b",
                           fontsize=6)

    fig.tight_layout()
    plt.show()


# universal embedding sent to vec
def __universal_embedding(corpus):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.load(module_url)
    # embed = hub.Module(module_url)
    similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
    similarity_message_encodings = embed(similarity_input_placeholder)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        corpus_embeddings_ = session.run(similarity_message_encodings,
                                         feed_dict={
                                             similarity_input_placeholder: corpus})
        # corr = np.inner(corpus_embeddings_, corpus_embeddings_)
        # heatmap(__sent_token(abstract) + __sent_token(seed), __sent_token(abstract) + __sent_token(seed), corr)
    return corpus_embeddings_

#
# # calculate
# def n_similarity(a, b):
#     similarity_list = {}
#     sim = float
#     for a_item in a:
#         for b_item in b:
#             sim = dot(a_item, b_item) / (norm(a_item) * norm(b_item))
#             if math.isnan(sim):
#                 sim = 0
#             similarity_list['sentId'] = '{}'.format(a_item)
#             similarity_list['similarity'] = '{}'.format(dot(a_item, b_item) / (norm(a_item) * norm(b_item)))
#     return similarity_list
#
# abstract = "Service-oriented architecture (SOA) is a new architectural style for developing distributed business applications. Nowadays, those applications are realized through web services, which are later grouped as web service compositions. Web service compositions language, like the BPELWS 2.0 standard, are extensions of imperative programming languages. Additionally, it presents a challenge for traditional white-box testing, due to its inclusion of specific instructions, concurrency, fault compensation and dynamic service discovery and invocation. In fact, there is a lack of unit testing approaches and tools, which has resulted in inefficient practices in testing and debugging of automated business processes. Therefore, we performed a systematic review study to analyze 27 different studies for unit testing approaches for BPEL. This paper aims to focus on a comprehensive review to identify a categorization, a description of test case generation approaches, empirical evidence, current trends in BPEL studies, and finally to end with future work for other researchers. "
# keywords="Systematic literature reviewMapping studySoftware engineeringTertiary study."
# print("abstract {}".format(__universal_embedding(__sent_token(abstract))))
# # sim = dot(a, b) / (norm(a) * norm(b))
# # print(n_similarity(__universal_embedding(__sent_token(abstract)), __universal_embedding(__sent_token(seed))))
#
#
# def test(abstract11):
#     dic = {}
#     list_of_dict = []
#     dic_sentence={}
#     i = 1
#     dic_sentence = __sent_token(abstract11)
#     abs1 = list(dic_sentence.keys())[0]
#     for item in dic_sentence:
#         dic[abs1]["sentId"] = '{}'.format(i)
#         # dic["abstractId"] = '{}'.format('abs1')
#         dic[abs1]["sent"] = item
#         # dic["vector"] = __universal_embedding(item)
#         list_of_dict.append(dic.copy())
#         i = i + 1
#     return list_of_dict
#
#
# # # print(test(abstract))

#
