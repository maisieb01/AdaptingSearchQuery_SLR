import mysql.connector
from mysql.connector import errorcode
import mysql.connector.errors
import json
import SLR_API.def_controller as em
from datetime import datetime

# import uuid
# import numpy as np

# import controllers.neo4j_controller as neo4j_ctrl

cnx = mysql.connector.connect(user='root', password='slradmin', host='localhost', database='slr_db')
cnx.ping(True)
curs = cnx.cursor()
DB_NAME = "slr_db"
curs.execute("USE {}".format(DB_NAME))


def create_embedded_abstracts_table():
    db_tables = {}
    db_tables['Embedded_Abstracts'] = (
        "CREATE TABLE `Embedded_Abstracts` ("
        "  `slr_id` int  NOT NULL AUTO_INCREMENT,"
        "  `abstract_id` varchar(36) NOT NULL,"
        "  `abstract_year` varchar(4) NOT NULL,"
        "  `abstract` LONGTEXT NOT NULL,"
        "  `slr_author` LONGTEXT NOT NULL,"
        "  `vec_keywords` LONGTEXT NOT NULL,"
        "  `vec_average` LONGTEXT NOT NULL,"
        "  `json_format` LONGTEXT NOT NULL,"
        "  `embedding_model` LONGTEXT NOT NULL,"
        "  `incl_excl` varchar(1) NOT NULL,"
        "  PRIMARY KEY (`slr_id`)"
        ") ENGINE=InnoDB")
    for table_name in db_tables:
        table_description = db_tables[table_name]
        try:
            print("Creating table {}: ".format(table_name), end='')
            curs.execute(table_description)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("already exists.")
            else:
                print(err.msg)
        else:
            print("The table is created.")


# create the database
def create_database(curs):
    try:
        curs.execute("CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format('slr_db'))
        create_embedded_abstracts_table()
    except mysql.connector.Error as err:
        print("Failed creating database: {}".format(err))
        exit(1)


# item[item_keys[0]]['AbstractID'],item[item_keys[0]]['Abstract'],item_keys[0],item['keywords'],item['vector'],json.dumps(item),embedding_model
def add_abstract(abstract_id, abstract_year, abstract, slr_author, vec_keywords, vec_average, json_format,
                 embedding_model, incl_excl):
    try:
        if incl_excl == 'yes':
            incl_excl = 1
        else:
            incl_excl = 0
        query = ("INSERT INTO slr_db.Embedded_Abstracts "
                 "(abstract_id, abstract_year, abstract, slr_author, vec_keywords, vec_average, json_format, embedding_model,incl_excl) "
                 "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)")
        abstract_data = (
            abstract_id, abstract_year, abstract, slr_author, ','.join(map(str, vec_keywords)),
            ','.join(map(str, vec_average)), json_format,
            embedding_model, incl_excl)
        curs.execute(query, abstract_data)
        cnx.commit()
    except mysql.connector.errors.DatabaseError as Err:
        print("An error occurred for inserting to table {} ".format(Err))




def add_iteration_result(term, tf_idf, synonym, reward, demote, search_query, review_name, corpus_type):
    try:
        query = ("INSERT INTO slr_db.iteration_result"
                 "(term,tf_idf,synonym,reward,demote,search_query,review_name,corpus_type,date_time)"
                 "VALUES (%s, %s, %s, %s, %s,%s, %s, %s, %s)")
        iteration_data = (term, tf_idf, synonym, reward, demote, search_query, review_name, corpus_type, datetime.now())
        curs.execute(query, iteration_data)
        cnx.commit()
    except mysql.connector.errors.DatabaseError as Err:
        print("An error occurred for inserting to table {} ".format(Err))


def add_idf(term, weight, iteration_id, review_name, corpus_type):
    try:
        query = ("INSERT INTO slr_db.tf_idf"
                 "(term,weight,iteration_id,review_name,corpus_type,date_time)"
                 "VALUES (%s, %s, %s, %s, %s,%s)")
        tf_idf_data = (term, weight, iteration_id, review_name, corpus_type, datetime.now())
        curs.execute(query, tf_idf_data)
        cnx.commit()
    except mysql.connector.errors.DatabaseError as Err:
        print("An error occurred for inserting to table {} ".format(Err))


def select_idf(review_name, corpus_type):
    try:
        query = ("SELECT term,weight,iteration_id,review_name,corpus_type From slr_db.tf_idf"
                 "WHERE review_name ='%s' and corpus_type ='%s'" % review_name % corpus_type)
        curs.execute(query)
        cnx.commit()
    except mysql.connector.errors.DatabaseError as Err:
        print("An error occurred for inserting to table {} ".format(Err))


def get_abstracts(embedding_model):
    try:

        embedding_model = em.convert_model(embedding_model)
        query = ("SELECT abstract_id ,json_format,embedding_model,vec_average,incl_excl FROM slr_db.Embedded_Abstracts "
                 "WHERE embedding_model = '%s' " % embedding_model)
        cnx = mysql.connector.connect(user='root', password='slradmin', host='localhost', database='slr_db')
        cnx.ping(True)
        curs = cnx.cursor()
        DB_NAME = "slr_db"
        curs.execute("USE {}".format(DB_NAME))
        curs.execute(query)
        parameters = []
        for row in curs:
            parameters.append(json.loads(row[1]))
        return parameters
    except mysql.connector.errors.DatabaseError as Err:
        cnx = mysql.connector.connect(user='root', password='slradmin')
        curs = cnx.cursor()
        DB_NAME = "slr_db"
        curs.execute("USE {}".format(DB_NAME))
        query = ("SELECT abstract_id ,json_format,embedding_model,vec_average FROM Embedded_Abstracts;")
        curs.execute(query)
        parameters = []
        for row in curs:
            parameters.append(json.loads(row[1]))
        return parameters


# delete all form a given table
def delete_all_from_table(table_name:str):
    try:
        query = ("delete from {};".format(table_name))
        cnx = mysql.connector.connect(user='root', password='slradmin', host='localhost', database='slr_db')
        cnx.ping(True)
        curs = cnx.cursor()
        DB_NAME = "slr_db"
        curs.execute("USE {}".format(DB_NAME))
        curs.execute(query)
        cnx.commit()
        for row in curs:
            return json.loads(row[1])
        return None
    except mysql.connector.errors.DatabaseError as Err:
        cnx = mysql.connector.connect(user='root', password='slradmin', host='localhost', database='SLR_DB')
        curs = cnx.cursor()
        DB_NAME = "slr_db"
        curs.execute("USE {}".format(DB_NAME))
        query = ("delete from {};".format(table_name))
        curs.execute(query)
        cnx.commit()
        for row in curs:
            return json.loads(row[1])
        return None


# drop the tables if needed
def drop_table(table_name: str):
    try:
        query = ("Drop table {};".format(table_name))
        cnx = mysql.connector.connect(user='root', password='slradmin', host='localhost', database='SLR_DB')
        cnx.ping(True)
        curs = cnx.cursor()
        DB_NAME = "slr_db"
        curs.execute("USE {}".format(DB_NAME))
        curs.execute(query)
        cnx.commit()
        for row in curs:
            return json.loads(row[1])
        return None

    except mysql.connector.errors.DatabaseError as Err:
        cnx = mysql.connector.connect(user='root', password='slradmin', host='localhost', database='slr_db')
        curs = cnx.cursor()
        DB_NAME = "slr_db"
        curs.execute("USE {}".format(DB_NAME))
        query = ("drop table {};".format(table_name))
        curs.execute(query)
        cnx.commit()
        for row in curs:
            return json.loads(row[1])
        return None

# select abstracts from abstract tables
# def get_abstracts(embedding_model, slr_author):
#     try:
#
#         embedding_model = em.convert_model(embedding_model)
#         query = ("SELECT abstract_id ,json_format,embedding_model,vec_average,incl_excl FROM slr_db.Embedded_Abstracts "
#                  "WHERE embedding_model = '%s' AND slr_author='%s'" % (embedding_model, slr_author))
#         cnx = mysql.connector.connect(user='root', password='slradmin', host='localhost', database='slr_db')
#         cnx.ping(True)
#         curs = cnx.cursor()
#         DB_NAME = "slr_db"
#         curs.execute("USE {}".format(DB_NAME))
#         curs.execute(query)
#         parameters = []
#         for row in curs:
#             parameters.append(json.loads(row[1]))
#         return parameters
#     except mysql.connector.errors.DatabaseError as Err:
#         cnx = mysql.connector.connect(user='root', password='slradmin')
#         curs = cnx.cursor()
#         DB_NAME = "slr_db"
#         curs.execute("USE {}".format(DB_NAME))
#         query = ("SELECT abstract_id ,json_format,embedding_model,vec_average FROM Embedded_Abstracts;")
#         curs.execute(query)
#         parameters = []
#         for row in curs:
#             parameters.append(json.loads(row[1]))
#         return parameters
#
#
# (`iteration_id`,
# `term`,
# `synonym`,
# `reward`,
# `demote`,
# `search_query`,
# `tf_idf`,
# `review_Id`,
# `review_name`)    
    
#
# dataset2 = get_abstracts("glove", "SLR_Kitchenham")
# # print(dataset2[0]["vector"])
#
# for d in dataset2:
#     print(d['SLR_Kitchenham']['incl_excl'])

# create db objects
# create_embedded_abstracts_table()
# drop_table("Embedded_Abstracts")
# delete_all_from_table("Embedded_Abstracts")
# get_abstracts("glove")
# delete_all_from_table("slr_db.embedded_abstracts")
