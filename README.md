# SLR Search Query Building Service

The main goals that drove the design and implementation of this platform were: i) allow third-parties to build support tools by integrating functionality offered by the proposed pipeline, ii) enable researchers to extend and adapt the implementation of the proposed techniques so as to facilitate experimentation, and iii) support the experimental evaluation of our propose pipeline. To deliver on these design goals, we implemented a flexible service-oriented platform that exposes the main functionality of the pipeline as services.

We designed the services in OpenAPI, for which we leveraged Swagger\footnote{https://swagger.io/}, a hosted service that supports the definition and API development and documentation life-cycle. We used off-the-shelf Python 3.7 libraries for implementing these services. For all the text pre-processing steps (e.g., removing stop words), we used the Python library NLTK\footnote{Natural Language ToolKit (https://www.nltk.org/)}. Tokenization were done using Stanford Core NLP 4.0\footnote{https://stanfordnlp.github.io/CoreNLP/}.


There are three main files in this project. 

**Server.Py** : This file includes the endpoints for swagger API ,

@app.route("/searchquery", methods=["POST"]) : 
# This endpoint is used for building search queries of given seeds and the result is saved into a csv file
@app.route("/callingAPI/IEEEXPLORE", methods=["POST"]) :
# This endpoint is used to execute boolean queries by calling digital libraries APIs and retrieves the query results into a csv file

@app.route("/similarity/queries", methods=["POST"]) : 
# This endpoint generates vectors for given search queries based on the selected embedding model and return the similarity result of the given queries(QPPs)
@app.route("/similarity/query/abstracts", methods=["POST"]) : 
# This endpoint generates vectors for given search queries and abstracts based on the selected embedding model and return the similarity result of the given inputs (QPPs)
@app.route("/similarity/query/abstracts", methods=["POST"]) :
# This endpoint generates vectors for given relevant asbtracts and retrievd abstracts based on the selected embedding model and return the similarity result of the given inputs(QPPs)
@app.route("/import/abstracts", methods=["POST"])
# This endpoint recieved labeled abstracts (relevant/irrelevant), to then extrct keywords from it and update the search terms rewards and demotes in Mysql database.

there are additional endpoints that we used during our studies.

**def_controller.py** includes all the definittions  
1. 



db_controller.py

  
  
  
 
