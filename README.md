# SLR Search Query Building Service

The main goals that drove the design and implementation of this platform were: i) allow third-parties to build support tools by integrating functionality offered by the proposed pipeline, ii) enable researchers to extend and adapt the implementation of the proposed techniques so as to facilitate experimentation, and iii) support the experimental evaluation of our propose pipeline. To deliver on these design goals, we implemented a flexible service-oriented platform that exposes the main functionality of the pipeline as services.

We designed the services in OpenAPI, for which we leveraged Swagger(https://swagger.io/), a hosted service that supports the definition and API development and documentation life-cycle. We used off-the-shelf Python 3.7 libraries for implementing these services. For all the text pre-processing steps (e.g., removing stop words), we used the Python library NLTK(Natural Language ToolKit (https://www.nltk.org/)). Tokenization were done using Stanford Core NLP 4.0(https://stanfordnlp.github.io/CoreNLP/).



  
  
  
 
