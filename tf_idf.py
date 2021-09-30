# Subsequently, the system ranks tokens based on their TF-IDF # weights and iteratively shows tokens to an analyst to adapt a rule.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd
import re

sentences = list()

with open(
        "C:\\Users\\maisieb01\\Desktop\\PHD\\Framework\\To GitHub\\Experiments\\Original\\Wohono\\Reported on the paper\RQ\\researchQuestion2.csv") as file:
    for line in file:
        for l in re.split(r"\.\s|\?\s|\!\s|\n", line):
            if l:
                sentences.append(l)

# cvec = CountVectorizer(stop_words='english', min_df=1, max_df=0.6, ngram_range=(1,2))
cvec = CountVectorizer(stop_words='english', ngram_range=(1, 1), min_df=0.3, max_df=0.7, max_features=100)
sf = cvec.fit_transform(sentences)
transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
weights_df= weights_df.sort_values(by='weight', ascending=False)
print(weights_df)
