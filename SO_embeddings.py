# The official gensim docs provide further details and comprehensive documentation on how a word2vec model can be used for various NLP tasks.
# Microsoft Visual C++ 14.0 is required. Get it with "Microsoft Visual C++ Build Tools": https://visualstudio.microsoft.com/downloads
# The the pre-trained model is stored in a .bin file (of approximate size 1.5 GB) which can be accessed at this link: http://doi.org/10.5281/zenodo.1199620

from gensim.models.keyedvectors import KeyedVectors
word_vect = KeyedVectors.load_word2vec_format("SO_vectors_200.bin", binary=True)

# Examples of semantic similarity queries

words=['virus','java','mysql']
for w in words:
    try:
        print(word_vect.most_similar(w))
    except KeyError as e:
            print(e)    
        
print(word_vect.doesnt_match("java c++ python bash".split()))

# Examples of analogy queries
print(word_vect.most_similar(positive=['python', 'eclipse'], negative=['java']))        
        
