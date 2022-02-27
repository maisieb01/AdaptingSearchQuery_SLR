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
        
