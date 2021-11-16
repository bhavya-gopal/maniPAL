# importing packages
import numpy as np
import nltk
# package that contains a pre-trained tokenizer
nltk.download('punkt')                      
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# create an array of words or tokens (word/character/number) from the sentence
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# find root/stem of the word, eg: ["eating", "eats", "eaten"] --> ["eat", "eat", "eat"]
def stem(word):
    
    return stemmer.stem(word.lower())

# encode as 1 if word exists in the sentence, 0 if not 
def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag