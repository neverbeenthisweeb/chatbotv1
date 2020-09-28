import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer


def tokenize(sentence: str) -> list:
    tokenized_sentence = nltk.word_tokenize(sentence, language="english")

    return tokenized_sentence


def stem(word: str) -> str:
    stemmed_word = PorterStemmer().stem(word.lower())

    return stemmed_word


def bag_of_words(tokenized_sentence: list, words: list) -> list:
    """
    Get a bag of words.

    Example:
    tokenized_sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag_of_words = [0, 1, 0, 1, 0, 0, 0]
    """
    # stem each words in tokenized sentence
    stemmed_sentence = [stem(w) for w in tokenized_sentence]

    # initialize a bag with 0 for each words
    bag_of_words = np.zeros(len(words), dtype=np.float)

    # fill in the bag respectively
    for idx, w in enumerate(words):
        if w in stemmed_sentence:
            bag_of_words[idx] = 1

    return bag_of_words
