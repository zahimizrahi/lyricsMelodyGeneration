import warnings
warnings.filterwarnings("ignore")
from data_processor import load_tokenized_data
import gensim
import pickle
import os
import numpy as np
import gc
from consts import *

EMBEDDING_DIM = 300
GLOVE_DIR = 'glove_pretrained/'

def extract_embedding_weights(tokenizer = None):
    if tokenizer is None:
      X, y, tokenizer = load_tokenized_data2()
    # prepare embedding matrix
    word_index = tokenizer.word_index
    num_words = len(word_index) + 1
    gc.collect()
    pretrained_embeddings = load_pretrained_embedding2()
    gc.collect()
    embedding_matrix, not_found = prepare_embedding_matrix2(num_words, EMBEDDING_DIM, word_index, pretrained_embeddings)
    gc.collect()
    return embedding_matrix

def prepare_embedding_matrix(num_of_words, embedding_dim, word_index, pretrained_embeddings):
    embedding_matrix = np.zeros((num_of_words, embedding_dim))
    not_found = []
    for word, i in word_index.items():
        word_encode = word.encode()
        embedding_vector = pretrained_embeddings.get(word_encode)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            not_found.append(word)
    return embedding_matrix, not_found

def load_pretrained_embedding():
    local_pickle_file = os.path.join(DOC2VEC_MODELS_PATHS, 'glove_embeddings.pickle')
    if os.path.exists(local_pickle_file):
        print("Loading glove_embeddings")
        with open(local_pickle_file, 'rb') as handle:
            embeddings_index = pickle.load(handle)
    else:
        embeddings_index = {}
        with open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), 'rb') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embeddings_index[word] = coefs
    return embeddings_index