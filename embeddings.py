import warnings
warnings.filterwarnings("ignore")
import gensim
import pickle
import os
import numpy as np
import gc
from consts import *

def extract_embedding_weights(tokenizer):
    if tokenizer is None:
      raise ValueError('Please provide a pre-training tokenizer')
    # prepare embedding matrix
    word_index = tokenizer.word_index
    num_words = len(word_index) + 1
    gc.collect()
    pretrained_embeddings = load_pretrained_embedding()
    gc.collect()
    embedding_matrix = prepare_embedding_matrix(num_words, EMBEDDING_DIM, word_index, pretrained_embeddings)
    gc.collect()
    return embedding_matrix

def prepare_embedding_matrix(num_of_words, embedding_dim, word_index, pretrained_embeddings):
    embedding_matrix = np.zeros((num_of_words, embedding_dim))
    unknown_tag = "unk_embedding"
    for word, i in word_index.items():
        embedding_vector = pretrained_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = pretrained_embeddings.get(unknown_tag)
    return embedding_matrix

def load_pretrained_embedding():
    local_pickle_file = os.path.join(DOC2VEC_MODELS_PATHS, 'glove_embeddings.pickle')
    if os.path.exists(local_pickle_file):
        print("Loading glove_embeddings")
        with open(local_pickle_file, 'rb') as handle:
            embedding =  pickle.load(handle)
    else:
        with open(os.path.join(GLOVE_DIR, f'glove.6B.{EMBEDDING_DIM}.txt'), 'r') as f:
            lines = f.readlines()
            split_lines = map(lambda line: line.split(), lines)
            embedding = {parts[0]: np.asarray(parts[1:], dtype='float32') for parts in split_lines}
        # The pre-trained vectors do not have an unknown token, and currently the code just ignores out-of-vocabulary words when producing the co-occurrence counts.
        unk_embedding = np.random.uniform(low=-0.04, high=0.04, size=(EMBEDDING_DIM,))
        embedding["unk_embedding"] = unk_embedding
    return embedding