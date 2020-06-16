import warnings
warnings.filterwarnings("ignore")
from dataset import load_tokenized_data
import gensim
import pickle
import os
import numpy as np
import gc

EMBEDDING_DIM = 300
GLOVE_DIR = 'glove_pretrained/'

def extract_embedding_weights():
    X, y, tokenizer = load_tokenized_data()
    # prepare embedding matrix
    word_index = tokenizer.word_index
    num_words = len(word_index) + 1
    gc.collect()
    print("stage1 -Done")
    pretrained_embeddings = load_pretrained_embedding()
    gc.collect()
    print("stage2 -Done")
    embedding_matrix, not_found = prepare_embedding_matrix(num_words, EMBEDDING_DIM, word_index, pretrained_embeddings)
    gc.collect()
    print("stage3 -Done")
    return embedding_matrix

def prepare_embedding_matrix(num_of_words, embedding_dim, word_index, pretrained_embeddings):
    embedding_matrix = np.zeros((num_of_words, embedding_dim))
    not_found = []
    for word, i in word_index.items():  #TODO: check also word in capitlal (for word2vec)
        word_encode = word.encode()
        embedding_vector = pretrained_embeddings.get(word_encode)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            not_found.append(word)  #TODO: solve unknown word in pretrained_embeddings (words with ')
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

#### our model
def load_embedding(self, filename='glove_pretrained/glove.6B.300d.txt'):
    """
    loads the embedding as dictionary of words vectors. Keys are distinct words
    and values are the word vectors.
    :param filename - a .txt file of the pretrained word2vec embedding vectors
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    split_lines = map(lambda line: line.split(), lines)

    embedding = {parts[0]: np.asarray(parts[1:], dtype='float32') for parts in split_lines}

    return embedding