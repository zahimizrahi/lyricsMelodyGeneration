import datetime
import math
import warnings

warnings.filterwarnings("ignore")
import keras.layers as KL
from keras.initializers import Constant
from keras import Sequential
from keras.layers import CuDNNLSTM as LSTM
from keras.layers import CuDNNGRU as GRU
import keras.backend as K
import numpy as np
from keras.backend import epsilon
from keras_layer_normalization import LayerNormalization
from keras import regularizers
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

# from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
EMBEDDING_DIM = 300
INPUT_LENGTH = 1
MELODY_VEC_LENGTH = 150
MELODY_CNN_VEC_LENGTH = 128


class LyricsMelodyModel:
    def __init__(self, tokenizer, embedding_matrix,
                 rnn_units=50,
                 bidirectional=True,
                 rnn_type='lstm',
                 dropout=0.3,
                 batch_size=32,
                 epochs=10,
                 optimizer=None,
                 show_summary=True,
                 shuffle=True,
                 verbose=False,
                 patience=3,
                 train_embedding=True,
                 is_layer_norm=True):

        self.rnn_type = rnn_type.lower()

        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = 'adam'
        self.shuffle = shuffle
        self.verbose = verbose
        self.patience = patience

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        num_words = len(tokenizer.word_index) + 1
        embedding_layer = KL.Embedding(num_words,
                                       EMBEDDING_DIM,
                                       embeddings_initializer=Constant(embedding_matrix),
                                       input_length=INPUT_LENGTH,
                                       trainable=train_embedding)
        lyrics_input = KL.Input(shape=(INPUT_LENGTH,))
        melody_input = KL.Input(shape=(MELODY_VEC_LENGTH,))

        lyrics = embedding_layer(lyrics_input)
        lyrics = KL.Flatten()(lyrics)
        if dropout > 0:
            lyrics = KL.Dropout(dropout)(lyrics)
            melody = KL.Dropout(dropout)(melody_input)
            combined = KL.Concatenate()([lyrics, melody])
        else:
            combined = KL.Concatenate()([lyrics, melody_input])
        combined = KL.Reshape((1, EMBEDDING_DIM + MELODY_VEC_LENGTH))(combined)

        # combined = rnn_type(rnn_units)(combined)
        if self.rnn_type == 'lstm':
            combined = KL.LSTM(rnn_units)(combined)
        elif self.rnn_type == 'gru':
            combined = KL.GRU(rnn_units)(combined)
        if bidirectional:
            combined = KL.Bidirectional(combined)
        if is_layer_norm:
            combined = LayerNormalization()(combined)
        combined = KL.Dense(num_words, kernel_regularizer=regularizers.l2(0.1), activation='softmax')(combined)
        model = Model(inputs=[lyrics_input, melody_input], outputs=[combined])

        if show_summary:
            model.summary()

        run_time = datetime.datetime.now().strftime('%m%d-%H%M%S')
        run_name = f'{run_time}_b{self.batch_size}_epochs{self.epochs}'

        self.callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=self.patience, verbose=self.verbose),
            ModelCheckpoint(f'{run_name}.h5', verbose=0, save_best_only=True, save_weights_only=True)]

        self.model = model
        self.tokenizer = tokenizer

    def train(self, X, y, validation_split=0.1):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network
        return self.model.fit(X, y,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  verbose=self.verbose,
                  shuffle=self.shuffle,
                  validation_split=validation_split,
                  callbacks=self.callbacks
                  )

    def predict(self, first_word, song, n_words):
        in_text, result = first_word, first_word
        # generate a fixed number of words
        for _ in range(n_words):
            # encode the text as integer
            encoded = self.tokenizer.texts_to_sequences([in_text])[0]
            encoded = np.array(encoded)

            words_probs = self.model.predict([[encoded], [song]], verbose=0)[0]

            # get 2 arrays of probs and word_tokens
            words_probs_enu = list(enumerate(words_probs))
            words_probs_sorted = sorted(words_probs_enu, key=lambda x: x[1],
                                        reverse=True)  # sorting in descending order
            words_tokens, words_probs = list(zip(*words_probs_sorted))
            # normalizre to sum 1
            words_probs = np.array(words_probs, dtype=np.float64)
            words_probs /= words_probs.sum().astype(np.float64)
            word_token = np.random.choice(words_tokens, p=words_probs)

            # map predicted word index to word
            out_word = get_word(word_token, self.tokenizer)
            # append to input
            in_text, result = out_word, result + ' ' + out_word
        return result

def get_encoded(text, tokenizer):
    encoded = tokenizer.texts_to_sequences([text])[0]
    encoded = np.array(encoded)
    return encoded


def get_word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word


def _perplexity(y_true, y_pred):
    cross_entropy = categorical_crossentropy(y_true, y_pred)
    perplexity = pow(2.0, cross_entropy)
    return perplexity
