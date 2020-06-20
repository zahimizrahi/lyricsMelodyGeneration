import warnings
warnings.filterwarnings("ignore")
import datetime
import math
import tensorflow as tf
if tf.__version__[0] == '2':
  from tensorflow.keras.layers import Embedding,  Bidirectional, Dense, LSTM, GRU, Dropout, Flatten, Input, Concatenate, Reshape
else:
  from tensorflow.keras.layers import CuDNNLSTM as LSTM
  from tensorflow.keras.layers import CuDNNGRU as GRU
from tensorflow.keras.initializers import Constant
from tensorflow.keras.backend import epsilon, pow, categorical_crossentropy
import numpy as np
from keras_layer_normalization import LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
EMBEDDING_DIM = 300
INPUT_LENGTH = 1
MELODY_VEC_LENGTH = 150
MELODY_CNN_VEC_LENGTH = 128

def word2idx(text, tokenizer):
    # word2idx("the food".split(), tokenizer)
    encoded = tokenizer.texts_to_sequences(text)[0]
    encoded = np.array(encoded)
    return encoded

def idx2word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word

class LyricsModel:
    def __init__(self,
                 tokenizer,
                 bidirectional=False,
                 rnn_units=50,
                 rnn_type='lstm',
                 dropout=0,
                 log_dir='logs/',
                 kernel_regularizer=0.01,
                 batch_size=128,
                 epochs=3,
                 lr=1e-2,
                 validation_split=0.2, pad=None,
                 loss='sparse_categorical_crossentropy', optimizer=None, embedding=None, metrics=['accuracy'],
                 patience=3, prefix='', verbose=1, embedding_dim=300, shuffle=True,
                 weight_matrix=None, vocabulary_size=None, show_summary=True):

        rnn_types = {
            'lstm': LSTM,
            'gru': GRU,
        }
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.rnn_type = rnn_types[rnn_type]
        self.verbose = verbose
        self.shuffle = shuffle
        if optimizer is None:
            self.optimizer = 'adam'
        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        self.EMBEDDING_DIM = embedding_dim
        self.vocabulary_size = len(tokenizer.word_index) + 1 #vocabulary_size
        self.weight_matrix = weight_matrix

        embedding_layer = Embedding(input_dim=self.vocabulary_size, output_dim=self.EMBEDDING_DIM,
                                    weights=[self.weight_matrix])
        model = Sequential()
        model.add(embedding_layer)
        if dropout > 0:
            model.add(Dropout(dropout))
        if bidirectional:
            model.add(Bidirectional(self.rnn_type(rnn_units)))
        else:
            model.add(LSTM(units=self.EMBEDDING_DIM))
            # model.add(rnn_type(rnn_units))

        model.add(Flatten())
        model.add(Dense(units=self.vocabulary_size, kernel_regularizer=regularizers.l2(kernel_regularizer), activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=metrics)
        if show_summary:
            print(model.summary())

        run_time = datetime.datetime.now().strftime('%m%d-%H%M%S')
        run_name = f'{prefix}_{run_time}_b{batch_size}_lr{lr}'

        self.callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=patience, verbose=verbose),
            ModelCheckpoint(f'{run_name}.h5', verbose=0, save_best_only=True, save_weights_only=True)]

        if tf.__version__[0] == '2':
            log_dir = f'logs/fit/{run_name}'
            self.callbacks.append(TensorBoard(log_dir=log_dir))
        self.model = model

    def eval(self, test_data, test_data_len):
        test_text = test_data
        self.model.evaluate(test_text, verbose=self.verbose,
                            steps=math.ceil((test_data_len - 1) / self.batch_size))  # callbacks=self.callbacks,

    def fit(self, X,y, validation_split=0.1):
        history = self.model.fit(X, y,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 verbose=self.verbose,
                                 shuffle=self.shuffle,
                                 validation_split=validation_split,
                                 callbacks=self.callbacks
                                 )
        return history

    def get_model(self):
        return self.model

    def predict_next_word(self, word_idxs, le_train=None):
        prediction = self.model.predict(x=word_idxs)
        return prediction


    # TODO: sample choice
    def sample(self, preds, temperature=1.0):
        if temperature <= 0:
            return np.argmax(preds)
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate(self, word_idxs, num_words=10):
        # print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
        for i in range(num_words):
            prediction = self.predict_next_word(np.array(word_idxs))
            idx = self.sample(prediction[-1], temperature=0.7)
            word_idxs.append(idx)
        return ' '.join(idx2word(word_idxs,self.tokenizer))
