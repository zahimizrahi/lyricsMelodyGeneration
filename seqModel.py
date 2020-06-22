from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras import callbacks as cb
from keras import utils as np_utils
from keras.layers import CuDNNLSTM
from keras.optimizers import SGD
import datetime
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Flatten
from keras import regularizers

from keras.layers.recurrent import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.utils.data_utils import get_file
import numpy as np

def idx2word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word

class seqModel:
    def __init__(self,
                 tokenizer,
                 rnn_units=50,
                 input_length = 1,
                 vocab_size = None,
                 rnn_type='lstm',
                 dropout=0.3,
                 kernal=0.1,
                 validation_split = 0.1,
                 batch_size=32,
                 epochs=10,
                 optimizer='adam',
                 show_summary=True,
                 shuffle=True,
                 verbose=True,
                 patience=3):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type.lower()
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = 'adam'
        self.shuffle = shuffle
        self.verbose = verbose
        self.patience = patience
        self.input_length = input_length

        self.vocab_size = len(tokenizer.word_index) + 1
        model = Sequential()
        if self.rnn_type == 'lstm':
          model.add(LSTM(rnn_units, input_shape=(input_length, 600)))
        elif self.rnn_type == 'gru':
          model.add(GRU(rnn_units, input_shape=(input_length, 600)))
        else:
          raise ValueError('Please provide a valid rnn type (GRU / LSTM)')

        if dropout > 0:
            model.add(Dropout(dropout))
        if is_layer_norm:
            model.add(LayerNormalization())
        model.add(Dense(units= self.vocab_size, kernel_regularizer=regularizers.l2(kernal), activation='softmax'))
        model.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        if show_summary:
            model.summary()

        run_time = datetime.datetime.now().strftime('%m%d-%H%M%S')
        run_name = f'{run_time}_b{self.batch_size}_epochs{self.epochs}'

        self.callbacks = [
            EarlyStopping(patience=self.patience, verbose=self.verbose),
            ModelCheckpoint(f'{run_name}.h5', verbose=0, save_best_only=True, save_weights_only=True)]
        log_dir = f'logs/fit/{run_name}'
        self.callbacks.append(TensorBoard(log_dir=log_dir))
        self.model = model

    def get_model(self):
        return self.model

    def train(self,train_x, train_y):
        return self.model.fit(train_x, train_y,batch_size=self.batch_size,epochs=self.epochs,
                              callbacks=self.callbacks,  validation_split=self.validation_split )

    def predict(self, word_model, X_test, first_word, n_word, seqMelodyIndexStart):
        model_input = np.zeros([0, self.input_length, 600])
        new_word = first_word

        for num_word in range(n_word):

            try:
                wordEmbedding = word_model[new_word]
            except:
                wordEmbedding = word_model["unk_embedding"]

            seqMelodyIndex = seqMelodyIndexStart + num_word
            seqMelodyEmbbeding = X_test[seqMelodyIndex][0][300:]

            model_input_new = np.concatenate([wordEmbedding, seqMelodyEmbbeding]).reshape(1, self.input_length, 600)
            model_input = np.append(model_input_new, model_input, axis=0)

            words_probs = my_model.predict(model_input)[0]
            words_probs_enu = list(enumerate(words_probs))
            words_probs_sorted = sorted(words_probs_enu, key=lambda x: x[1], reverse=True)  # sorting in descending order

            words_tokens, words_probs = list(zip(*words_probs_sorted))
            # normalizre to sum 1
            words_probs = np.array(words_probs, dtype=np.float64)
            words_probs /= words_probs.sum().astype(np.float64)
            word_token = np.random.choice(words_tokens, p=words_probs)
            # map predicted word index to word
            new_word = idx2word(word_token, self.tokenizer)
            # append to input
            result = result + ' ' + new_word
        return result






