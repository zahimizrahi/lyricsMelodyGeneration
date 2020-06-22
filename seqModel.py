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

class seqModel:
    def __init__(self,
                 rnn_units=50,
                 input_length = 1,
                 vocab_size = None,
                 rnn_type='lstm',
                 bidirectional=True,
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
        model.add(Dense(units=vocab_size, kernel_regularizer=regularizers.l2(kernal), activation='softmax'))
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

# TODO: PRDEICT!
# # predict:
# for song in testDF['song_name']:
#     print('song: ', song)
#     songDF = testDF[testDF['song_name'] == song]
#     lyrics = songDF['lyrics'][0]
#     tokens = nltk.word_tokenize(lyrics)
#     startWord = tokens[0]
#     print(startWord)
#     print(predict(model=model, startWord=startWord, song=song, num_to_generate=15, test_x=test_x, locDict=locDict,
#                   method='All'))







