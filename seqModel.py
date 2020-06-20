from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras import callbacks as cb

from keras import utils as np_utils
from keras.layers import CuDNNLSTM
from keras.optimizers import SGD

from __future__ import print_function
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Flatten


from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.utils.data_utils import get_file

class seqModel:
    def __init__(self, tokenizer= None, embedding_matrix = None,
                 rnn_units=50,
                 vocab_size = None,
                 melody_vec_dim = 150,
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
        self.vocab_size = vocab_size
        self.rnn_type = rnn_type.lower()
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = 'adam'
        self.shuffle = shuffle
        self.verbose = verbose
        self.patience = patience
        model = Sequential()
        model.add(LSTM(50, input_shape=(10, 600)))
        model.add(Dense(units=vocab_size))
        model.add(Activation('softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        if show_summary:
            model.summary()

        run_time = datetime.datetime.now().strftime('%m%d-%H%M%S')
        run_name = f'{run_time}_b{self.batch_size}_epochs{self.epochs}'

        self.callbacks = [
            EarlyStopping(patience=self.patience, verbose=self.verbose),
            ModelCheckpoint(f'{run_name}.h5', verbose=0, save_best_only=True, save_weights_only=True)]

        self.model = model

    def train(self,train_x, train_y, vocab_size):
        return self.model.fit(train_x, train_y,batch_size=self.batch_size,epochs=self.epochs, callbacks=self.callbacks ) # [TensorBoardColabCallback(tbc)])

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







