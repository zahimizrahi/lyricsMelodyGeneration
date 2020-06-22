import os
ROOT_PATH = "."
DATA_PATH = "Data"
MIDI_PATH = "midi_files"
LYRICS_TRAIN = "lyrics_train_set.csv"
LYRICS_TEST = "lyrics_test_set.csv"
DIR_MELODY = 'Data/midi_files'
LYRICS_DIR = 'Data/'
TEXT_DATA = os.path.join(LYRICS_DIR, 'unified_lyrics_dump.txt')
SEQUENCE_LEN  = 2  # x is 1 and y is 1 -> During each step of the training phase, your architecture will receive as input one word of the lyrics.
STEP = 1
DOC2VEC_MODELS_PATHS = 'Data/Models'
#https://github.com/brangerbriz/midi-glove
NOTE_EMBEDDING_PATH = './vectors_d300.txt'