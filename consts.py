import os

ROOT_PATH = "."
DATA_PATH = "Data"
MIDI_PATH = "midi_files"
LYRICS_TRAIN = "lyrics_train_set.csv"
LYRICS_TEST = "lyrics_test_set.csv"
DIR_MELODY = 'Data/midi_files'
LYRICS_DIR = 'Data/'
TEXT_DATA = os.path.join(LYRICS_DIR, 'unified_lyrics_dump.txt')

MAX_SEQUENCE_LENGTH = 1  # During each step of the training phase, your architecture will receive as input one word of the lyrics.
VALIDATION_SPLIT = 0.2
DOC2VEC_MODELS_PATHS = 'Data/Models'

#https://github.com/brangerbriz/midi-glove
NOTE_EMBEDDING_PATH = './vectors_d300.txt'