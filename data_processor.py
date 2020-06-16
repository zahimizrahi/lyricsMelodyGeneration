import warnings
warnings.filterwarnings("ignore")
import string
import nltk
import os
import numpy as np
import pickle
import itertools
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from tqdm import tqdm
from midi_utils import get_song_vector, extract_midi_piano_roll
import joblib
from consts import *
import pandas as pd

def get_midi_path(artist, song_name):
    return '{}_-_{}.mid'.format(artist.strip().replace(' ', '_'), song_name.strip().replace(' ', '_'))

def clean_data(lyrics_set):
    clean_lyrics = lyrics_set.str.replace(r"''", '"').replace(r'`', "'").replace("/", "-").replace('\?\?\?', '?')
    # Replace all brackets, and weird chars and :;#"
    reg = r'[:;#\*"ã¤¼©¦­\]\[\}\{\)(?!]'
    clean_lyrics = clean_lyrics.str.replace(r'[\[\]()\{\}:;#\*"ã¤¼©¦­!?]', '')

    # replace words with space
    for w in [" '", '\.\.', '\.\.\.', '\.\.\.\.', '\.\.\.\.\.']:
        clean_lyrics = clean_lyrics.str.replace(w, ' ')

    clean_lyrics = clean_lyrics.str.replace('&', '.')
    # remove words
    for w in ['chorus', '\-\-\-', '\-\-']:
        clean_lyrics = clean_lyrics.str.replace(w, '')

    clean_lyrics = clean_lyrics.apply(lambda x: ' '.join(x.split()))
    return clean_lyrics

def getIgnoreWordList(df_train, MIN_WORD_FREQUENCY=2):
    text = ' '.join([w for w in df_train.Text])
    tokens = word_tokenize(text)
    text_in_words = tokens
    word_freq = Counter(text_in_words)
    ignored_words = set()
    for k, v in word_freq.items():
        if word_freq[k] < MIN_WORD_FREQUENCY:
            ignored_words.add(k)
    words = set(text_in_words)
    print('Unique words before ignoring:', len(words))
    print('Ignoring words with frequency <', MIN_WORD_FREQUENCY)
    words = sorted(set(words) - ignored_words)
    print('Unique words after ignoring:', len(words))
    return ignored_words

def getIgnoreText(line, ignored_words):
        if ignored_words is None:
            raise ValueError("turn on Ignore Word")
        line_array = line.split()
        line_array = [word for word in line_array if word not in ignored_words]
        return ' '.join(line_array)

def prepare_data(min_ignore_word_frequency=2, max_sentence=300, type='train', ignored_words=None):
    midi_files_list = [filename.lower() for filename in os.listdir(os.path.join(ROOT_PATH, DATA_PATH, MIDI_PATH))]
    train_or_test = LYRICS_TRAIN if type == 'train' else LYRICS_TEST
    lyrics_files_dir = os.path.join(ROOT_PATH, DATA_PATH, train_or_test)
    df_train = pd.read_csv(lyrics_files_dir, usecols=[0, 1, 2], names=['Artist', 'SongName', 'Text'])
    df_train.Text = clean_data(df_train.Text)

    if min_ignore_word_frequency > 1:
        if ignored_words is None:
            if type != 'train':
                raise ValueError('if type is not train - ignored_words cant to be Empty')
            ignored_words = getIgnoreWordList(df_train, MIN_WORD_FREQUENCY=min_ignore_word_frequency)
        df_train.Text = df_train.Text.apply(lambda line: getIgnoreText(line, ignored_words))

    df_train['MelodyPath'] = df_train.apply(lambda x: get_midi_path(x['Artist'], x['SongName']), axis=1)
    # in case there is no melody path for some of the rows in lyrics train
    df_train['MelodyPath'] = df_train['MelodyPath'].apply(lambda x: x if x in midi_files_list else None)
    df_train = df_train[df_train['MelodyPath'].notna()]
    df_train = df_train.reset_index(drop=True)

    if max_sentence != -1:
        df_train['Text'] = df_train['Text'].apply(lambda x: ' '.join(x.split()[:max_sentence]))

    X = []
    y = []
    for i, lyrics in enumerate(df_train.Text):
        splitted_lyrics = [token for token in word_tokenize(lyrics)]
        sub_x = []
        sub_y = []
        for j in range(len(splitted_lyrics) - 1):
            sub_x.append(splitted_lyrics[j])
            sub_y.append(splitted_lyrics[j + 1])
        X.append(np.array(sub_x))
        y.append(np.array(sub_y))
    df_train['X'] = X
    df_train['y'] = y

    return df_train, ignored_words

def load_data(is_melody_model=True, pre_embedding_melody=None, min_ignore_word_frequency=2, max_sentence=300):
    df_train, ignored_words = prepare_data(min_ignore_word_frequency=min_ignore_word_frequency,max_sentence=max_sentence)
    text_x = np.concatenate(df_train.X)
    text_y = np.concatenate(df_train.y)
    X = np.hstack(text_x)
    y = np.hstack(text_y)

    if is_melody_model:
        # melody embedding handaling
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@222
        # if melody_type == 'doc2vec':
        # songs.append(np.array([get_song_vector(midi_path, models)]*count))
        # else:
        #           songs.append([extract_midi_piano_roll(midi_path)] * count)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@222

        if pre_embedding_melody == None:
            local_pickle_file = os.path.join(DOC2VEC_MODELS_PATHS, 'dict_embedding_melody.pickle')
            if os.path.exists(local_pickle_file):
                print(f'Load {local_pickle_file}')
                with open(local_pickle_file, 'rb') as handle:
                    pre_embedding_melody = pickle.load(handle)
            else:
                pre_embedding_melody = get_dict_embedding()

        df_train = df_train[df_train['MelodyPath'].isin(pre_embedding_melody.keys())]
        # test_df = test_df[test_df['MelodyPath'].isin(pre_embedding_melody.keys())]
        df_train['EmbeddingMelody'] = df_train.MelodyPath.apply(lambda melody: pre_embedding_melody[melody])
        df_train = df_train.reset_index(drop=True)
        # test_df['EmbeddingMelody'] = test_df.MelodyPath.apply(lambda melody: embedding_melody[melody])
        df_train["EmbeddingMelody_multi"] = df_train.apply(lambda row: np.array([row["EmbeddingMelody"]] * len(row.X)),
                                                           axis=1)
        songs = np.vstack(df_train["EmbeddingMelody_multi"])
        # end melody embedding handaling

        text_x = np.concatenate(df_train.X)
        text_y = np.concatenate(df_train.y)
        X = np.hstack(text_x)
        y = np.hstack(text_y)
        return X, y, songs
    return X, y

def init_tokenizer(text):
    tokenizer = Tokenizer(filters='', oov_token='oov_token')
    tokenizer.fit_on_texts([text])
    return tokenizer

def load_vocab(X = None):
    if X is None:
      X, _= load_data(is_melody_model=False, min_ignore_word_frequency = -1, max_sentence = -1)
    return list(set(X.flatten()))

def load_tokenized_data(is_melody_model=True, melody_type='doc2vec', max_samples=-1, pre_embedding_melody=None,
                         min_ignore_word_frequency=2, max_sentence=300):
    if is_melody_model:
        X, y, songs = load_data(is_melody_model=is_melody_model, pre_embedding_melody=pre_embedding_melody,
                                 min_ignore_word_frequency=min_ignore_word_frequency, max_sentence=max_sentence)
    else:
        X, y = load_data(with_melody=with_melody, melody_type=melody_type,
                          min_ignore_word_frequency=min_ignore_word_frequency, max_sentence=max_sentence)

    all_songs_words = ' '.join(load_vocab(X=X))
    tokenizer = init_tokenizer(all_songs_words)

    X = [lst[0] for lst in tokenizer.texts_to_sequences(X)]
    y = [lst[0] for lst in tokenizer.texts_to_sequences(y)]
    y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

    if max_samples != -1:
        X = X[:max_samples]
        y = y[:max_samples]

    if is_melody_model:
        if max_samples != -1:
            songs = songs[:max_samples, :]
        return X, y, tokenizer, songs
    else:
        return X, y, tokenizer


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4

def load_test_data(with_melody=False, melody_type='doc2vec'):
    parsed_songs = prepare_data(type='test')

    if with_melody:
        models = {name: joblib.load(os.path.join(DOC2VEC_MODELS_PATHS, f'{name}_model.joblib')) for name in
                  ['drums', 'melody', 'harmony']}

        for i, song_data in tqdm(enumerate(parsed_songs), total=len(parsed_songs), desc='Loading the songs embedding'):
            try:
                if melody_type == 'doc2vec':
                    parsed_songs[i]['melody_embedding'] = np.array(get_song_vector(song_data['midi_path'], models))
                else:
                    parsed_songs[i]['melody_embedding'] = extract_midi_piano_roll(song_data['midi_path'])

            except Exception as e:
                print(r"Couldn't load {}, issue with the midi file.".format(song_data['midi_path']))
                continue

    return parsed_songs


def load_tokenized_test(tokenizer, with_melody=False, melody_type='doc2vec'):
    parsed_songs = load_test_data(with_melody=with_melody, melody_type=melody_type)
    for i, song_data in enumerate(parsed_songs):
        parsed_songs[i]['tokenized_X'] = [tokenizer.texts_to_sequences(song_data['X'])]
        parsed_songs[i]['tokenized_y'] = [tokenizer.texts_to_sequences(song_data['y'])]
        parsed_songs[i]['categorical_y'] = to_categorical(parsed_songs[i]['tokenized_y'], num_classes=len(tokenizer.word_index) + 1)
    return parsed_songs



def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4