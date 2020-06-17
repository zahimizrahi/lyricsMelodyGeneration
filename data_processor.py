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
from midi_utils import *
import joblib
from consts import *
import pandas as pd
from nltk import word_tokenize
from collections import Counter

class DataProcessor:

    def get_midi_path(self, artist, song_name):
        return '{}_-_{}.mid'.format(artist.strip().replace(' ', '_'), song_name.strip().replace(' ', '_'))

    def clean_data(self, lyrics_set):
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

    def getIgnoreWordList(self, df_train, MIN_WORD_FREQUENCY=2):
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

    def getIgnoreText(self, line, ignored_words):
            if ignored_words is None:
                raise ValueError("turn on Ignore Word")
            line_array = line.split()
            line_array = [word for word in line_array if word not in ignored_words]
            return ' '.join(line_array)

    def prepare_data(self, min_ignore_word_frequency=2, max_sentence=300, type='train', ignored_words=None):
        midi_files_list = [filename.lower() for filename in os.listdir(os.path.join(ROOT_PATH, DATA_PATH, MIDI_PATH))]
        train_or_test = LYRICS_TRAIN if type == 'train' else LYRICS_TEST
        lyrics_files_dir = os.path.join(ROOT_PATH, DATA_PATH, train_or_test)
        df_train = pd.read_csv(lyrics_files_dir, usecols=[0, 1, 2], names=['Artist', 'SongName', 'Text'])
        df_train.Text = self.clean_data(df_train.Text)

        if min_ignore_word_frequency > 1:
            if ignored_words is None:
                if type != 'train':
                    raise ValueError('if type is not train - ignored_words cant to be Empty')
                ignored_words = self.getIgnoreWordList(df_train, MIN_WORD_FREQUENCY=min_ignore_word_frequency)
            df_train.Text = df_train.Text.apply(lambda line: self.getIgnoreText(line, ignored_words))

        df_train['MelodyPath'] = df_train.apply(lambda x: self.get_midi_path(x['Artist'], x['SongName']), axis=1)
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

    def load_data(self,
                  is_melody_model=True, pre_embedding_melody=None, type='train', min_ignore_word_frequency=2, max_sentence=300, ignored_words = None):
        df_train, ignored_words = self.prepare_data(min_ignore_word_frequency=min_ignore_word_frequency,max_sentence=max_sentence, type=type, ignored_words = ignored_words)
        text_x = np.concatenate(df_train.X)
        text_y = np.concatenate(df_train.y)
        X = np.hstack(text_x)
        y = np.hstack(text_y)

        if is_melody_model:
            # melody embedding handaling
            if pre_embedding_melody == None:
                embedding_melody = get_embedding_melody()
            else:
                embedding_melody = pre_embedding_melody

            df_train = df_train[df_train['MelodyPath'].isin(embedding_melody.keys())]
            # test_df = test_df[test_df['MelodyPath'].isin(pre_embedding_melody.keys())]
            df_train['EmbeddingMelody'] = df_train.MelodyPath.apply(lambda melody: embedding_melody[melody])
            df_train = df_train.reset_index(drop=True)
            # test_df['EmbeddingMelody'] = test_df.MelodyPath.apply(lambda melody: embedding_melody[melody])
            df_train["EmbeddingMelody_multi"] = df_train.apply(lambda row: np.array([row["EmbeddingMelody"]] * len(row.X)),
                                                               axis=1)
            songs = np.vstack(df_train["EmbeddingMelody_multi"])
            # End melody embedding handaling

            text_x = np.concatenate(df_train.X)
            text_y = np.concatenate(df_train.y)
            X = np.hstack(text_x)
            y = np.hstack(text_y)
            return X, y, songs, ignored_words
        return X, y, ignored_words

    def init_tokenizer(self, text):
        tokenizer = Tokenizer(filters='', oov_token='oov_token')
        tokenizer.fit_on_texts([text])
        return tokenizer

    def load_vocab(self, X = None):
        if X is None:
          X, _= self.load_data(is_melody_model=False, min_ignore_word_frequency = -1, max_sentence = -1)
        return list(set(X.flatten()))

    def load_tokenized_data(self, is_melody_model=True, max_samples=-1,type='train', pre_embedding_melody=None, min_ignore_word_frequency=2, max_sentence=300, ignored_words = None, tokenizer = None):
        if is_melody_model:
            X, y, songs, ignored_words = self.load_data(is_melody_model=is_melody_model, pre_embedding_melody=pre_embedding_melody,type=type,
                                     min_ignore_word_frequency=min_ignore_word_frequency, max_sentence=max_sentence, ignored_words = ignored_words)
        else:
            X, y, ignored_words = self.load_data(type=type, is_melody_model=is_melody_model,min_ignore_word_frequency=min_ignore_word_frequency, max_sentence=max_sentence, ignored_words = ignored_words)

        if tokenizer is None:
            all_songs_words = ' '.join(self.load_vocab(X=X))
            tokenizer = self.init_tokenizer(all_songs_words)

        X = [lst[0] for lst in tokenizer.texts_to_sequences(X)]
        y = [lst[0] for lst in tokenizer.texts_to_sequences(y)]

        if is_melody_model:
            y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

        if max_samples != -1:
            X = X[:max_samples]
            y = y[:max_samples]

        if is_melody_model:
            if max_samples != -1:
                songs = songs[:max_samples, :]
            return X, y, tokenizer, songs, ignored_words
        else:
            return np.array(X), np.array(y), tokenizer, ignored_words

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4

    def load_test_data(self, ignored_words, with_melody=False, melody_type='doc2vec'):
        df_test, _ = self.prepare_data(type='test', ignored_words = ignored_words)
        if with_melody:
            models = get_melody_models()
            for row in df_test.iterrows():
                try:
                    # if melody_type == 'doc2vec':
                    # use pre_embedding_melody --> df_test['EmbeddingMelody'] = df_test.MelodyPath.apply(lambda melody: pre_embedding_melody[melody])
                    row['EmbeddingMelody'] = np.array(get_song_vector(row["MelodyPath"], models))
                    # else:
                    #     parsed_songs[i]['melody_embedding'] = extract_midi_piano_roll(song_data['midi_path'])
                except Exception as e:
                    row['EmbeddingMelody'] = None
                    print(r"Couldn't load {}, issue with the midi file.".format(song_data['midi_path']))
        return df_test


    def load_tokenized_test(self, tokenizer, is_melody_model=False, melody_type='doc2vec', ignored_words = None):
        if tokenizer is None:
            raise ValueError('Please provide a pre-training tokenizer')
        # parsed_songs = load_test_data(with_melody=with_melody, melody_type=melody_type, ignored_words = ignored_words)
        #
        # for i, song_data in enumerate(parsed_songs):
        #     parsed_songs[i]['tokenized_X'] = [tokenizer.texts_to_sequences(song_data['X'])]
        #     parsed_songs[i]['tokenized_y'] = [tokenizer.texts_to_sequences(song_data['y'])]
        #     parsed_songs[i]['categorical_y'] = to_categorical(parsed_songs[i]['tokenized_y'], num_classes=len(tokenizer.word_index) + 1)
        #
        return self.load_tokenized_data(tokenizer = tokenizer, is_melody_model=is_melody_model, ignored_words = ignored_words, type='test')
        # return parsed_songs



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