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

        clean_lyrics = clean_lyrics.str.replace('...', '.')
        clean_lyrics = clean_lyrics.str.replace('&', '.')
        # remove words
        for w in ['chorus', '\-\-\-', '\-\-']:
            clean_lyrics = clean_lyrics.str.replace(w, '')

        clean_lyrics = clean_lyrics.apply(lambda x: ' '.join(x.split()))
        return clean_lyrics

    def getIgnoreWordList(self, df, MIN_WORD_FREQUENCY=2):
        text = ' '.join([w for w in df.Text])
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
        df = pd.read_csv(lyrics_files_dir, usecols=[0, 1, 2], names=['Artist', 'SongName', 'Text'])
        df.Text = self.clean_data(df.Text)

        if min_ignore_word_frequency > 1:
            if ignored_words is None:
                if type != 'train':
                    raise ValueError('if type is not train - ignored_words cant to be Empty')
                ignored_words = self.getIgnoreWordList(df, MIN_WORD_FREQUENCY=min_ignore_word_frequency)
            df.Text = df.Text.apply(lambda line: self.getIgnoreText(line, ignored_words))

        df['MelodyPath'] = df.apply(lambda x: self.get_midi_path(x['Artist'], x['SongName']), axis=1)
        # in case there is no melody path for some of the rows in lyrics train
        df['MelodyPath'] = df['MelodyPath'].apply(lambda x: x if x in midi_files_list else None)
        df = df[df['MelodyPath'].notna()]
        df = df.reset_index(drop=True)

        if max_sentence != -1:
            df['Text'] = df['Text'].apply(lambda x: ' '.join(x.split()[:max_sentence]))
        X = []
        y = []
        for i, lyrics in enumerate(df.Text):
            splitted_lyrics = [token for token in word_tokenize(lyrics)]
            sub_x = []
            sub_y = []
            for j in range(len(splitted_lyrics) - 1):
                sub_x.append(splitted_lyrics[j])
                sub_y.append(splitted_lyrics[j + 1])
            X.append(np.array(sub_x))
            y.append(np.array(sub_y))
        df['X'] = X
        df['y'] = y

        return df, ignored_words

    def load_data(self,
                  is_melody_model=True, pre_embedding_melody=None, type='train',melody_type='doc2vec',
                  min_ignore_word_frequency=2, max_sentence=300, ignored_words = None):

        df, ignored_words = self.prepare_data(min_ignore_word_frequency=min_ignore_word_frequency,max_sentence=max_sentence, type=type, ignored_words = ignored_words)
        text_x = np.concatenate(df.X)
        text_y = np.concatenate(df.y)
        X = np.hstack(text_x)
        y = np.hstack(text_y)
        catch = {}
        catch['ignored_words'] = ignored_words
        if is_melody_model:
            # melody embedding handaling
            if pre_embedding_melody == None:
                embedding_melody = get_embedding_melody(melody_type=melody_type)
            else:
                embedding_melody = pre_embedding_melody

            df = df[df['MelodyPath'].isin(embedding_melody.keys())]
            df['EmbeddingMelody'] = df.MelodyPath.apply(lambda melody: embedding_melody[melody])
            df = df.reset_index(drop=True)
            df["EmbeddingMelody_multi"] = df.apply(lambda row: np.array([row["EmbeddingMelody"]] * len(row.X)),axis=1)
            songs = np.vstack(df["EmbeddingMelody_multi"])
            # End melody embedding handaling

            text_x = np.concatenate(df.X)
            text_y = np.concatenate(df.y)
            X = np.hstack(text_x)
            y = np.hstack(text_y)
            catch['df'] = df
            return X, y, songs, catch
        catch['df'] = df
        return X, y, catch

    def init_tokenizer(self, text):
        tokenizer = Tokenizer(filters='', oov_token='oov_token')
        tokenizer.fit_on_texts([text])
        return tokenizer

    def load_vocab(self, X = None):
        if X is None:
          X, _,_= self.load_data(is_melody_model=False, min_ignore_word_frequency = -1, max_sentence = -1)
        return list(set(X.flatten()))

    def load_tokenized_data(self, is_melody_model=True, melody_type='doc2vec',max_samples=-1,type='train',
                            pre_embedding_melody=None, min_ignore_word_frequency=2, max_sentence=300, ignored_words = None, tokenizer = None):
        if type == 'test':
            if tokenizer is None:
                raise ValueError('Please provide a pre-training tokenizer')
            if ignored_words is None:
                raise ValueError('Please provide a ignored_words')

        if is_melody_model:
            X, y, songs, catch = self.load_data(is_melody_model=is_melody_model, melody_type=melody_type,  pre_embedding_melody=pre_embedding_melody,type=type,
                                     min_ignore_word_frequency=min_ignore_word_frequency, max_sentence=max_sentence, ignored_words = ignored_words)
        else:
            X, y, catch = self.load_data(type=type, is_melody_model=is_melody_model, melody_type=melody_type , min_ignore_word_frequency=min_ignore_word_frequency, max_sentence=max_sentence, ignored_words = ignored_words)

        if tokenizer is None:
            all_songs_words = ' '.join(self.load_vocab(X=X))
            tokenizer = self.init_tokenizer(all_songs_words)

        catch['tokenizer'] = tokenizer
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
            return X, y, songs, catch
        else:
            return np.array(X), np.array(y), catch