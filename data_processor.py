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
from seq_utils import *
import joblib
import pandas as pd
from collections import Counter
from embeddings import load_pretrained_embedding
from utils import *
from consts import *


class DataProcessor:

    def prepare_data(self, min_ignore_word_frequency=2, max_sentence=300, type='train', ignored_words=None):
        midi_files_list = [filename.lower() for filename in os.listdir(os.path.join(ROOT_PATH, DATA_PATH, MIDI_PATH))]
        train_or_test = LYRICS_TRAIN if type == 'train' else LYRICS_TEST
        lyrics_files_dir = os.path.join(ROOT_PATH, DATA_PATH, train_or_test)
        df = pd.read_csv(lyrics_files_dir, usecols=[0, 1, 2], names=['Artist', 'SongName', 'Text'])
        df.Text = clean_data(df.Text)

        if min_ignore_word_frequency > 1:
            if ignored_words is None:
                if type != 'train':
                    raise ValueError('if type is not train - ignored_words cant to be Empty')
                ignored_words = getIgnoreWordList(df, MIN_WORD_FREQUENCY=min_ignore_word_frequency)
            df.Text = df.Text.apply(lambda line: getIgnoreText(line, ignored_words))

        df['MelodyPath'] = df.apply(lambda x: get_midi_path(x['Artist'], x['SongName']), axis=1)
        # in case there is no melody path for some of the rows in lyrics train
        df['MelodyPath'] = df['MelodyPath'].apply(lambda x: x if x in midi_files_list else None)
        df = df[df['MelodyPath'].notna()]
        df = df.reset_index(drop=True)

        if max_sentence > 0:
            df['Text'] = df['Text'].apply(lambda x: ' '.join(x.split()[:max_sentence]))
        X = []
        y = []
        for i, lyrics in enumerate(df.Text):

            splitted_lyrics = [token for token in word_tokenize(lyrics)]
            if max_sentence > 0:
                splitted_lyrics = splitted_lyrics[:max_sentence]
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
            embedding_melody = {k.lower(): v for k, v in embedding_melody.items() if len(v) > 0}
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

    def init_tokenizer(self, text = None, text_not_flatten = True):
        if text_not_flatten:
            text = ' '.join(self.load_vocab(X=text))
        tokenizer = Tokenizer(filters='', oov_token='oov_token')
        tokenizer.fit_on_texts([text])
        return tokenizer

    def load_vocab(self, X = None):
        if X is None:
          X, _,_= self.load_data(is_melody_model=False, min_ignore_word_frequency = -1, max_sentence = -1)
        return list(set(X.flatten()))

    def fit_transfer_tokenizer(self, X, y, tokenizer=None ):
        if tokenizer is None:
            tokenizer = self.init_tokenizer(text=X)
        X = [lst[0] for lst in tokenizer.texts_to_sequences(X)]
        y = [lst[0] for lst in tokenizer.texts_to_sequences(y)]
        return tokenizer, X, y

    def load_tokenized_data(self, is_melody_model=True, melody_type='doc2vec', method = 'All', max_samples=-1,type='train',
                            pre_embedding_melody=None, min_ignore_word_frequency=2, max_sentence=300, ignored_words = None, tokenizer = None):
        if type == 'test':
            if tokenizer is None:
                raise ValueError('Please provide a pre-training tokenizer')
            if ignored_words is None:
                raise ValueError('Please provide a ignored_words')

        if not is_melody_model:
            X, y, catch = self.load_data(type=type, is_melody_model=is_melody_model, melody_type=melody_type,
                                         min_ignore_word_frequency=min_ignore_word_frequency, max_sentence=max_sentence,
                                         ignored_words=ignored_words)
            if type == 'test':
                return np.array(X), np.array(y), catch

            catch['tokenizer'], X, y = self.fit_transfer_tokenizer(X, y, tokenizer)
            if max_samples != -1:
                X = X[:max_samples]
                y = y[:max_samples]
            return np.array(X), np.array(y), catch

        #else - is_melody_model
        if method == 'All':
            X, y, songs, catch = self.load_data(is_melody_model=is_melody_model, melody_type=melody_type,  pre_embedding_melody=pre_embedding_melody,type=type,
                                         min_ignore_word_frequency=min_ignore_word_frequency, max_sentence=max_sentence, ignored_words = ignored_words)

            tokenizer, X, y = self.fit_transfer_tokenizer(X, y, tokenizer)
            catch['tokenizer'] = tokenizer
            y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

            if max_samples != -1:
                X = X[:max_samples]
                y = y[:max_samples]
                songs = songs[:max_samples, :]
            return X, y, songs, catch

        else: # if method=='Seq2Seq
            print('\nprepare data\n')
            df, ignored_words = self.prepare_data(min_ignore_word_frequency=min_ignore_word_frequency,
                                                  max_sentence=max_sentence, type=type, ignored_words=ignored_words)
            print('\nget_sillabel_sequences\n')
            sequences, wordSequencesDict, noteSequencesDict = get_sillabel_sequences(df)
            df = df[df['MelodyPath'].isin(wordSequencesDict.keys())]
            df = df.reset_index(drop=True)

            word_model = load_pretrained_embedding()
            if tokenizer is None:
                all_songs_words = ' '.join(list(set(np.array(sequences).flatten())))
                tokenizer = self.init_tokenizer(text = all_songs_words, text_not_flatten = False)

            allNoteEmbeddingsDict = ExtractGloveEmbeddingDict()

            print('\nprepare data sets\n')
            X, y, locDict = concatinatingNotesAndWord(wordSequencesDict=wordSequencesDict,
                                                      noteSequencesDict=noteSequencesDict,
                                                      word_model=word_model,
                                                      allNoteEmbeddingsDict=allNoteEmbeddingsDict,
                                                      sequences=sequences, tokenizer=tokenizer)

            catch = {}
            catch['tokenizer'] = tokenizer
            catch['df'] = df
            catch['word_model'] = word_model
            catch['locDict'] = locDict
            catch['ignored_words'] = ignored_words
            return X, y, catch