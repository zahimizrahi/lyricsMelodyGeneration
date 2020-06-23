import numpy as np
import scipy
from const import *
from embeddings import *

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

def getIgnoreWordList(df, MIN_WORD_FREQUENCY=2):
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

def getIgnoreText(line, ignored_words):
    if ignored_words is None:
        raise ValueError("turn on Ignore Word")
    line_array = line.split()
    line_array = [word for word in line_array if word not in ignored_words]
    return ' '.join(line_array)

def get_midi_path(artist, song_name):
    return '{}_-_{}.mid'.format(artist.strip().replace(' ', '_'), song_name.strip().replace(' ', '_'))

def word2idx(text, tokenizer):
    # word2idx("the food".split(), tokenizer)
    encoded = tokenizer.texts_to_sequences(text)[0]
    encoded = np.array(encoded)
    return encoded

def idx2word(index, tokenizer):
    for word, idx in tokenizer.word_index.items():
        if idx == index:
            return word

def cosine_distance_wordembedding_method(s1, s2):
    word_model =load_pretrained_embedding()
    vector_1 = np.mean([word_model[word] for word in s1.split()],axis=0)
    vector_2 = np.mean([word_model[word] for word in s2.split()],axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    cosinn_score = round((1-cosine)*100,2)
    # print('Word Embedding method with a cosine distance asses that our two sentences are similar to',cosinn_score,'%')
    return cosinn_score