import nltk
from consts import *
from midi_utils import *

def countSillabelsPerWord(word):
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    try:
        if word[0] in vowels:
            count += 1
    except:
        print(word)
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count += 1
    return count

def countSillabelsPerSong(lyrics):
    songCount = 0

    for word in lyrics:
        if word == 'TRACK_START':
            continue
        count = countSillabelsPerWord(word)
        songCount += count

    return songCount

def getSillabelSequences(trainDF, ignored_words,  midiUrlDict, SEQUENCE_LEN = SEQUENCE_LEN , STEP = STEP ):
    sequences = []
    noteSequencesDict = defaultdict(list)
    wordSequencesDict = defaultdict(list)
    removeList = [',', '', '.', '(', '!', '?', ')', ']', '[', ':']
    for index, row in trainDF.iterrows():
        songName = row['song_name']
        author = row['author']
        tokens = nltk.word_tokenize(row['lyrics'])
        tokens = [w for w in tokens if w not in removeList]
        tokens = [w.lower() for w in tokens]
        midiUrl = midiUrlDict[author][songName]
        if not midiUrl:
            continue
        notes_string = get_note_string(path=midiUrl)
        if not notes_string:
            continue
        notesList = notes_string.split(" ")
        endOfList = False
        for i in range(0, len(tokens) - SEQUENCE_LEN, STEP):
            # Only add the sequences where no word is in ignored_words
            if len(set(tokens[i: i + SEQUENCE_LEN + 1]).intersection(ignored_words)) == 0:
                words = tokens[i: i + SEQUENCE_LEN]
                songCount = countSillabelsPerSong(lyrics=words)
                subNotes = []
                for i in range(songCount):
                    if endOfList:
                        break
                    try:
                        note = notesList.pop(-1)
                        subNotes.append(note)
                    except:
                        endOfList = True

                sequences.append(words)
                wordSequencesDict[songName].append(words)
                noteSequencesDict[songName].append(subNotes)
    assert len(wordSequencesDict) == len(noteSequencesDict)
    return sequences, wordSequencesDict, noteSequencesDict

def concatinatingNotesAndWord(wordSequencesDict, noteSequencesDict, word_model, allNoteEmbeddingsDict, sequences, tokenizer):
    train_x = np.zeros([len(sequences), 10, 600])
    train_y = np.zeros([len(sequences)])
    locDict = defaultdict(list)
    i = 0
    for song in wordSequencesDict.keys():
        locDict[song].append(i)
        if len(noteSequencesDict[song]) != len(wordSequencesDict[song]):
            continue
        for index, sequence in enumerate(wordSequencesDict[song]):
            subNotes = noteSequencesDict[song][index]
            embeddedSequenceMidi = getNotesEmbedded(embeddings_dict=allNoteEmbeddingsDict, notesList=subNotes,dim_size=300)
            if len(embeddedSequenceMidi) != 300:
                continue
            for t, word in enumerate(sequence):
                try:
                    wordEmmbedding = word_model[word]
                except:
                    wordEmmbedding = word_model["unk_embedding"]
                train_x[i, t] = np.concatenate([wordEmmbedding, embeddedSequenceMidi])
            try:
                train_y[i] = word2idx(sequence[-1], tokenizer)
            except:
                train_y[i] = 0
            i += 1
        locDict[song].append(i)

    return train_x, train_y, locDict