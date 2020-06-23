import nltk
from consts import *
from midi_utils import *
from collections import defaultdict

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

def word2idx(text, tokenizer):
    # word2idx("the food".split(), tokenizer)
    encoded = tokenizer.texts_to_sequences(text)[0]
    encoded = np.array(encoded)
    return encoded

def get_org_midi_path(midi_path):
  for path in os.listdir('Data/midi_files/'):
    if midi_path in path.lower():
      return 'Data/midi_files/'+ path
  return None

def get_sillabel_sequences(df):
  sequences = []
  noteSequencesDict = defaultdict(list)
  wordSequencesDict = defaultdict(list)

  for index, row in df.iterrows():
      tokens = row['X']
      song_name = row['MelodyPath']
      midi_path = get_org_midi_path(song_name)
      notes_string = get_note_string(path=midi_path)
      if not notes_string:
        print(f'extract note from {midi_path} failed')
        continue
      notesList = notes_string.split(" ")
      endOfList = False

      for i in range(0, len(tokens) - SEQUENCE_LEN, STEP):
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
          wordSequencesDict[song_name].append(words)
          noteSequencesDict[song_name].append(subNotes)
  return sequences, wordSequencesDict, noteSequencesDict

def concatinatingNotesAndWord(wordSequencesDict, noteSequencesDict, word_model, allNoteEmbeddingsDict, sequences, tokenizer):
    train_x = np.zeros([len(sequences), SEQUENCE_LEN-1, 600])
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
            for t, word in enumerate(sequence[:-1]):
                try:
                    wordEmmbedding = word_model[word]
                except:
                    wordEmmbedding = word_model["unk_embedding"]
                train_x[i, t] = np.concatenate([wordEmmbedding, embeddedSequenceMidi])
            try:
                train_y[i] = word2idx([sequence[-1]], tokenizer)
            except:
                train_y[i] = 0
            i += 1
        locDict[song].append(i)

    return train_x, train_y, locDict