import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pretty_midi
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
from consts import *
import pickle
import cv2
import joblib
import pathlib

def get_embedding_melody(melody_type='doc2vec'):
    pickle_name = 'dict_embedding_melody.pickle' if melody_type=='doc2vec' else 'songs_embedding_glove_dict.pickle'
    local_pickle_file = os.path.join(DOC2VEC_MODELS_PATHS, pickle_name)
    if os.path.exists(local_pickle_file):
        print(f'Load {local_pickle_file}')
        with open(local_pickle_file, 'rb') as handle:
            embedding_melody = pickle.load(handle)
    else:
        embedding_melody = get_dict_embedding(melody_type = melody_type)
    return embedding_melody

def get_melody_models(models_path = DOC2VEC_MODELS_PATHS):
    print('Load melody embbeding melodies')
    models = {name: joblib.load(os.path.join(models_path, f'{name}_model.joblib')) for name in
              ['drums', 'melody', 'harmony']}
    return models

def get_dict_embedding(models_path = DOC2VEC_MODELS_PATHS , dir_melody = DIR_MELODY, melody_type='doc2vec' ):
    models = get_melody_models(models_path = models_path) if melody_type=='doc2vec' else  ExtractGloveEmbeddingDict()
    midi_paths = pathlib.Path(dir_melody)
    midi_files = list(midi_paths.glob('*'))
    songs_vectors = []
    song_names = []
    for midi_file in tqdm(midi_files, total=len(midi_files)):
        try:
            if melody_type == 'doc2vec':
                songs_vectors.append(get_song_vector(str(midi_file), models))
            else:
                songs_vectors.append(get_notes_embeddings(models, str(midi_file), dim_size=300))
            song_names.append(midi_file.name.strip().lower())
        except Exception as e:
          print(e)
          print(f"Invalid song: {midi_file.name}")
    return dict(zip(song_names, songs_vectors))

def check_if_melody(instrument, silence_threshold=0.7, mono_threshold=0.8, fs=10):
    """
    Check if the given instrument is Melody, Harmony or too silence

    :param instrument: the object that contain the note information
    :param silence_threshold: the threshold that above it the instrument considired to be too quiet.
    :param mono_threshold: the threshold that above it the instrument considered to be a Melody
    :param fs: the rate to sample from the midi
    :return: True - the instrument is considered as melody, False - the instrument considered as harmony, -1 - the
    instrument considered as too quiet.
    """
    # Extract all of the notes of the instrument
    piano_roll = instrument.get_piano_roll(fs=fs)

    # extract the timeframes the contain notes
    timeframes_with_notes_indexes = np.unique(np.where(piano_roll != 0)[1])
    piano_roll_notes = piano_roll[:, timeframes_with_notes_indexes]

    n_timeframes = piano_roll.shape[1]
    n_notes = piano_roll_notes.shape[1]
    n_silence = n_timeframes - n_notes
    n_mono = np.count_nonzero(np.count_nonzero(piano_roll_notes > 0, axis=0) == 1)

    # check if instrument is too quiet
    if silence_threshold <= float(n_silence)/n_timeframes:
        return -1

    if mono_threshold <= float(n_mono)/n_notes:
        return True
    return False

def number_to_note(number):
    """
    Extract note name from note number

    :param number: index of note
    :return: note name or "r" for rest
    """
    if number == 128:
        return 'r'
    else:
        return pretty_midi.note_number_to_name(number)

def extract_notes_from_melody(instrument, window_size=50, fs=5, training_output=True):
    """
    Extract the notes strings from the melody instrument

    :param instrument: the object that contain the note information
    :param window_size: size of output "sentence"
    :param fs: the rate to sample from the midi
    :param training_output: if True - extract sentences in window_size size, if False - extract all of the notes in one
                            list
    :return: notes in string format
    """

    # Extract all of the notes of the instrument
    instrument_timeframes = instrument.get_piano_roll(fs=fs)

    # find where is the first note
    melody_start = np.min(np.where((np.sum(instrument_timeframes, axis=0) > 0)))
    melody_piano_roll = instrument_timeframes[:, melody_start:]

    # ignore the velocity of the melody
    melody_piano_roll = (melody_piano_roll > 0).astype(float)

    # add an index for the rest notes, and assign 1 in those indexes
    rests = np.sum(melody_piano_roll, axis=0)
    rests = (rests == 0).astype(float)
    melody_piano_roll = np.insert(melody_piano_roll, 128, rests, axis=0)

    # if training_output=True, split the samples to windows otherwise extract one list with all of the notes
    if training_output:
        X = []
        for i in range(0, melody_piano_roll.shape[1] - window_size):
            window = melody_piano_roll[:, i:i + window_size]
            X.append([number_to_note(np.where(note==1)[0][0]) for note in window.T])
        return np.array(X)
    else:
        return [number_to_note(np.where(note == 1)[0][0]) for note in melody_piano_roll.T]


def extract_notes_from_harmony(instrument, window_size=200, fs=5, training_output=True):
    """
    Extract the notes strings from the melody instrument

    :param instrument: the object that contain the note information
    :param window_size: size of output "sentence"
    :param fs: the rate to sample from the midi
    :param training_output: if True - extract sentences in window_size size, if False - extract all of the notes in one
                            list
    :return: notes in string format
    """

    # Extract all of the notes of the instrument
    instrument_timeframes = instrument.get_piano_roll(fs=fs)

    # find where is the first note
    harmony_start = np.min(np.where((np.sum(instrument_timeframes, axis=0) > 0)))
    harmony_piano_roll = instrument_timeframes[:, harmony_start:]

    # ignore the velocity of the melody
    harmony_piano_roll = (harmony_piano_roll > 0).astype(float)

    # add an index for the rest notes, and assign 1 in those indexes
    rests = np.sum(harmony_piano_roll, axis=0)
    rests = (rests == 0).astype(float)
    harmony_piano_roll = np.insert(harmony_piano_roll, 128, rests, axis=0)

    # if training_output=True, split the samples to windows otherwise extract one list with all of the notes
    if training_output:
        X = []
        for i in range(0, harmony_piano_roll.shape[1] - window_size):
            window = harmony_piano_roll[:, i:i + window_size]
            X.append(['-'.join([number_to_note(note_num) for note_num in np.where(note==1)[0]]) for note in window.T])
        return np.array(X)
    else:
        return ['-'.join([number_to_note(note_num) for note_num in np.where(note==1)[0]]) for note in harmony_piano_roll.T]

def prepare_midi_embeddings_dataset(fs=10):
    # prepare 3 different samples - drums, harmony and melody
    X_drums = []
    X_melody = []
    X_harmony = []

    list_midi_files = os.listdir(os.path.join(DATA_PATH, MIDI_PATH))
    for midi_file in tqdm(list_midi_files, total=len(list_midi_files)):
        # load the midi file
        try:
            midi_obj = pretty_midi.PrettyMIDI(os.path.join(DATA_PATH, MIDI_PATH, midi_file))
        except Exception as e:
            print('Error in loading {} file: {}'.format(midi_file, e))
            continue

        # parse each one of the instruments
        for inst in midi_obj.instruments:

            # if that drums we need to have special handling
            if inst.is_drum:
                inst.is_drum = False
                # check that notes give information
                if np.count_nonzero(inst.get_piano_roll(fs=fs)) == 0:
                    continue

                X = extract_notes_from_harmony(inst, fs=fs)
                X_drums.append(X)

                inst.is_drum = True
                continue

            # if its not drums and there is no notes - dont use it
            if np.count_nonzero(inst.get_piano_roll(fs=fs) != 0) == 0:
                continue

            # now check if that instrument is melody or harmony
            is_melody = check_if_melody(inst)
            if is_melody == True:
                X = extract_notes_from_melody(inst, fs=fs)
                X_melody.append(X)
            elif is_melody == False:
                X = extract_notes_from_harmony(inst, fs=fs)
                X_harmony.append(X)
            else:
                # Instrument is too quiet
                continue

    return X_drums, X_melody, X_harmony


def get_song_vector(midi_path, models, fs=10):
    # load the doc2vec models
    drum_model = models['drums']
    melody_model = models['melody']
    harmony_model = models['harmony']

    # extract the notes from the instruments in the midi_file
    midi_obj = pretty_midi.PrettyMIDI(midi_path)
    melody_notes = []
    harmony_notes = []
    drums_notes = []

    for inst in midi_obj.instruments:

        # if that drums we need to have special handling
        if inst.is_drum:
            inst.is_drum = False
            # check that notes give information
            if np.count_nonzero(inst.get_piano_roll(fs=fs)) == 0:
                continue

            drums_notes += extract_notes_from_harmony(inst, fs=fs, training_output=False)

            inst.is_drum = True
            continue

        # if its not drums and there is no notes - dont use it
        if np.count_nonzero(inst.get_piano_roll(fs=fs) != 0) == 0:
            continue

        # now check if that instrument is melody or harmony
        is_melody = check_if_melody(inst)
        if is_melody == True:
            melody_notes += extract_notes_from_melody(inst, fs=fs, training_output=False)
        elif is_melody == False:
            harmony_notes += extract_notes_from_harmony(inst, fs=fs, training_output=False)
        else:
            # Instrument is too quiet
            continue

    drums_embedding = drum_model.infer_vector(drums_notes)
    melody_embedding = melody_model.infer_vector(melody_notes)
    harmony_embedding = harmony_model.infer_vector(harmony_notes)
    return np.hstack([drums_embedding, melody_embedding, harmony_embedding])

# OPTION 2 - EXTRACT GLOVE EMBEDDING

def getNotesEmbedded(embeddings_dict, notesList, dim_size=300):
    total_vec = np.zeros(dim_size)
    total_num_of_notes = 0
    notesString = ' '.join(notesList)
    notes_by_instr = notesString.split('TRACK_START ')
    for all_notes in notes_by_instr:
        if all_notes == '':
            continue
        notes = all_notes.split(' ')
        instr_vec = np.zeros(dim_size)
        num_of_notes = 0
        for index, note in enumerate(notes):
            num_of_notes += 1
            if note == '':
                continue
            instr_vec += embeddings_dict[note] if note in embeddings_dict else embeddings_dict['<unk>']
        total_num_of_notes += num_of_notes
        total_vec += instr_vec

    noteEmbeddings = total_vec / total_num_of_notes
    return noteEmbeddings

def parse_midi(path):
    midi = None
    try:
        midi = pretty_midi.PrettyMIDI(path)
        midi.remove_invalid_notes()
    except:
        pass
    return midi

def get_percent_monophonic(pm_instrument_roll):
    mask = pm_instrument_roll.T > 0
    notes = np.sum(mask, axis=1)
    n = np.count_nonzero(notes)
    single = np.count_nonzero(notes == 1)
    if single > 0:
        return float(single) / float(n)
    elif single == 0 and n > 0:
        return 0.0
    else:  # no notes of any kind
        return 0.0

def filter_monophonic(pm_instruments, percent_monophonic=0.99):
    return [i for i in pm_instruments if get_percent_monophonic(i.get_piano_roll()) >= percent_monophonic]

def get_note_string(path):
    midi = parse_midi(path)
    if midi is not None:
        buff = ''
        for instrument in filter_monophonic(midi.instruments, 0):
            buff += 'TRACK_START '
            buff += ' '.join([str(n.pitch) for n in instrument.notes]) + ' '
        return buff
    return ''

def get_notes_embeddings(embeddings_dict, midi_path, dim_size=300):
    notes_string = get_note_string(midi_path)
    if not notes_string:
        return ''
    total_vec = np.zeros(dim_size)
    total_num_of_notes = 0
    notes_by_instr = notes_string.split('TRACK_START ')
    for all_notes in notes_by_instr:
        if all_notes == '':
            continue
        notes = all_notes.split(' ')
        instr_vec = np.zeros(dim_size)
        num_of_notes = 0
        for note in enumerate(notes):
            num_of_notes += 1
            if note == '':
                continue
            instr_vec += embeddings_dict[note] if note in embeddings_dict else embeddings_dict['<unk>']
        total_num_of_notes += num_of_notes
        total_vec += instr_vec
    return total_vec / total_num_of_notes

# function to build a embedding to midi files folders - based on glove embedding - 300 dim
# for more information: https://github.com/brangerbriz/midi-glove
# extract glove embedding
def ExtractGloveEmbeddingDict(noteEmbeddingsUrl=NOTE_EMBEDDING_PATH):
    embeddings_dict = {}
    with open(noteEmbeddingsUrl) as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            emb = line.split(' ')
            embeddings_dict[emb[0]] = np.array([float(num) for num in emb[1:]])
    return embeddings_dict

# build embedding dict to all the files based on glove embedding
def create_note_embedding_dict(noteEmbeddingsUrl=NOTE_EMBEDDING_PATH):
    emb_dict = ExtractGloveEmbeddingDict(noteEmbeddingsUrl=noteEmbeddingsUrl)
    embedding_dict = {}
    for i, song in enumerate(tqdm(os.listdir(DIR_MELODY))):
        midi_path = os.path.join(DIR_MELODY, song)
        song_embedding = get_notes_embeddings(emb_dict, midi_path, dim_size=300)
        embedding_dict[song] = song_embedding
    return embedding_dict
