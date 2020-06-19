import joblib
import numpy as np

def prepare_doc2vec(X):
    """
    Trainig a Doc2Vec model where doc == song

    :param X: Songs Lyrics
    :return: Doc2Vec model
    """
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(X)]
    model = Doc2Vec(documents, vector_size=50, window=5, min_count=1, workers=4)
    return model

def build_melody_embedding_models():
    # load the entire melody, harmony and drums samples we extracted from the songs
    data = joblib.load('midi_preprocess_data.joblib')

    # Extract each one of the samples type seperatly 
    X_drums = np.vstack([i for i in data[0] if len(i.shape) > 1])
    X_melody = np.vstack([i for i in data[1] if len(i.shape) > 1])
    X_harmony = np.vstack([i for i in data[2] if len(i.shape) > 1])

    # Train the Doc2Vec models
    drums_model = prepare_doc2vec(X_drums)
    melody_model = prepare_doc2vec(X_melody)
    harmony_model = prepare_doc2vec(X_harmony)

    # Dump the models for future loading
    joblib.dump(drums_model ,'Data/Models/drums_model.joblib')
    joblib.dump(melody_model ,'Data/Models/melody_model.joblib')
    joblib.dump(harmony_model,'Data/Models/harmony_model.joblib')
