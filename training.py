import cv2
import pickle
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from random import choice

def _save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def _load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def train():
    embed_faces = _load_pickle("./embed_faces.pkl")
    y_labels = _load_pickle("./y_labels.pkl")
    embed_faces=np.asarray(embed_faces)
    y_labels=np.asarray(y_labels)
    print(embed_faces.shape)
    #covert du lieu y_labels
    output_enc = LabelEncoder()
    output_enc.fit(y_labels)
    y_labels = output_enc.transform(y_labels)
    _save_pickle(output_enc, "./output_enc.pkl")

    ids = np.arange(len(y_labels))

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(np.stack(embed_faces), y_labels, ids, test_size = 0.1, stratify = y_labels)

    model = SVC(kernel='linear',probability=True)
    model.fit(X_train, y_train)
    _save_pickle(model, "./modelsvm.pkl")
    print("da luu model")
    print(y_test)
    yhat_test = model.predict(X_test)
    score_test = accuracy_score(y_test, yhat_test)
    print('Accuracy=%.3f' % (score_test*100))
