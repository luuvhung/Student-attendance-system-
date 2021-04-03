# Hàm load model
## Load model từ Caffe
import cv2
import os
import numpy as np
import pickle
from keras.models import load_model

EMBEDDING_FL = "facenet_keras.h5"
#load model
model = load_model(EMBEDDING_FL)

def _load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj
def _save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

faces=_load_pickle("./faces.pkl")

def get_embedding(model, faces):
    # scale pixel values
    emb_vecs = []
    for face in faces:
        face=cv2.resize(face,(160,160))
        face = face.astype('float32')
        # standardization
        mean, std = face.mean(), face.std()
        face = (face-mean)/std
        # transfer face into one sample (3 dimension to 4 dimension)
        sample = np.expand_dims(face, axis=0)
        # make prediction to get embedding
        vec = model.predict(sample)
        emb_vecs.append(vec[0])
    return emb_vecs
embed_faces = get_embedding(model, faces)
print(type(embed_faces))
_save_pickle(embed_faces, "./embed_faces.pkl")