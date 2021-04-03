import cv2
import pickle
import os
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from random import choice

import datetime
import time
import csv

def _save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def _load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def get_embedding(model, face):
    # scale pixel values
    face=cv2.resize(face,(160,160))
    face = face.astype('float32')
    # standardization
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    vec = model.predict(sample)
    return vec[0]

def diemdanh():
    EMBEDDING_FL = "facenet_keras.h5"
    #load model
    model1 = load_model(EMBEDDING_FL)
    model= _load_pickle("./modelsvm.pkl")
    output_enc=_load_pickle("./output_enc.pkl")
    print("da load xong cac model")
    cam = cv2.VideoCapture(0)
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
    while(True):
        _, frame = cam.read()
        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections =net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.8:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10  
                img1=frame[startY:endY,startX:endX]
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
                face_emb=get_embedding(model1,np.array(img1))
                face_emb = np.expand_dims(face_emb, axis=0)
                name=model.predict(face_emb)
                yhat_prob = model.predict_proba(face_emb)
                class_index = name[0]
                class_probability = round(yhat_prob[0,class_index] * 100,1)
                #lay nhan
                predict_names = output_enc.inverse_transform(name)
                if predict_names!=None:
                    #print(yhat_prob[0])
                    if class_probability>60:
                        person=predict_names[0]+"  "+str(class_probability)+"%"
                        cv2.putText(frame,person, (startX+6, startY-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                        ###
                    else:
                        person="unknow"
                        cv2.putText(frame,person, (startX-10, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()