from imutils import paths
import cv2
import pickle
from keras.models import load_model
import numpy as np
import argparse
import csv
import os
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False 

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

def _save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def _load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def addSV(Id,name): 
    if(is_number(Id) and name.isalpha()):
        directory = name
        
            # Parent Directory path 
        parent_dir = "database/"
        parent_dir1 = "database1/"
        
            # Path 
        path = os.path.join(parent_dir, directory) 
        os.mkdir(path)
        path = os.path.join(parent_dir1, directory) 
        os.mkdir(path) 
        print("Thu muc '% s' da duoc tao" % directory)
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
        cap = cv2.VideoCapture(0)
        sampleNum=0
        takephoto=0
        while(True):
            _, frame = cap.read()
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
                        #incrementing sample number
                    if takephoto==1:
                        sampleNum=sampleNum+1
                            #saving the captured face in the dataset folder TrainingImage
                        cv2.imwrite("database/"+directory+"/"+ str(sampleNum) + ".jpg", frame[startY:endY,startX:endX])
                        cv2.imwrite("database1/"+directory+"/"+"a"+ str(sampleNum) + ".jpg", frame)
                    cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,str(sampleNum),(50,50), font, 1,(0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame,'Nhan nut C de chup anh',(200,50), font, 0.6,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('frame',frame)
                #wait for 100 miliseconds
            if cv2.waitKey(1) & 0xFF == ord('c'):
                takephoto=1
                # break if the sample number is morethan 100
            if sampleNum>99:
                break
        cap.release()
        cv2.destroyAllWindows()
        res = "Anh cua ID : " + Id + name +"da duoc luu"
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        print(res)

    y_labels = _load_pickle("./y_labels.pkl")
    embed_faces = _load_pickle("./embed_faces.pkl")
    DATASET_PATH = "./Database/"+directory+"/"

    image_links = list(paths.list_images(DATASET_PATH))
    faces = []
    dem=0
    for image_link in image_links:
        # Lấy nhãn của ảnh
        name = directory
        dem = dem+1
        print(dem)
        # Đọc ảnh
        face = cv2.imread(image_link)
        faces.append(face)
        y_labels.append(name)


    EMBEDDING_FL = "facenet_keras.h5"
    #load model
    model = load_model(EMBEDDING_FL)
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
        embed_faces.append(vec[0])

    _save_pickle(embed_faces, "./embed_faces.pkl")
    _save_pickle(y_labels, "./y_labels.pkl")
#Id=input("nhap ID: ")
#name=input("nhap Ten: ")
#addSV(Id,name)