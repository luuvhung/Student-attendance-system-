from imutils import paths
import cv2
import pickle
import numpy as np
DATASET_PATH = "./Database"

def _model_processing():
    image_links = list(paths.list_images(DATASET_PATH))
    images_file = [] 
    y_labels = []
    faces = []

    for image_link in image_links:
        split_img_links = image_link.split("\\")
    # Lấy nhãn của ảnh
        name = split_img_links[-2] 
    # Đọc ảnh
        face = cv2.imread(image_link)
        faces.append(face)
        y_labels.append(name)
        images_file.append(image_links)
    return faces, y_labels, images_file
print(y_label)
faces, y_labels, images_file = _model_processing()

def _save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def _load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj

_save_pickle(faces, "./faces.pkl")
_save_pickle(y_labels, "./y_labels.pkl")
_save_pickle(images_file, "./images_file.pkl")