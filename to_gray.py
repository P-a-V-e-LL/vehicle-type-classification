import os
from PIL import Image
import numpy as np
import cv2


data_dir = "/home/pavel/Desktop/University/Diplom/cars196"
train_dir = data_dir + "/train"
val_dir = data_dir + "/val"


for car_class in os.listdir(train_dir):
    print(car_class)
    item = os.path.join(train_dir, car_class)
    x = os.listdir(item)
    for img in x:
        target = os.path.join(item, img)
        im_gray = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
        #d = Image.open(target)
        #arr = np.asarray(d, dtype='uint8')
        #x, y, _ = arr.shape

        #k = np.array([[[0.2989, 0.587, 0.114]]])
        #arr2 = np.round(np.sum(arr*k, axis=2)).astype(np.uint8).reshape((x, y))

        #img2 = Image.fromarray(arr2)
        os.remove(target)
        cv2.imwrite(target, im_gray)


for car_class in os.listdir(val_dir):
    print(car_class)
    item = os.path.join(val_dir, car_class)
    x = os.listdir(item)
    for img in x:
        target = os.path.join(item, img)
        im_gray = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
        #d = Image.open(target)
        #arr = np.asarray(d, dtype='uint8')
        #x, y, _ = arr.shape

        #k = np.array([[[0.2989, 0.587, 0.114]]])
        #arr2 = np.round(np.sum(arr*k, axis=2)).astype(np.uint8).reshape((x, y))

        #img2 = Image.fromarray(arr2)
        os.remove(target)
        cv2.imwrite(target, im_gray)
