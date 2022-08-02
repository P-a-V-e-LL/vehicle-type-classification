import os
from PIL import Image
import numpy as np
import cv2
import argparse

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root_dir",
        required=True,
        help="Path to data."
    )
    return vars(ap.parse_args())

def main():
    args = get_arguments()
    train_dir = args['root_dir'] + "/train"
    val_dir = args['root_dir'] + "/val"
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

if __name__ == '__main__':
    main()
