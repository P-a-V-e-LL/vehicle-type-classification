import os
import argparse
#import tensorflow as tf
import cv2
import numpy as np
from classifier import Classifier

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-mp",
        "--model_path",
        required=True,
        help="Path to .h5 keras model."
    )
    ap.add_argument(
        "--test_image_path",
        default="./data/dataset3.11_new_split/val/LADA VAZ Kalina/car_903.jpg",
        help="Path to test image."
    )
    ap.add_argument(
        "--test_dir_path",
        default="./data/d3.11nst/test/",
        help="Path to test directory with images."
    )
    ap.add_argument(
        "-pfp",
        "--pickle_file_path",
        required=True,
        help="Path to .pickle data file."
    )
    return vars(ap.parse_args())

def recall1(class1, class2):
    if class1 == class2:
        return 1
    return 0

def image_to_np(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image = np.asarray(image)
    return image

def main():
    args = get_arguments()
    o = Classifier(args['model_path'], args['pickle_file_path'], 16)
    recall_metric = 0
    recall_count = 0
    time_list = []
    time_count = 0
    sl = 0
    for cl in os.listdir(args['test_dir_path']):
        path = os.path.join(args['test_dir_path'], cl)
        for car in os.listdir(path):
            embedding, d = o.get_predict(image_to_np(os.path.join(path, car)))
            sl += d
            #embedding = o.get_predict(image_to_np(os.path.join(path, car)))
            recall_count += 1
            time_count += 1
            recall_metric += recall1(embedding['model'], cl)
            time_list.append(embedding['recognize_time'])
    print("RECALL@1 ", recall_metric/recall_count*100)
    print("Среднее время распознавания изображения ", sum(time_list)/time_count)
    print("NULL images = ", sl)
    #image = image_to_np(args['test_image_path'])

    #print(o.get_predict(image))

def solo_img():
    '''Для определения дистанции не к выборке, а к одному изображению.'''
    args = get_arguments()
    o = Classifier(args['model_path'], args['pickle_file_path'], 16)
    emb1 = o.get_embedding_vector(image_to_np(args['test_image_path']))
    emb2 = o.get_embedding_vector(image_to_np(args['test_image_path']))
    print(o.get_distance(emb1, emb2))
    #embedding, d = o.get_predict(image_to_np(args['test_image_path']))
    #print(embedding)
    #print(embedding['model'], d)

if __name__ == '__main__':
    main()
    #solo_img()
