'''
[{'class_name': 'lada', 'distanse': 0.4}, {...}]]
'''

import os
import sys
from tensorflow.keras.models import load_model
from PIL import Image
from numpy import asarray
from tensorflow.keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt
from numpy import expand_dims
import pickle
import argparse

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-mp",
        "--model_path",
        required=True,
        help="Path to .h5 keras model."
    )
    ap.add_argument(
        "-tip",
        "--test_image_path",
        required=True,
        help="Path to test image."
    )
    ap.add_argument(
        "-pfp",
        "--pickle_file_path",
        required=True,
        help="Path to .pickle data file."
    )
    ap.add_argument(
        "--min_dist",
        type=int,
        default=0.6,
        help="Min distanse between classes."
    )
    ap.add_argument(
        "--knn_count",
        type=int,
        default=10,
        help="Nearest neighbours count."
    )
    return vars(ap.parse_args())

def get_v(filename, model, required_size=(160, 160)):
  image = Image.open(filename)
  image = asarray(image)
  image = Image.fromarray(image)
  image = image.resize(required_size)
  car_array = asarray(image)
  car_array = car_array.astype('float32')
  mean, std = car_array.mean(), car_array.std()
  car_array = (car_array - mean) / std
  samples = expand_dims(car_array, axis=0)
  samples = np.stack((samples.copy(),)*3, axis=-1)
  yhat = model.predict(samples)
  return yhat[0]

def pickle_to_data(fname):
    '''Выгружает данные из файла в словарь.'''
    f = open(fname, "rb")
    x = pickle.load(f)
    f.close()
    return x

def get_distance(emb1, emb2):
    '''Вычисляет расстояние между векторами.'''
    return np.linalg.norm(emb1 - emb2)/20

def add_distance(name, dist, nearest_list, n):
    '''Добавляет дистанцию в список ближайших если значение меньше максимального в списке.'''
    if len(nearest_list) < n:
        nearest_list.append({"class_name": name, "distance": dist})
        #nearest_list = sorted(nearest_list, key=lambda a: a['distance'], reverse=False)
        return nearest_list
    if dist < nearest_list[-1]["distance"]:
        nearest_list.append({"class_name": name, "distance": dist})
        nearest_list = sorted(nearest_list, key=lambda a: a['distance'],  reverse=False)
        if len(nearest_list) > n:
            nearest_list.pop(-1)
    return nearest_list

def print_dist(l):
    for i in l:
        print("{0}:\t\t\t {1}".format(i["class_name"], i["distance"]))

def main():
    args = get_arguments()
    model = load_model(args["model_path"], compile=False)
    nearest_list = []
    class_embeddings = pickle_to_data(args["pickle_file_path"])
    test_embedding = get_v(args["test_image_path"], model)

    for i in class_embeddings.keys():
        for embedding in class_embeddings[i]:
            distanse = get_distance(test_embedding, embedding['embedding'])
            if (distanse < args["min_dist"]):
                nearest_list = add_distance(i, distanse, nearest_list, args["knn_count"])
    print_dist(nearest_list)


if __name__ == '__main__':
    main()
