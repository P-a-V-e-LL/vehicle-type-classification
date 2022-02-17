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

model_path = "./models/model_20211203_rmsprop.h5"   # путь к модели
test_image = r'/home/pavel/Desktop/University/Diplom/test (not in val)/dataset/Chevrolet Captiva/Chevrolet Captiva 2013_81616476.jpg'       # путь к изображению для распознавания
pickle_file = "./embedding_data/dataset3.11_val.pickle"               # путь к .pickle файлу сохраненных эмбеддингов

model = load_model(model_path, compile=False)

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

def add_distance(name, dist, nearest_list):
    '''Добавляет дистанцию в список ближайших если значение меньше максимального в списке.'''
    if len(nearest_list) < 10:
        nearest_list.append({"class_name": name, "distance": dist})
        #nearest_list = sorted(nearest_list, key=lambda a: a['distance'], reverse=False)
        return nearest_list
    if dist < nearest_list[-1]["distance"]:
        nearest_list.append({"class_name": name, "distance": dist})
        nearest_list = sorted(nearest_list, key=lambda a: a['distance'],  reverse=False)
        if len(nearest_list) > 10:
            nearest_list.pop(-1)
    return nearest_list

def print_dist(l):
    for i in l:
        print("{0}:\t\t {1}".format(i["class_name"], i["distance"]))

def main():
    nearest_list = []
    class_embeddings = pickle_to_data(pickle_file)
    test_embedding = get_v(test_image, model)

    for i in class_embeddings.keys():
        #if i == "Chevrolet Niva":
        for embedding in class_embeddings[i]:
            distanse = get_distance(test_embedding, embedding)
            if (distanse < 0.6):
                nearest_list = add_distance(i, distanse, nearest_list)
    print_dist(nearest_list)

main()
