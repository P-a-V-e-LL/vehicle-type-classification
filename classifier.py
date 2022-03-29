import os
import math
import time
import numpy as np
from PIL import Image
import pickle
#import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model

def pickle_to_data(fname):
    '''Выгружает данные из файла в словарь.'''
    f = open(fname, "rb")
    x = pickle.load(f)
    f.close()
    return x

class Classifier:

    def __init__(self, model_path: str, pickle_data_path: str, min_distance: int, required_size=(160, 160)):
        #self.model_path = model_path
        #self.pickle_data_path = pickle_data_path
        self.model = load_model(model_path, compile=False)
        self.pickle_data = pickle_to_data(pickle_data_path)
        self.required_size = required_size
        self.min_distance = min_distance # задать значение по умолчанию после выяснения данного параметра у текущей модели.

    def get_distance(self, embedding_1, embedding_2):
        '''Вычисляет расстояние между двумя векторами.'''
        return np.linalg.norm(embedding_1 - embedding_2)

    def get_embedding(self, image):
        '''Получает вектор embedding из изображения.'''
        #image = Image.open(filename)
        #image = asarray(image) # изображение поступает в классификатор в формате numpy массива
        image = Image.fromarray(image)
        image = image.resize(self.required_size)
        car_array = np.asarray(image)
        car_array = car_array.astype('float32')
        #print(np.unique(car_array))
        #print("-"*100)
        #mean, std = car_array.mean(), car_array.std()
        #car_array = (car_array - mean) / std
        for i in range(len(car_array)):
            car_array[i] /= 255
        #print(np.unique(car_array))
        #print("-"*100)
        samples = np.expand_dims(car_array, axis=0)
        samples = np.stack((samples.copy(),)*3, axis=-1)
        yhat = self.model.predict(samples)
        return yhat[0]

    def add_distance(self, name, dist, nearest_list, n):
        '''Добавляет дистанцию в список ближайших если значение меньше максимального в списке.'''
        if len(nearest_list) < n:
            nearest_list.append({"class_name": name, "distance": dist})
            return nearest_list
        if dist < nearest_list[-1]["distance"]:
            nearest_list.append({"class_name": name, "distance": dist})
            nearest_list = sorted(nearest_list, key=lambda a: a['distance'],  reverse=False)
            if len(nearest_list) > n:
                nearest_list.pop(-1)
        return nearest_list

    def get_predict(self, image):
        knn_list = []
        start_time = time.time()
        target_embedding = self.get_embedding(image)

        for car in self.pickle_data.keys():
            for embedding in self.pickle_data[car]:
                distance = self.get_distance(target_embedding, embedding)
                if distance < self.min_distance:
                    knn_list = self.add_distance(car, distance, knn_list, 11)
        recognize_time = time.time() - start_time
        return {'model':knn_list[0]["class_name"], 'recognize_time': recognize_time, 'debug_info': knn_list[1:]}
