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

model_path = "./models/model_20211203_rmsprop.h5"
root_dir_val = "../dataset3.11_2/val"

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

classes = os.listdir(root_dir_val)
classes_count = len(classes)
model = load_model(model_path, compile=False)


def get_min_max_val_img(dir, classes):
    '''
    Возвращает максимальное и минимальное количество изображений в классах.
    dir - папка выборки данных;
    classes - список классов из root_dir.
    '''
    min = 1000000
    max = -1
    i = 0
    val = os.listdir(dir)
    while i < classes_count:
        for auto in os.listdir(os.path.join(dir, val[i])):
          o = os.path.join(dir, val[i])
          #o = os.path.join(o, auto)
          if len(os.listdir(o)) > max:
              max = len(os.listdir(o))
          if len(os.listdir(o)) < min:
              min = len(os.listdir(o))
        i += 1
    return max, min

def class_to_embedding_list(dir, n):
    '''
    Преобразовывает класс (папку с изображениями) в список n векторов.
    dir - папка класса (val/dir);
    n - количество изображений для извлечения векторов;
    name - название класса.
    Возвращает список векторов.
    '''
    embeddings = []
    #embeddings = [name]
    i = 0
    for img in os.listdir(dir):
        if i != n:
            o = os.path.join(dir, img)
            emb = get_v(o, model)
            embeddings.append(emb)
            i += 1
        else:
            break
    return embeddings

def choise(dir, classes):
    max, min = get_min_max_val_img(dir, classes)
    print("Введите 1 чтобы сдлать сохранение одного изображения из класса.")
    print()
    print("Введите 2 чтобы сдлать сохранение n изображений из класса.")
    print()
    print("Введите 3 чтобы сдлать сохранение всех изображений из класса.")
    print()
    c = int(input("Выберите вариант: "))
    if c == 1:
        n = 1
    elif c == 2:
        print("Минимальное возможное значение - {0}, максимальное - {1}".format(min, max))
        print()
        n = int(input("Ввдеите n: "))
    elif c == 3:
        n = max
    else:
        print("Ошибка!")
    return n

def data_to_pickle(root_dir):
    '''
    Основная функция для исполнения.
    Записывает каждый класс в файл формата .pickle как список векторов.
    root_dir - папка выборки данных.
    Записываются значения вида {class_name: [embeddings_list]}.
    '''
    data_name = input("Введите название набора данных: ")
    cl = os.listdir(root_dir)
    class_embeddings = {}
    ch = choise(root_dir, cl)
    for c in cl:
        class_embeddings[c] = class_to_embedding_list(os.path.join(root_dir, c), ch)
        print("{0} сохранен!".format(c))
    f = open("./embedding_data/" + data_name + ".pickle", "wb+")
    pickle.dump(class_embeddings, f)
    f.close()
    print("Данные сохранены!")

def pickle_to_data(fname):
    f = open(fname, "rb")
    x = pickle.load(f)
    f.close()
    return x

data_to_pickle(root_dir_val)
