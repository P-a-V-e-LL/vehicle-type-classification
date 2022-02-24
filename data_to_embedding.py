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
        "--model_path",
        required=True,
        help="Path to .h5 keras model."
    )
    ap.add_argument(
        "--root_dir",
        required=True,
        help="Path to data dir."
    )
    ap.add_argument(
        "--all",
        default=False,
        help="All images flag."
    )
    ap.add_argument(
        "--data",
        default=False,
        help="All images flag."
    )
    ap.add_argument(
        "--n",
        type=int,
        default=1,
        help="Class images amount."
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

def get_min_max_val_img(dir, classes):
    '''
    Возвращает максимальное и минимальное количество изображений в классах.
    dir - папка выборки данных;
    classes - список классов из root_dir.
    '''
    classes_count = len(classes)
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

def class_to_embedding_list(dir, n, model):
    '''
    Преобразовывает класс (папку с изображениями) в список n векторов.
    dir - папка класса (val/dir);
    n - количество изображений для извлечения векторов.
    Возвращает список векторов.
    '''
    embeddings = []
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

def data_to_pickle(root_dir, model, name, n=0, max=False):
    '''
    Основная функция для исполнения.
    Записывает каждый класс в файл формата .pickle как список векторов.
    root_dir - папка выборки данных.
    Записываются значения вида {class_name: [embeddings_list]}.
    '''
    cl = os.listdir(root_dir)
    class_embeddings = {}
    for c in cl:
        if max:
            nn = len(os.listdir(os.path.join(root_dir, c)))
            class_embeddings[c] = class_to_embedding_list(os.path.join(root_dir, c), nn, model)
        else:
            class_embeddings[c] = class_to_embedding_list(os.path.join(root_dir, c), n, model)
        print("{0} сохранен!".format(c))
    #f = open("./embedding_data/" + data_name + ".pickle", "wb+")
    f = open("./embedding_data/" + name + ".pickle", "wb+")
    pickle.dump(class_embeddings, f)
    f.close()
    print("Данные сохранены!")

def pickle_to_data(fname):
    f = open(fname, "rb")
    x = pickle.load(f)
    f.close()
    return x

def main():
    args = get_arguments()
    model = load_model(args['model_path'], compile=False)
    classes = os.listdir(args['root_dir'])
    classes_count = len(classes)
    max, min = get_min_max_val_img(args['root_dir'], classes)
    if args['all']:
        #data_to_pickle(args['root_dir'], max, model, args['data'])
        data_to_pickle(args['root_dir'], model, args['data'], max=True)
    else:
        data_to_pickle(args['root_dir'], model, args['data'], n=args['n'],)

if __name__ == '__main__':
    main()
