from PIL import Image
from numpy import asarray
import tensorflow as tf
import os
from scipy.spatial import distance
from tensorflow.keras.preprocessing import image
import numpy as np
from numpy import load
from numpy import expand_dims
from numpy import savez_compressed
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model, clone_model
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model("model_20211203_rmsprop.h5", compile=False)

BD = {}

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

def add1(name, img):
    BD[name] = get_v(img, model)
    print(name + " saved.")

def recognize(img1):
    predict = "None"
    min = 100
    for car in BD.keys():
        i = np.linalg.norm(get_v(img1, model) - BD[car])/20
        if min == 100:
            min = i
            predict = car
        if i < min and i < 0.6:
            min = i
            predict = car
    print("It is {0}, {1}".format(predict, min))


path = "/home/pavel/Desktop/University/Diplom/dataset3.11/train/"
os.system("cls" if os.name == 'nt' else "clear")

add1("Niva", os.path.join(path, "Chevrolet Niva/Chevrolet Niva 2017_93701106.jpg")) # Niva Front Gray
add1("Hyndai Creta", os.path.join(path, "Hyndai Creta/Hyundai Creta 2019_94118112.jpg")) # Front White
add1("Kia Sportage", os.path.join(path, "Kia Sportage/Kia Sportage 2012_11836954.jpg")) # Kia Sporage Front Gray
add1("LADA VAZ Lardus", os.path.join(path, "LADA VAZ Largus/LADA (ВАЗ) Largus 2019_77264105.jpg")) # Largus Front white

print()
recognize(os.path.join(path, "Chevrolet Niva/Chevrolet Niva 2017_99598886.jpg")) # Niva Front White
recognize(os.path.join(path, "Chevrolet Niva/Chevrolet Niva 2017_63775080.jpg")) # Niva Back Gray
recognize(os.path.join(path, "Chevrolet Niva/car_73548.jpg")) # Niva Back White Fog
recognize(os.path.join(path, "Chevrolet Niva/car_70214.jpg")) # Niva Front Black Noise

recognize(os.path.join(path, "Hyndai Creta/car_93749.jpg")) # Hyndai Creta Front Gray

recognize(os.path.join(path, "Kia Sportage/car_92196.jpg")) # Kia SportAge Gray Fog Mirror
recognize(os.path.join(path, "Kia Sportage/car_87548.jpg")) # Kia SportAge Gray angle
