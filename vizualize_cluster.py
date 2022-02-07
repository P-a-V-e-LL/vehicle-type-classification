import matplotlib
#import tk
import os
import sys
from tensorflow.keras.models import load_model
from PIL import Image
from numpy import asarray
from tensorflow.keras.preprocessing import image
import numpy as np
from numpy import expand_dims
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt

#matplotlib.use("TkAgg")

model_path = "../model_20211203_rmsprop.h5" #path to .h5 model
path = "../dataset3.11_2/val" # data path

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

i = 0
y = 0

classes_count = 10 # len(os.listdir(path))

classes = os.listdir(path)[:10]

print("Counting embeddings...")

while i < classes_count:
  for auto in os.listdir(os.path.join(path, classes[i])):
    o = os.path.join(path, classes[i])
    o = os.path.join(o, auto)
    if y == 0:
      a = np.array([get_v(o, model)])
    else:
      a = np.append(a, get_v(o, model))
    if y == 0:
      b = np.array([i])
    else:
      b = np.append(b, i)
    y += 1
    print("...")
  i += 1

e = a.reshape(y, 128)

print("Consuming results...")

embeddings = TSNE(n_jobs=4).fit_transform(e)
vis_x = embeddings[:, 0]
vis_y = embeddings[:, 1]

plt.figure(300) # открытие нового окна для отображения графика

plt.scatter(vis_x, vis_y, c=b, cmap=plt.cm.get_cmap("jet", classes_count), marker='.')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()
