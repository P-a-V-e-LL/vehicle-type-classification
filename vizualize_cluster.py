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
import argparse

#matplotlib.use("TkAgg")

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-mp",
        "--model_path",
        required=True,
        help="Path to .h5 keras model."
    )
    ap.add_argument(
        "--root_dir",
        required=True,
        help="Path to data."
    )
    ap.add_argument(
        "--filename",
        required=True,
        help="Data filename."
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

def main():
    args = get_arguments()

    i = 0
    y = 0

    model = load_model(args['model_path'], compile=False)

    classes_count = len(os.listdir(args['root_dir']))

    classes = os.listdir(args['root_dir'])

    print("Counting embeddings...")

    while i < classes_count:
      print('{0}/{1}'.format(i, classes_count))
      for auto in os.listdir(os.path.join(args['root_dir'], classes[i])):
        o = os.path.join(args['root_dir'], classes[i])
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
        print(auto)
      i += 1

    e = a.reshape(y, 128)

    print("Consuming results...")

    embeddings = TSNE(n_jobs=16).fit_transform(e)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]

    plt.figure(300) # открытие нового окна для отображения графика

    plt.scatter(vis_x, vis_y, c=b, cmap=plt.cm.get_cmap("jet", classes_count), marker='.')
    plt.colorbar(ticks=range(classes_count))
    plt.clim(-0.5, classes_count-0.5)
    plt.savefig(args['filename'], dpi=300)
    #plt.show()

if __name__ == '__main__':
    main()
