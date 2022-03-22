import tensorflow as tf
import os
import sys
from PIL import Image
from numpy import asarray
#from tensorflow.keras.preprocessing import image
import numpy as np
from numpy import expand_dims
import argparse

#model_path = "./models/facenet_l2_bn.tflite"
image_path = "./data/dataset3.11_new_split/val/GAZ Volga/car_710.jpg"

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-mp",
        "--model_path",
        required=True,
        help="Path to tflite model."
    )
    return vars(ap.parse_args())

def prep(filename, required_size=(160, 160)):
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
  return samples


def main():
    args = get_arguments()
    interpreter = tf.lite.Interpreter(model_path=args['model_path'])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array(prep(image_path), dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)

if __name__ == '__main__':
    main()
