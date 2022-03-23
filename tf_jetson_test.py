import os
import sys
from PIL import Image
from numpy import asarray
#from tensorflow.keras.preprocessing import image
import numpy as np
from numpy import expand_dims
import argparse
import time
import tflite_runtime.interpreter as tflite


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-mp",
        "--model_path",
        #required=True,
        default="./models/facenet_l2_bn.tflite",
        help="Path to tflite model."
    )
    ap.add_argument(
        "--data_path",
        default="./data/dataset3.11_new_split/val/GAZ Volga/",
        help="Path to class directory."
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
    time_list = []
    time_sum = 0
    interpreter = tflite.Interpreter(model_path=args['model_path'])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    count = len(os.listdir(args['data_path']))
    for image in os.listdir(args['data_path']):
        image_path = os.path.join(args['data_path'], image)
        start = time.time()

        input_data = np.array(prep(image_path), dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        a = time.time()-start
        print(a)
        time_list.append(a)
        time_sum += a
        #print(output_data)
    print("-"*100)
    print("Одно изображение обработано за: ", sum(time_list)/len(time_list))
    print("Изображений в секунду обработано: ", count/time_sum)

if __name__ == '__main__':
    main()
