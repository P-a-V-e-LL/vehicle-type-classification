import os
import argparse
import tensorflow as tf
from PIL import Image
import numpy as np
from classifier import Classifier

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-mp",
        "--model_path",
        required=True,
        help="Path to .h5 keras model."
    )
    ap.add_argument(
        "--test_image_path",
        default="./data/dataset3.11_new_split/val/LADA VAZ Kalina/car_903.jpg",
        help="Path to test image."
    )
    ap.add_argument(
        "-pfp",
        "--pickle_file_path",
        required=True,
        help="Path to .pickle data file."
    )
    return vars(ap.parse_args())

def image_to_np(filename):
    image = Image.open(filename)
    image = np.asarray(image)
    return image

if __name__ == '__main__':
    args = get_arguments()
    o = Classifier(args['model_path'], args['pickle_file_path'], 6)

    image = image_to_np(args['test_image_path'])

    print(o.get_predict(image))
