from tensorflow.keras.models import load_model
import tensorflow as tf
import argparse
import os

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--action",
        default=False,
        type=bool,
        help="'True' to unfreeze; ignore to freeze."
    )
    ap.add_argument(
        "--model_path",
        required=True,
        help="Path to model file"
    )
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = get_arguments()
    model = load_model(args['model_path'], compile=False)

    for layer in model.layers[:-1]:
        layer.trainable = args['action'] # False

    for layer in model.layers[:-1]:
        print(layer, layer.trainable)

    print(model.layers[-1], model.layers[-1].trainable)
