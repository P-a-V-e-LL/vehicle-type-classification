import os
import sys
from tensorflow.keras.models import load_model
import argparse
import tensorflow as tf

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-mp",
        "--model_path",
        required=True,
        help="Path to .h5 keras model."
    )
    ap.add_argument(
        "--filename",
        required=True,
        help="Name to new model."
    )
    return vars(ap.parse_args())

def main():
    args = get_arguments()
    model = load_model(args['model_path'], compile=False)
    new_model = tf.lite.TFLiteConverter.from_keras_model(model)
    tflmodel = new_model.convert()
    file = open( './models/'+args['filename']+'.tflite' , 'wb+' )
    file.write(tflmodel)
    print("Done.")

if __name__ == '__main__':
    main()
