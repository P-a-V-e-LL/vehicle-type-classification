from tensorflow.keras.models import load_model
import argparse
import os

model = load_model('./models/l2_facenet.h5')

for layer in model.layers[:-1]:
    layer.trainable = False

for layer in model.layers[:-1]:
    print(layer, layer.trainable)

print(model.layers[-1], model.layers[-1].trainable)
