'''Обучение нейронной сети.'''

#from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image
from numpy import asarray
import tensorflow as tf
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
from tensorflow_addons.losses.metric_learning import pairwise_distance
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import argparse
import os

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-mp",
        "--model_path",
        required=True,
        help="Path to .h5 keras model."
    )
    ap.add_argument(
        "-td",
        "--train_data",
        required=True,
        help="Path to training dataset."
    )
    ap.add_argument(
        "-vd",
        "--val_data",
        required=True,
        help="Path to val dataset."
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size."
    )
    ap.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Training epochs amount."
    )
    return vars(ap.parse_args())

def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)

def check(path):
    i = 0
    for folder in os.listdir(path):
        n = os.path.join(path, folder)
        i += len(os.listdir(n))
    return i

img_width, img_height = 160, 160 				# целевой размер изображения для обучения
save_model_path = './models/'             # путь к месту сохранения обученной модели


def triplet_accuracy(y_true, y_pred):
    batch_size =tf.cast(tf.size(y_true), dtype=tf.dtypes.float32)
    # Build pairwise squared distance matrix
    pdist_matrix = pairwise_distance(y_pred, squared=True) # was pairwise_distance
    # Build pairwise binary adjacency matrix.
    adjacency = tf.cast(tf.math.equal(y_true, tf.transpose(y_true)), dtype=tf.dtypes.float32)
    # Invert so we can select negatives only.
    adjacency_not = 1-adjacency
    # Convert to decision with thresholding at 0.5
    predicted = tf.cast(tf.math.less_equal(pdist_matrix, 0.5), dtype=tf.dtypes.float32)
    # Calculate true positives and true negatives
    true_trues = tf.reduce_sum(tf.cast(
        tf.math.multiply(predicted, adjacency)
        , dtype=tf.dtypes.float32))
    true_falses = tf.reduce_sum(tf.cast(
        tf.math.multiply(1-predicted, adjacency_not)
        , dtype=tf.dtypes.float32))
    # Calculate percentage
    return (true_trues+true_falses)/(batch_size*batch_size)


def main():
    args = get_arguments()
    nb_train_samples = check(args['train_data'])#47734
    nb_val_samples = check(args['val_data'])
    model = load_model(args['model_path'])
    #model = clone_model(model)
    #sgd = SGD(lr=0.001, momentum=0.001, nesterov=True)
    opt = RMSprop(learning_rate=0.001, centered=True)

    model.compile(loss=tfa.losses.TripletSemiHardLoss(),
                  optimizer=opt,  #Adam(learning_rate=1e-5),
    	      metrics=[triplet_accuracy]) # 'accuracy' or triplet_accuracy

    datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen.flow_from_directory(args['train_data'],
                                                  target_size=(img_width, img_height),
                                                  batch_size=args['batch_size'],
                                                  class_mode='sparse')

    val_generator = datagen.flow_from_directory(args['val_data'],
                                                  target_size=(img_width, img_height),
                                                  batch_size=args['batch_size'],
                                                  class_mode='sparse')

    my_callbacks = [
                    tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path + 'model_new_callback.h5',
                                                       monitor='val_loss',
                                                       mode='min',
                                                       save_best_only=True),
    ]

    import datetime
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25, verbose=1)

    model.fit_generator(train_generator,
              steps_per_epoch=nb_train_samples // int(args['batch_size']),
              epochs=int(args['epochs']),
              #callbacks=my_callbacks,
    	      callbacks=[tensorboard_callback, reduce_lr, my_callbacks],
              validation_data=val_generator,
              validation_steps=nb_val_samples // int(args['batch_size']))

    model.save(save_model_path+'model_new.h5')

if __name__ == '__main__':
    main()
