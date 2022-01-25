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

def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)


img_width, img_height = 160, 160 				# целевой размер изображения для обучения
epochs = 250 							# количество эпох при обучении
batch_size = 128 						# размер батча
nb_train_samples = 47734 					# количество тренировочных изображений
nb_val_samples = 3158 					# количество валидационных изображений

train_dir = './car196/train' # './dataset3.11_2/train'    	# путь к обучающей выборке
val_dir = './car196/val' # './dataset3.11_2/val'         	# путь к валидационной выборке
model_path = '/home/inventos/neuroweb/datasets/to_serv/facenet_keras.h5'    # путь к исходной модели
save_model_path = './'             				# путь к месту сохранения обученной модели


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


model = load_model(model_path)
#model = clone_model(model)

#sgd = SGD(lr=0.001, momentum=0.001, nesterov=True)

opt = RMSprop(learning_rate=0.001, centered=True)


model.compile(loss=tfa.losses.TripletSemiHardLoss(),
              optimizer=opt,  #Adam(learning_rate=1e-5),
	      metrics=[triplet_accuracy]) # 'accuracy' or triplet_accuracy

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(train_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode='sparse')

val_generator = datagen.flow_from_directory(val_dir,
                                              target_size=(img_width, img_height),
                                              batch_size=batch_size,
                                              class_mode='sparse')

my_callbacks = [
                tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path + 'model_scratch_new.h5',
                                                   monitor='val_loss',
                                                   mode='min',
                                                   save_best_only=True),
]

import datetime
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25, verbose=1)

#lr_callback = 

model.fit_generator(train_generator,
          steps_per_epoch=nb_train_samples // batch_size,
          epochs=epochs,
          #callbacks=my_callbacks,
	  callbacks=[tensorboard_callback, reduce_lr],
          validation_data=val_generator,
          validation_steps=nb_val_samples // batch_size)

model.save(save_model_path+'model_new.h5')
