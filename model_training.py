'''Обучение нейронной сети.'''

import argparse
import os
import datetime
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.models import load_model
from tensorflow_addons.losses.metric_learning import pairwise_distance
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from dataset.balansed_image_dataset import balanced_image_dataset_from_directory


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
        "--epochs",
        type=int,
        required=True,
        help="Training epochs amount."
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size."
    )
    ap.add_argument(
        "--model_name",
        type=str,
        default="model_new",
        help="Название эксперимента."
    )
    return vars(ap.parse_args())


def check(path):
    i = 0
    for folder in os.listdir(path):
        n = os.path.join(path, folder)
        i += len(os.listdir(n))
    return i


def triplet_accuracy(y_true, y_pred):
    batch_size =tf.cast(tf.size(y_true), dtype=tf.dtypes.float32)
    # Build pairwise distance matrix
    pdist_matrix = pairwise_distance(y_pred, squared=False) # was pairwise_distance
    # Build pairwise binary adjacency matrix.
    adjacency = tf.cast(tf.math.equal(y_true, tf.transpose(y_true)), dtype=tf.dtypes.float32)
    # Invert so we can select negatives only.
    adjacency_not = 1 - adjacency
    # Convert to decision with thresholding at 0.5
    predicted = tf.cast(tf.math.less_equal(pdist_matrix, 0.5), dtype=tf.dtypes.float32)
    # Calculate true positives and true negatives
    tp = tf.reduce_sum(
        tf.cast(
            tf.math.multiply(predicted, adjacency),
            dtype=tf.dtypes.float32
        )
    )
    tn = tf.reduce_sum(
        tf.cast(
            tf.math.multiply(1-predicted, adjacency_not),
            dtype=tf.dtypes.float32
        )
    )
    # Calculate percentage
    return (tp + tn) / (batch_size * batch_size)


def main():
    args = get_arguments()
    seed = 3
    img_width, img_height = 160, 160 # целевой размер изображения для обучения
    save_model_path = './models/' # путь к месту сохранения обученной модели
    #  Train batch (128) = num_classes_per_batch_train * num_images_per_class_train (должно быть кратно 2)
    num_classes_per_batch_train = 8
    num_images_per_class_train = 16
    #  Val batch (64) = num_classes_per_batch_val * num_images_per_class_val (должно быть кратно 2)
    num_classes_per_batch_val = 8
    num_images_per_class_val = 8
    nb_train_samples = check(args['train_data'])
    nb_val_samples = check(args['val_data'])
    model = load_model(args['model_path'], compile=False)
    exp_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = "_".join(
        [args['model_name'], exp_time]
    )
    log_dir = os.path.join("./logs/", exp_time)
    # model = clone_model(model)
    # opt = SGD(lr=0.001, momentum=0.001, nesterov=True)
    # opt =  Adam(learning_rate=1e-5)
    opt = RMSprop(learning_rate=0.001, centered=True)

    model.compile(
        loss=tfa.losses.TripletSemiHardLoss(),
        optimizer=opt, 
        metrics=[triplet_accuracy]
    )

    train_generator = balanced_image_dataset_from_directory(
        args['train_data'],
        num_classes_per_batch=num_classes_per_batch_train,
        num_images_per_class=num_images_per_class_train,
        image_size=(img_width, img_height),
        seed=seed,
        augment=False,
        safe_triplet=True
    )
    val_generator = balanced_image_dataset_from_directory(
        args['val_data'],
        num_classes_per_batch=num_classes_per_batch_val,
        num_images_per_class=num_images_per_class_val,
        image_size=(img_width, img_height),
        seed=seed,
        augment=False,
        safe_triplet=True
    )

    # datagen = ImageDataGenerator(rescale=1. / 255)
    # train_generator = datagen.flow_from_directory(args['train_data'],
    #                                               target_size=(img_width, img_height),
    #                                               batch_size=args['batch_size'],
    #                                               class_mode='sparse')
    # val_generator = datagen.flow_from_directory(args['val_data'],
    #                                               target_size=(img_width, img_height),
    #                                               batch_size=args['batch_size'],
    #                                               class_mode='sparse')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_model_path, f'{exp_name}_best.h5'),
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25, verbose=1)
    model.fit(
        train_generator,
        epochs=int(args['epochs']),
    	callbacks=[tensorboard_callback, reduce_lr, model_checkpoint],
        validation_data=val_generator,
    )
    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=nb_train_samples // args['batch_size'],
    #     epochs=int(args['epochs']),
    #     callbacks=[tensorboard_callback, reduce_lr, model_checkpoint],
    #     validation_data=val_generator,
    #     validation_steps=nb_val_samples // args['batch_size']
    # )
    model.save(os.path.join(save_model_path, f'{exp_name}.h5'))


if __name__ == '__main__':
    main()
