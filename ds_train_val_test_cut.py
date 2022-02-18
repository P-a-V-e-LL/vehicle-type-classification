import numpy as np
import os
import shutil
import argparse

'''Распределить все классы по папкам train, val, test.'''

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        required=True,
        help="Path to data."
    )
    ap.add_argument(
        "--train_dir",
        required=True,
        help="Path to train location."
    )
    ap.add_argument(
        "--val_dir",
        required=True,
        help="Path to val location."
    )
    ap.add_argument(
        "--cut",
        required=True,
        default=0.2,
        help="val data %"
    )
    return vars(ap.parse_args())

test_dir = None

def create_classes(dir, train_dir, val_dir):
    '''Создание папок train, val, (test) со всеми классами исходного датасета.'''
    for item in os.listdir(dir):
        os.makedirs(os.path.join(train_dir, item))
        os.makedirs(os.path.join(val_dir, item))
        #os.makedirs(os.path.join(test_dir, item))


def copy_set(source_dir, train_dir, val_dir, test_dir, cut = 0.2):
    '''Копирование изображений в папки train, val, test из исходной'''
    print("Start")
    for car_class in os.listdir(source_dir):
        print(car_class)
        item = os.path.join(source_dir, car_class)
        x = os.listdir(item)
        amount = len(x)# - cut
        d = int(amount * cut)
        for i in range(0, d):
            shutil.copy2(os.path.join(item, str(x[i])), os.path.join(val_dir, car_class))
        for i in range(d, amount):
            shutil.copy2(os.path.join(item, str(x[i])), os.path.join(train_dir, car_class))
        #for i in range(-1, -cut-1, -1):
        #    shutil.copy2(os.path.join(item, str(x[i])), os.path.join(test_dir, car_class))

def main():

    create_classes(args['data_dir'], args['train_dir'], args['val_dir'])
    copy_set(args['data_dir'], args['train_dir'], args['val_dir'], test_dir)
    print('END')

if __name__ == '__main__':
    main()
