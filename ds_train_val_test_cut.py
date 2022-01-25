import numpy as np
import os
import shutil

'''Распределить все классы по папкам train, val, test.'''

dir = '/home/pavel/Desktop/University/Diplom/cars196/images (copy)'
#curr_dir = '/home/pavel/Desktop/University/Diplom/cars196/images (copy)'
train_dir = '/home/pavel/Desktop/University/Diplom/cars196/train'
val_dir = '/home/pavel/Desktop/University/Diplom/cars196/val'
#test_dir = '/home/pavel/Desktop/University/Diplom/ds1_dirs/test'
test_dir = None

def create_classes():
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

create_classes()
copy_set(dir, train_dir, val_dir, test_dir)
print('END')
