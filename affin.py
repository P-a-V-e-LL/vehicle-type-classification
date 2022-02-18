import os
import numpy as np
import random
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import argparse

'''Добавление к изоюражению различных эффектов и сохранение их.
ДЛЯ ЧАСТНОГО СЛУЧАЯ!'''

'''
Rotate - ok
Noise - ok
Contrast - ok
Some weather effects (rain) - ok
Crop - ok
Mirroring - ok
MotionBlur - ok
'''


dir = "/home/pavel/Desktop/University/Diplom/cars196/train" 			 #  root_dir набора данных

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root_dir",
        required=True,
        help="Path to data."
    )
    return vars(ap.parse_args())

def mult_save(images):
    for image in images:
        new = save_dir + "/" + "car_"+ str(random.randint(1, 100000)) +".jpg"
        print("Saving", new)
        imageio.imwrite(new, image)


aff = iaa.Affine(rotate=(-20, 20)) # поворот                            #1
mblur = iaa.MotionBlur(k=5) # размытие при движении                       #2
gcon = iaa.GammaContrast(gamma=[1.0, 2.0])                             #3
flip = iaa.Fliplr(0.5) # указывается процент изображений для поворота    #4
rain = iaa.Rain(drop_size=(0.10, 0.15))                                  #5
snow = iaa.Snowflakes(flake_size=(0.7, 0.75), speed=(0.001, 0.03))       #5
fog = iaa.Fog(0)                                                        #5
crop = iaa.CropAndPad(percent=(-0.3, 0))                                 #6
noise = iaa.AdditiveGaussianNoise(scale=0.15*255, per_channel=True)        #7

# радномно оп одному или с наложением нескольких
#seq = iaa.Sequential([
    #iaa.Affine(rotate=(-25, 25)), # поворот                            #1
    #iaa.MotionBlur(k=5), # размытие при движении                       #2
    #iaa.GammaContrast(gamma=[1.0, 2.0]),                               #3
    #iaa.Fliplr(0.5), # указывается процент изображений для поворота    #4
    #iaa.Rain(drop_size=(0.10, 0.15)),                                  #5
    #iaa.Snowflakes(flake_size=(0.7, 0.75), speed=(0.001, 0.03)),       #5
    #iaa.Fog(0),                                                        #5
    #iaa.CropAndPad(percent=(-0.3, 0)),                                 #6
    #iaa.AdditiveGaussianNoise(scale=0.15*255, per_channel=True)        #7
#])
#], random_order=True)

ia.seed(42)

def augment(img):
    sometimes = lambda s: iaa.Sometimes(0.4, s)

    affine = iaa.Affine(
        scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        rotate=(-10, 10),
        shear=(-3, 3),
    )
    seq = iaa.Sequential(
        [
            sometimes(iaa.CropAndPad(percent=(-0.1, 0))),
            affine,
            iaa.SomeOf((1, 5),
            [
                sometimes(
                    iaa.OneOf([
                        iaa.OneOf([
                            iaa.GaussianBlur((1, 1.2)),
                            iaa.AverageBlur(k=(1, 3)),
                            iaa.MedianBlur(k=(1, 3)),
                            iaa.MotionBlur(k=(3, 5))
                        ]),
                        iaa.GammaContrast(gamma=[1.0, 2.0]),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.005*255), per_channel=0.001)
                    ])
                ),
                sometimes(iaa.Fliplr(0.5)),
                sometimes(iaa.OneOf([
                    iaa.Rain(drop_size=(0.10, 0.15)),
                    iaa.Snowflakes(flake_size=(0.7, 0.75), speed=(0.001, 0.03)),
                    iaa.Fog(0)
                ]))
            ], random_order=True)
        ], random_order=True)
    return seq(image=img)


#for image in os.listdir(test):
#    for i in range(6):
#        car_image = imageio.imread(os.path.join(test, image))
#        new = test + "/" + "car_"+ str(random.randint(1, 100000)) +".jpg"
#        print("Saving", new)
#        car_image = augment(car_image)
#        imageio.imwrite(new, car_image)

for folder in os.listdir(dir):
    print(folder)
    car_class = os.path.join(dir, folder)
    save_dir = car_class
    base = os.listdir(car_class)
    while (len(os.listdir(car_class)) < 210):  # здесь указывается общее число изображений в каждом классе, по умолчанию 210
        for image in base:
            car_image = imageio.imread(os.path.join(car_class, image))
            new = save_dir + "/" + "car_"+ str(random.randint(1, 100000)) +".jpg"
            print("Saving", new)
            car_image = augment(car_image)
            imageio.imwrite(new, car_image)
