import os
import numpy as np
import random
import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import argparse
import uuid

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

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root_dir",
        required=True,
        help="Path to data."
    )
    ap.add_argument(
        "--img_count",
        type=int,
        default=300,
        help="Image count to each class in root_dir."
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
            iaa.SomeOf((1, 5),
            [
                sometimes(
                    iaa.OneOf([
                        sometimes(iaa.OneOf([
                            iaa.Rain(nb_iterations=(1, 2), drop_size=(0.10, 0.18)),
                            iaa.Snowflakes(flake_size=(0.6, 0.7), speed=(0.001, 0.02)),
                            iaa.Fog(0)
                        ])),
                        iaa.OneOf([
                            iaa.GaussianBlur((1, 1.2)),
                            iaa.AverageBlur(k=(1, 2)),
                            iaa.MedianBlur(k=(1, 3)),
                            iaa.MotionBlur(k=(3, 5))
                        ]),
                        iaa.GammaContrast(gamma=[1.0, 1.5]),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.005*255), per_channel=0.001)
                    ])
                ),
                sometimes(iaa.OneOf([
                    affine,
                    sometimes(iaa.Fliplr(0.5)),
                    sometimes(iaa.CropAndPad(percent=(-0.1, 0)))
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

def main():
    args = get_arguments()
    for folder in os.listdir(args['root_dir']):
        print(folder)
        car_class = os.path.join(args['root_dir'], folder)
        #car_class = os.path.join(car_class, 'back')
        save_dir = car_class
        base = os.listdir(car_class)
        if len(os.listdir(car_class)) == 0:
            continue
        while (len(os.listdir(car_class)) < args['img_count']):  # здесь указывается общее число изображений в каждом классе, по умолчанию 210
            for image in base:
                car_image = imageio.imread(os.path.join(car_class, image))
                new = save_dir + "/" + str(uuid.uuid4().hex) + "_affin_" + str(random.randint(1, 10000000)).zfill(8) +".jpg"
                print("Saving", new)
                car_image = augment(car_image)
                imageio.imwrite(new, car_image)

if __name__ == '__main__':
    main()
