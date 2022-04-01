'''
Программа вырезает кабины автомобилей из видео.
Каждую вырезанную кабину называет в соответствии с номером автомобиля.
Использует нейросеть распознавания номеров, распознавания классов автомобилей.
Используем для формирования первоначального (неразмеченного) датасета.
'''

import sys
import os
import argparse
import numpy as np
import cv2
import onnxruntime
import sys
import time
import datetime
import math
import random
import csv

import logging
import random
from itertools import groupby
import tensorflow as tf
import tflite_runtime.interpreter as tflite
from PIL import Image
import argparse

from yolox.onnx_model import InferenceYoloxOnnx
from yolox.visualisation import vis

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-vp",
        "--video_path",
        required=True,
        help="Путь до папки с видео."
    )
    ap.add_argument( # save_path
        "--save_path",
        default="./car_detect/frames/",
        help="Папка для сохранения изображений."
    )
    ap.add_argument( # names_file
        "--class_file_path",
        default='./car_detect/names.txt' ,
        help="Путь до txt файла с классами."
    )
    ap.add_argument( # model_path
        "--model_path",
        type=str,
        required=True,
        help="Путь до модели в формате onnx.",
    )
    ap.add_argument(
        "--csv_data",
        type=str,
        required=True,
        help="Путь до csv файла для записи/дозаписи данных в таблицу.",
    )
    ap.add_argument(
        "--box_length",
        type=int,
        default=0,
        help="Размер рамки для отступа вокруг рамок классов.",
    )
    return vars(ap.parse_args())

def get_coordinates(cl, boxes):
    final = {}
    for i in range(len(cl)):
        if int(cl[i]) == 1:
            final[1] = boxes[i]
        else:
            final[1] = 0
        if int(cl[i]) == 2:
            final[2] = boxes[i]
        else:
            final[2] = 0
        if int(cl[i]) == 3:
            final[3] = boxes[i]
        else:
            final[3] = 0
    return final


def write_table(writer, image_path, car, cabin, license, day=None, night=None, box=0):
    '''Записывает полученные данные о классах прямоугольников в таблицу.
    Сделать возможность добавить рамку (ко всем или конкретно?);
    Сделать возможность записат новый/дозаписать существующий файл.
    Открытие файла на запись и дозапись. В main.'''
    pass

'''MAIN BLOCK'''

def main():
    args = get_arguments()

    if os.path.exists(args['csv_data']):
        f = open(args['csv_data'], "a", encoding="UTF8")
        writer = csv.writer(f)
    else:
        f = open(args['csv_data'], "w", encoding="UTF8")
        writer = csv.writer(f)
        writer.writerow(["image_path", "car", "cabin", "license"])
        #writer.writerow([image_path, car, cabin, license])
        #f.close()

    os.makedirs('./car_detect/frames/', exist_ok=True)
    yolox = InferenceYoloxOnnx(args['model_path'])

    for file in os.listdir(args['video_path']):
#        try:
        v_path = os.path.join(args['video_path'], file)
        new_name = os.path.join(args['video_path'], "(ok)"+file)
        print("-"*20)
        print("STARTED", v_path)
        print("-"*20)
        video_capture = cv2.VideoCapture(v_path)
        pos_frame = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        i = 1
        j = 0
        t1 = time.time()

        while True:
            i = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            # вместо i можно использовать video_capture.get(cv2.CAP_PROP_POS_FRAMES) -- начинается с 1.0
            flag, frame = video_capture.read()
            #frame = cv2.cvtColor(np.float32(frame), cv2.COLOR_BGR2GRAY) # замена цвета кадра
            if flag:
                if (i > -1 and i <= 1000000):# спец. тествовый вариант для детекта машины, убрать при дальнейшей
                    yolox.preprocess_image(frame)
                    dets = yolox.run()
                    if dets is not None:
                        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                        print(final_boxes)
                        #frame = vis(frame, final_boxes, final_scores, final_cls_inds, conf=yolox.nms_conf_thr)
                        name = args['save_path']+"car_"+str(random.randint(1,10000))+".jpg"
                        cv2.imwrite(name, frame)
                        final = get_coordinates(final_cls_inds, final_boxes)
                        writer.writerow([name, final[1], final[2], final[3]])
                print("Frame {0} / {1}".format(int(video_capture.get(cv2.CAP_PROP_POS_FRAMES)), int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))))
            else:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
                print("frame is not ready")
                cv2.waitKey(1000)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("TIME---------", time.time()-t1)
                break
            if video_capture.get(cv2.CAP_PROP_POS_FRAMES) == video_capture.get(cv2.CAP_PROP_FRAME_COUNT):
                print("TIME---------", str(datetime.timedelta(seconds=int(time.time()-t1))))
                break

        video_capture.release()
        cv2.destroyAllWindows()
        f.close()
        os.rename(v_path, new_name)
        #except:
    #        os.rename(v_path, new_name)
#            continue



if __name__ == '__main__':
    main()
