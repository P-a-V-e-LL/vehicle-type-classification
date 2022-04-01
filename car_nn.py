'''
Программа вырезает кабины автомобилей из видео.
Каждую вырезанную кабину называет в соответствии с номером автомобиля.
Использует нейросеть распознавания номеров, распознавания классов автомобилей.
Используем для формирования первоначального (неразмеченного) датасета.
'''
'''DELETE'''
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

import logging
import random
from itertools import groupby
import tensorflow as tf
import tflite_runtime.interpreter as tflite
from PIL import Image
import argparse

onnx_path_demo = './car_detect/yolov4_2_3_512_512_static.onnx'   	 # путь к нейронной сети распознавания автомобилей (выделение классов)
model_path = './car_detect/anpr_ru_one_linear_20210426.tflite'   	 # путь к нейронной сети распознавания номеров
names_file = './car_detect/names.txt'                             	 # путь к файлу классов
save_path = './car_detect/frames/'                  	 # путь для сохранения результатов работы программы

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-vp",
        "--video_path",
        required=True,
        help="Path to video folder."
    )
    return vars(ap.parse_args())

def prep(session, image_src):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    return img_in


def post_processing(img, conf_thresh, nms_thresh, output):

    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors

    # [batch, num, 1, 4]
    box_array = output[0]
    # [batch, num, num_classes]
    confs = output[1]

    t1 = time.time()

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2] # was 2

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)

    t3 = time.time()

    #print('-----------------------------------')
    #print('       max and argmax : %f' % (t2 - t1))
    #print('                  nms : %f' % (t3 - t2))
    #print('Post processing total : %f' % (t3 - t1))
    #print('-----------------------------------')

    return bboxes_batch


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    import cv2
    num = None # перенести в цикл ниже (или нет)
    conf = -1
    cabine_flag = False
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        #id = random.randint(1, 100000000)
        past_num = ''
        past_conf = -1

        box = boxes[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            try:
                if savename: # сохраняем вырезанную кабину если она есть
                    if cls_id == 2 and cabine_flag and y1-10 > height * 0.35 and y2+10 < height and x2+10 < width: # условия
                        print("Saving new cabine with license: ", num)
                        cut_img_cab = img[y1-10:y2+10, x1-10:x2+10] # рамка в 10 пикселей
                        cv2.imwrite(savename.replace(".jpg", "")+num+'.jpg', cut_img_cab)
                        print("Success!")
                        cabine_flag = False
                    elif cls_id == 0 and cls_conf>=0.99 and y1-20 > height * 0.5 and y2+20 < height and x2+20 < width:
                        # вручную добавляю рамку в 20 pix чтобы сетка распознавала номера, значение условия может меняться
                        if x2-x1 > 200:
                           cut_img = img[y1-20:y2+20, x1:x2]
                        else:
                            cut_img = img[y1-20:y2+20, x1-20:x2+20]

                        num, conf = license_recognize(cut_img, x)

                        if num != past_num or conf > past_conf:
                            #cv2.imwrite(savename.replace(".jpg", "")+'licence_'+num+'.jpg', cut_img) # убрать в итоговой версии вырезанные номера не требуются
                            cabine_flag = True
                            past_num = num
                            past_conf = conf
                            print("New license found: ", num)
            except Exception as err:
                print("FAIL. -- ", err)
    print("----------------------------")
    return img


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


'''Распознавание текста с номеров (Класс)'''
class LicensePlateSignature:
    def __init__(self,model_path=''):
        self.alphabet = []
        if model_path is '':
            dir_path = os.path.dirname(os.path.realpath(__file__))
            model_path =  dir_path.replace('/lib','/neuroweb/anpr/failback') + '/anpr_ru_20210426.tflite'
        self.load_alphabet(model_path)
        self.interpreter = tflite.Interpreter(model_path)
        self.signature_name = 'serving_default'
        self.my_signature = self.interpreter.get_signature_runner(self.signature_name)
        self.data = None

    def get_literal(self, p):
      i = np.argmax(p, 0)
      v = p[i]
      if v > 0.4:
          l = self.alphabet[i]
      else:
          l = '-'
      return l, v

    def load_alphabet(self, model_path):
        alphabet_path = os.path.dirname(model_path) + '/alphabet.txt'
        self.alphabet = [x.rstrip('\n') for x in open(alphabet_path,'r')]

    def preprocess_image(self, image, enable_clahe=False):
        if enable_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
        img = cv2.resize(image, (256, 128))
        img = img.astype(np.float32)
        img -= np.amin(img)
        img /= (np.amax(img) or 1)
        img[img == 0] = 0.0001
        img = img.T
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        self.data = img
        return img

    def run(self):
        results = self.my_signature(image_input=self.data)
        self.interpreter.reset_all_variables()
        res = np.array(results['out'])
        inputs = res[2:, :]
        g = np.argsort(inputs, axis=1)
        c = np.argmax(inputs, 1)
        old = 0
        d = {}
        for k, g in groupby(c):
            d[old] = k
            for r in g:
                old +=1
        u = {k: self.alphabet[v] for k, v in d.items() if v < len(self.alphabet)}
        out_prob = np.zeros((len(u.keys()), res.shape[-1]))

        for i , k in enumerate(u.keys()):
            out_prob[i, :] = inputs[k, :]

        number = ''
        confidence = 1
        for k in list(out_prob):
            lit, c = self.get_literal(k)
            number += lit
            confidence = min(confidence, c)

        if len(number) < 7:
            number = '-------'
        return number, confidence


def license_recognize(path, x):
    #license = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x.preprocess_image(path)#, enable_clahe=True)
    return x.run()


def detect(session, image_src_1, image_src_2):
    targ = np.zeros([2, 3, 512, 512], dtype=np.float32)
    targ[0, :] = prep(session, image_src_1)
    targ[1, :] = prep(session, image_src_2)

    img_in = targ

    print("Shape of the network input: ", img_in.shape)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_in})
    boxes = post_processing(img_in, 0.96, 0, outputs) # (img_in, 0.4, 0, outputs) 0.9785
    namesfile = names_file
    class_names = load_class_names(namesfile)
    plot_boxes_cv2(image_src_1, boxes[0], savename=save_path, class_names=class_names)
    plot_boxes_cv2(image_src_2, boxes[1], savename=save_path, class_names=class_names)
    return outputs


'''MAIN BLOCK'''

for file in os.listdir(args['video_path']):
    args = get_arguments()
    os.makedirs('./car_detect/frames/', exist_ok=True)
    try:
        v_path = os.path.join(args['video_path'], file)
        new_name = os.path.join(args['video_path'], "(ok)"+file)
        print("-"*20)
        print("STARTED", v_path)
        print("-"*20)
        session = onnxruntime.InferenceSession(onnx_path_demo)
        print("The model expects input shape: ", session.get_inputs()[0].shape)
        x = LicensePlateSignature(model_path=model_path)

        video_capture = cv2.VideoCapture(v_path)
        pos_frame = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        i = 1
        j = 0
        t1 = time.time()

        while True:
            i = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            # вместо i можно использовать video_capture.get(cv2.CAP_PROP_POS_FRAMES) -- начинается с 1.0
            flag, frame = video_capture.read()
            frame = cv2.cvtColor(np.float32(frame), cv2.COLOR_BGR2GRAY) # замена цвета кадра
            if flag:
                if (i > -1 and i <= 1000000): # спец. тествовый вариант для детекта машины, убрать при дальнейшей работе
                    if j == 0:
                        f1 = frame
                        j += 1
                    elif j == 1:
                        out = detect(session, f1, frame)
                        j = 0
                    else:
                        print("ERROR!")
                        j = 0
                        break
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
        os.rename(v_path, new_name)
    except:
        os.rename(v_path, new_name)
        continue
