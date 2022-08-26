import os
import cv2
import argparse
import random
import shutil
import json
import numpy as np

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_path",
        required=True,
        help="Путь до обрабатываемой папки"
    )
    ap.add_argument(
        "--frame_plus",
        type=int,
        default=10,
        help="Дополнительная рамка к прямоугольнику"
    )
    ap.add_argument(
        "--save_path",
        required=True,
        help="Путь до папки сохранения результатов"
    )
    return vars(ap.parse_args())



def main():
    args = get_arguments()
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    for cl in os.listdir(args['data_path']):
        print(cl)

        if not os.path.exists(os.path.join(args['save_path'], cl)):
            os.makedirs(os.path.join(args['save_path'], cl))

        class_path = os.path.join(args['data_path'], cl)
        json_class_path = os.path.join(class_path, 'data')
        images_class_path = os.path.join(class_path, 'images')
        for data in os.listdir(json_class_path):
            data_path = os.path.join(json_class_path, data)
            data_name = data.replace('.json', '')
            with open(data_path) as json_file:
                data = json.load(json_file)

            for index in data.keys():
                if 'cabin' in data[index].keys():
                    x1, y1, x2, y2 = int(data[index]['cabin']['dot1'][0]) - args['frame_plus'], int(data[index]['cabin']['dot1'][1]) - args['frame_plus'], \
                                     int(data[index]['cabin']['dot2'][0]) + args['frame_plus'], int(data[index]['cabin']['dot2'][1]) + args['frame_plus']
                    img = cv2.imread(os.path.join(images_class_path, data_name+'.jpg'))#, cv2.IMREAD_GRAYSCALE)
                    crop_img = img[y1:y2, x1:x2]
                    image = np.asarray(crop_img)
                    cv2.imwrite(os.path.join(os.path.join(args['save_path'], cl), data_name+'.jpg'), image)
                    #cv2.imwrite(os.path.join(os.path.join(args['save_path'], cl), cl.replace('platesmania ', '')+"_"+str(random.randint(1, 1000000)) + '.jpg'), image)

if __name__ == '__main__':
    main()
