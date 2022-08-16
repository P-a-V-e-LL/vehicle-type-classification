import os
import cv2
import random
import argparse
import numpy as np
import shutil
import json
from yolox.onnx_model import InferenceYoloxOnnx

def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help="Путь до модели в формате onnx.",
    )
    ap.add_argument(
        "--data_path",
        help="Путь до обрабатываемой папки"
    )
    return vars(ap.parse_args())

def dir_to_predict(img_path, cnn):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = np.asarray(img)
    cnn.preprocess_image(image)
    predict = cnn.run()
    return predict

def to_fullhd(img):
    #img = cv2.imread('3.jpg')
    old_image_height, old_image_width, channels = img.shape

    # create new image of desired size for padding
    if old_image_width > 1920:
        new_image_width = old_image_width
    else:
        new_image_width = 1920

    if old_image_height > 1080:
        new_image_height = old_image_height
    else:
        new_image_height = 1080
    result = np.zeros((new_image_height,new_image_width, channels), dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height,x_center:x_center+old_image_width] = img
    return result

def main():
    args = get_arguments()

    decoder = {0:'licence', 1: 'car', 2: 'cabin', 3: 'day', 4: 'night'}
    annotator_data = {}

    new_data_path = os.path.join(args['data_path'], "anomaly")
    #new_data = os.path.join(new_data_path, "data")
    #new_images = os.path.join(new_data_path, "images")
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path)
        #os.makedirs(new_data)
        #os.makedirs(new_images)

    yolox = InferenceYoloxOnnx(args['model'])
    for cl in os.listdir(args['data_path']):
        print(cl)
        if cl != "anomaly":
            class_path = os.path.join(args['data_path'], cl)
            anomaly_path = os.path.join(new_data_path, cl)
            if not os.path.exists(anomaly_path):
                os.makedirs(anomaly_path)
                os.makedirs(os.path.join(anomaly_path, "data"))
                os.makedirs(os.path.join(anomaly_path, "images"))
            for car in os.listdir(class_path):
                car_path = os.path.join(class_path, car)
                img = cv2.imread(car_path)
                img = to_fullhd(img)
                yolox.preprocess_image(img)
                dets = yolox.run()
                if dets is not None:
                    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                    if 4. in final_cls_inds:
                        print("night")
                        continue
                    #print(final_boxes)
                    #print(final_scores)
                    #print(final_cls_inds)
                    #print("-"*30)
                    for i in range(len(final_cls_inds)):
                        annotator_data[str(int(final_cls_inds[i]))] = {
                            decoder[int(final_cls_inds[i])] : {
                                "shape": "rect",
                                "dot1": [final_boxes[i][0], final_boxes[i][1]],
                                "dot2": [final_boxes[i][2], final_boxes[i][3]],
                            }
                        }
                    cv2.imwrite(os.path.join(os.path.join(anomaly_path, "images"), car), img)
                    with open(os.path.join(os.path.join(anomaly_path, "data"), car.replace(".jpg", ".json")), "w+") as json_file:
                        json.dump(annotator_data, json_file)


if __name__ == '__main__':
    main()
