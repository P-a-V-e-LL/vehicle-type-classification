#!/usr/bin/env python3

import argparse
import cv2
import os

from yolox.onnx_model import InferenceYoloxOnnx
from yolox.visualisation import vis


def make_parser():
    parser = argparse.ArgumentParser("Пример запуска YOLOX в ONNX формате.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Путь до модели в формате onnx.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Путь до тестируемого изображения.",
    )
    parser.add_argument(
        "--save_image",
        default=True,
        action="store_true",
        help="Сохранять ли изображение с найденными прямоугольниками."
    )
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    yolox = InferenceYoloxOnnx(args.model)
    img = cv2.imread(args.image_path)
    yolox.preprocess_image(img)
    dets = yolox.run()
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        print(final_boxes)
        print(final_scores)
        print(final_cls_inds)
        img = vis(
            img,
            final_boxes,
            final_scores,
            final_cls_inds,
            conf=yolox.nms_conf_thr
        )
    if args.save_image:
        output_path = os.path.join(
            os.path.dirname(args.image_path),
            os.path.splitext(os.path.basename(args.image_path))[0] + '_visualized.jpg'
        )
        cv2.imwrite(output_path, img)
