"""Модуль для визуализации предсказаний нейронной сети детектора."""

import cv2
import numpy as np


def vis(
    img,
    boxes,
    scores,
    cls_ids,
    conf=0.1
):
    """Отрисовка обрамляющих прямоугольников. Цвет определяется в массиве `_COLORS`.
        Используется ограничение на визуализацию по уверенности в прямоугольнике. Порог - `conf`
    """

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(round(box[0]))
        y0 = int(round(box[1]))
        x1 = int(round(box[2]))
        y1 = int(round(box[3]))

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{:.1f}'.format(score * 100)
        txt_color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.3, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)

        txt_bk_color = (255, 255, 255)
        cv2.rectangle(
            img,
            (x0, y0 - 2),
            (x0 + txt_size[0] + 1, y0 - int(1.5*txt_size[1]) + 1),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 - txt_size[1] + 5), font, 0.3, txt_color, thickness=1)

    return img

# Цвета в BGR пространстве
_COLORS = np.array(
    [
        1.0, 0.0, 0.0,   # синий - licence(ГРЗ)
        0.0, 0.0, 1.0,   # красный - car(машина)
        0.0, 0.608, 0.0, # темно-зеленый - cabin(кабина)
        0.5, 0.0, 0.0,   # темно-синий - day(день)
        1.0, 1.0, 1.0    # белый - night(ночь)
    ]
).astype(np.float32).reshape(-1, 3)