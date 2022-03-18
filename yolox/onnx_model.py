"""Модуль для выполения нейронной сети дететора объектов с помощью onnxruntime"""

import cv2
import numpy as np
import onnxruntime


def nms(
    boxes,
    scores,
    nms_thr
):
    """NMS для одного класса.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms_class_agnostic(
    boxes,
    scores,
    nms_thr,
    score_thr
):
    """Многоклассовая NMS. Версия, не учитывающая отдельный класс.
    """
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets


class InferenceYoloxOnnx:


    def __init__(
        self,
        model: str,
        network_input_shape: tuple=(640, 640),
        nms_iou_thr: float=0.75,
        nms_conf_thr: float=0.7,
        with_p6: bool=False,
        cam_id: int=0
    ) -> None:
        self.model = model
        self.network_input_shape = network_input_shape
        self.nms_iou_thr = nms_iou_thr
        self.nms_conf_thr = nms_conf_thr
        self.with_p6 = with_p6
        self.cam_id = cam_id
        self._session = onnxruntime.InferenceSession(self.model)
        self._r_x = 1.0
        self._r_y = 1.0
        self._ort_inputs = {}

    
    def preprocess_image(
        self,
        image: np.ndarray,
        swap: tuple=(2, 0, 1)
    ) -> None:
        """Предобработка изображения. Сохранение оригинальных пропрорций и приведение
            к входному размеру нейронной сети детектора
        """
        self._r_y = self.network_input_shape[0] / image.shape[0]
        self._r_x = self.network_input_shape[1] / image.shape[1]
        resized_image = cv2.resize(
            image,
            self.network_input_shape,
            interpolation=cv2.INTER_AREA
        ).astype(np.uint8)
        resized_image = resized_image.transpose(swap)
        resized_image = np.ascontiguousarray(resized_image, dtype=np.float32)
        self._ort_inputs = {self._session.get_inputs()[0].name: resized_image[None, :, :, :]}


    def _posprocess(
        self,
        outputs
    ):
        """Постобработка результатов нейронной сети. Приведение координат cxcywh к xyxy.
            А также приведение к оригинальным координатам изображения.
        """
        grids = []
        expanded_strides = []
        if not self.with_p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]
        hsizes = [self.network_input_shape[0] // stride for stride in strides]
        wsizes = [self.network_input_shape[1] // stride for stride in strides]
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        boxes = outputs[0][:, :4]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        scores = outputs[0][:, 4:5] * outputs[0][:, 5:]
        dets = multiclass_nms_class_agnostic(
            boxes_xyxy,
            scores,
            self.nms_iou_thr,
            self.nms_conf_thr
        )
        dets[:, 0] /= self._r_x
        dets[:, 1] /= self._r_y
        dets[:, 2] /= self._r_x
        dets[:, 3] /= self._r_y
        return dets


    def run(
        self
    ):
        """Вычисление результата нейронной сети и постобработка результатов.
        """
        detections = None
        if self._ort_inputs is not None:
            output = self._session.run(None, self._ort_inputs)
            detections = self._posprocess(output[0])
        return detections
