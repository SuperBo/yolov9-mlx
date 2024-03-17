import numpy as np

from yolov9_mlx import detect


def test_single_nms_0():
    detects = np.array([
        [10, 10, 20, 20],
        [8, 8, 18, 18],
        [12, 12, 20, 20]
    ])
    scores = np.array([0.8, 0.4, 0.6])

    boxes, keeps = detect.single_nms(detects, scores, 0.6)

    assert len(keeps) == 2
    assert list(keeps) == [0, 1]
    assert boxes.shape[0] == 2
    assert list(boxes[0]) == [10, 10, 20, 20]


def test_single_nms_1():
    detects = np.array([
        [0, 0, 40, 40],
        [20, 20, 60, 60],
        [65, 65, 80, 80]
    ], dtype=np.float32)
    scores = np.array([0.4, 0.8, 0.6])

    boxes, keeps = detect.single_nms(detects, scores, 0.8)

    assert len(keeps) == 3
    assert list(keeps) == [1, 2, 0]
    assert boxes.shape[0] == 3
    assert list(boxes[0]) == [20, 20, 60, 60]


def test_xywh_to_xyxy():
    box = np.array([
        [10, 10, 4, 6],
        [20, 10, 6, 4],
        [80, 200, 40, 200]
    ], dtype=np.int32)

    box_xy = detect.xywh_to_xyxy(box)

    assert box_xy.shape[-1] == 4
    assert list(box_xy[0]) == [8, 7, 12, 13]
    assert list(box_xy[1]) == [17, 8, 23, 12]
    assert list(box_xy[2]) == [60, 100, 100, 300]
