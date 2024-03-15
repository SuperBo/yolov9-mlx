import pytest
from mlx import core as mx

from yolov9_mlx.models import yolo


def test_yolov9_inference_1():
    model = yolo.Yolov9(num_classes=80)
    model.train(False)
    im = mx.random.normal([1, 64, 64, 3], dtype=mx.float32)

    result = model(im)

    assert result is not None
    assert len(result) == 4
    assert result[0] is not None
    assert result[1] is not None
