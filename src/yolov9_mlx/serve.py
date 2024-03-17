import pathlib

import bentoml
from PIL.Image import Image
from mlx import core as mx
import numpy as np

from yolov9_mlx.models import yolo
from yolov9_mlx import detect
from yolov9_mlx.data import coco


@bentoml.service(
    resources={"cpu": "1",  "memory" : "3Gi"},
    traffic={"timeout": 10},
)
class YoloService:
    def __init__(self) -> None:
        model = yolo.Yolov9CConverted()
        model.load_weights("yolov9-c-converted.safetensors")
        model.eval()

        self.model = mx.compile(model)
        print("Model Yolov9-C-Converted loaded")

        self.max_dim = 640
        self.names = coco.NAMES

    @bentoml.api
    def detect(self, image: Image) -> list[tuple[list[int], str, float]]:
        """Detect object in image."""
        im0 = self._resize_image(image)
        size0 = im0.size
        imx = mx.array((np.array(im0)[:, :, :3] / 255.0)[None, :], dtype=mx.float16)

        y, _ = self.model(imx)
        boxes, classes = detect.batch_non_max_suppression(y)

        boxes = boxes[0]
        classes = classes[0]
        if boxes is None:
            return []

        boxes = detect.scale_boxes(boxes, size0[0], size0[1], image.width, image.height)

        result = []
        for b, cls_ in zip(boxes, classes):
            result.append((
                list(b),
                self.names[cls_.argmax()],
                float(cls_.max())
            ))
        print(result)
        return result

    def _resize_image(self, img: Image) -> Image:
        """Resizes too large image."""
        width, height = img.size
        max_dim = max(width, height)

        if max_dim > self.max_dim:
            r = self.max_dim / max_dim
            width = int(width * r)
            height = int(height * r)

        new_width = detect.make_divisible_by_32(width)
        new_height = detect.make_divisible_by_32(height)

        if new_width == width and new_height == height:
            return img

        return img.resize((new_width, new_height))
