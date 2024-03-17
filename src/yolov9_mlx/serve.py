import bentoml
from PIL.Image import Image

from yolov9_mlx.models import yolo


@bentoml.service(
    resources={"cpu": "1",  "memory" : "3Gi"},
    traffic={"timeout": 10},
)
class YoloService:
    def __init__(self) -> None:
        model = yolo.Yolov9C()
        model.load_weights("yolov9-c.safetensors")
        model.eval()
        self.model = model
        print("Model Yolov9C loaded")

        self.max_dim = 1920

    @bentoml.api
    def detect(self, img: Image) -> list[int]:
        """Detect object in image."""
        x = self._resize_image(img)
        x.img
        y1, y2, _, _ = self.model()

    def _resize_image(self, img: Image) -> Image:
        """Resizes too large image."""
        im_width, im_height = img.size

        ratio = self.max_dim / max(im_height, im_width)  # ratio
        if ratio < 1.0:  # image too large
            img = img.resize((int(im_width * ratio), int(im_height * ratio)))

        return img
