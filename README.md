# Yolov9 MLX

[Yolov9](https://github.com/WongKinYiu/yolov9) code written in [MLX](https://ml-explore.github.io/mlx/build/html/index.html).

![Yolov9 Perfomance](https://github.com/WongKinYiu/yolov9/raw/main/figure/performance.png)

## Installation

You can install using pip

```sh
pip insall .
```

## Load model

Pretrained weights from Yolov9:

- [yolov9-c-converted.safetensors](https://github.com/SuperBo/yolov9-mlx/releases/download/v0.1.0/yolov9-c-converted.safetensors)
- [yolov9-e-converted.safetensors](https://github.com/SuperBo/yolov9-mlx/releases/download/v0.1.0/yolov9-e-converted.safetensors)
- [yolov9-c.safetensors](https://github.com/SuperBo/yolov9-mlx/releases/download/v0.1.0/yolov9-c.safetensors)
- [yolov9-e.safetensors](https://github.com/SuperBo/yolov9-mlx/releases/download/v0.1.0/yolov9-e.safetensors)

Model can be load as following snippet.

```python
from yolov9_mlx.models import yolo

model = yolo.Yolov9CConverted()
model.load_weights("yolov9-c-converted.safetensors")

y, d = model(im)
```
For more details on how to run detect, please refer to [serve.py](src/yolov9_mlx/serve.py)

## Inference service

A model endpoint example is also available in [serve.py](src/yolov9_mlx/serve.py)].

Start service

```sh
pdm sync -G serving
pdm run serve
```

Send request to service
```sh
curl -XPOST -F 'image=@img.jpg' http://localhost:3000/detect
```
