from typing import Literal, Optional

import mlx.core as mx
from mlx import nn

from yolov9_mlx import utils


### Basic Modules


def autopad(kernel_size: int, padding: Optional[int] = None, dilation: int = 1) -> int:
    """Padding to same shape outputs."""
    if padding:
        return padding

    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1  # actual kernel-size

    return kernel_size // 2 # auto-pad


def make_divisible(x, divisor):
    """Returns nearest lower x divisible by divisor
    """
    if isinstance(divisor, mx.array):
        divisor = int(mx.eval(mx.max(divisor))) # to int
    return (x // divisor) * divisor


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension
    """
    def __init__(self, axis: int = 1) -> None:
        super().__init__()
        self.axis = axis

    def __call__(self, xs):
        return mx.concatenate(xs, axis=self.axis)


class ExtendedConv2d(nn.Module):
    """Extends mlx.nn.Conv2d with groups support."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        groups: int = 1,
        bias: bool = True
    ) -> None:
        """Initializes ExtendedConv2d instance.
        """
        if in_channels % groups != 0:
            raise ValueError(f"{in_channels} is not divisible by {groups}")
        if groups < 1:
            raise ValueError(f"{groups} is less than 1")

        self.convs = [
            nn.Conv2d(
                in_channels // groups,
                out_channels // groups,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )
            for _ in range(groups)
        ]
        self.groups = groups

    def __call__(self, x):
        if self.groups == 1:
            return self.convs[0](x)

        xs = mx.split(x, self.groups, axis=-1)
        xs = [conv_fn(x) for conv_fn, x in zip(self.convs, xs)]
        return mx.concatenate(xs, axis=-1)


class Conv(nn.Module):
    """Conventional convolution block with batchnorm."""

    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        eps: float = 0.001,
        momentum: float = 0.1,
        activation: Literal["silu"] | Literal["identity"] = "silu"
    ) -> None:
        """Inits Convolution block.
        Args:
            in_channels
            out_channels
            kernel_size
            stride
            eps
            momentum
        """
        super().__init__()

        if groups == 1:
            self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=autopad(kernel_size, padding, 1),
                bias=False,
            )
        else:
            self.conv = ExtendedConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=autopad(kernel_size, padding, 1),
                groups=groups,
                bias=False,
            )
        self.batchnorm = nn.BatchNorm(out_channels, eps=eps, momentum=momentum)

        if activation == "identity":
            self.activation = nn.Identity()
        else:
            self.activation = nn.silu

    def __call__(self, x):
        return self.activation(self.batchnorm(self.conv(x)))


class ADown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__
        c_in = in_channels // 2
        c_out = out_channels // 2

        self.avg_pool = nn.AvgPool2d(2, 1, 0)
        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.conv1 = Conv(c_in, c_out, 3, 2, 1, momentum=0.03)
        self.conv2 = Conv(c_in, c_out, 1, 1, 0, momentum=0.03)

    def __call__(self, x):
        x = self.avg_pool(x)
        x1, x2 = x.split(2, axis=-1)

        x1 = self.conv1(x1)
        x2 = self.conv2(self.max_pool(x2))

        return mx.concatenate([x1, x2], axis=-1)


class RepConvN(nn.Module):
    """RepConv is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ) -> None:
        super().__init__()

        self.conv1 = Conv(
            in_channels, out_channels, kernel_size, stride,
            padding=1,
            activation="identity"
        )
        self.conv2 = Conv(
            in_channels, out_channels, 1, stride,
            padding=0,
            activation="identity"
        )

    def __call__(self, x):
        return nn.silu(mx.add(self.conv1(x), self.conv2(x)))


class SPPool2d(nn.Module):
    """SP Pooling Module Block."""
    def __init__(self, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

    def __call__(self, x):
        return self.pool(x)


class RepNBottleneck(nn.Module):
    """Standard Bottleneck."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        kernel_size: tuple[int, int] = (3, 3),
        expansion: float = 0.5
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = RepConvN(in_channels, hidden_channels, kernel_size[0], stride=1)
        self.conv2 = Conv(hidden_channels, out_channels, kernel_size[1], stride=1)
        self.add_input = shortcut and in_channels == out_channels

    def __call__(self, x):
        out = self.conv2(self.conv1(x))
        if self.add_input:
            out = mx.add(out, x)
        return out


class RepNCSP(nn.Module):
    """CSP Bottleneck with 3 convolution layers."""

    def __init__(self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv3 = Conv(hidden_channels + hidden_channels, out_channels, 1, 1)

        bottlenecks = [
            RepNBottleneck(hidden_channels, hidden_channels, shortcut, expansion=1.0)
            for _ in range(n)
        ]
        self.bottlenecks = nn.Sequential(*bottlenecks)

    def __call__(self, x):
        x1 = self.bottlenecks(self.conv1(x))
        x2 = self.conv2(x)

        x3 = mx.concatenate([x1, x2], axis=-1)
        return self.conv3(x3)


### GELAN


class SPPELAN(nn.Module):
    """SPP-ELAN Module Block."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int) -> None:
        super().__init__()

        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.pool1 = SPPool2d(5)
        self.pool2 = SPPool2d(5)
        self.pool3 = SPPool2d(5)
        self.conv2 = Conv(4 * hidden_channels, out_channels, 1, 1)

    def __call__(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.pool2(x2)
        x4 = self.pool3(x3)

        x5 = mx.concatenate([x1, x2, x3, x4], axis=-1)
        return self.conv2(x5)


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN Module Block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden1_channels: int,
        hidden2_channels: int,
        n: int = 1
    ) -> None:
        super().__init__()

        self.conv1 = Conv(in_channels, hidden1_channels, 1, 1, momentum=0.03)
        self.conv2 = nn.Sequential(
            RepNCSP(hidden1_channels // 2, hidden2_channels, n),
            Conv(hidden2_channels, hidden2_channels, 3, 1, momentum=0.03)
        )
        self.conv3 = nn.Sequential(
            RepNCSP(hidden2_channels, hidden2_channels, n),
            Conv(hidden2_channels, hidden2_channels, 3, 1, momentum=0.03)
        )
        self.conv4 = Conv(hidden1_channels + (2 * hidden2_channels), out_channels, 1, 1, momentum=0.03)

    def __call__(self, x):
        x1, x2 = mx.split(self.conv1(x), 2, axis=-1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)

        x5 = mx.concatenate([x1, x2, x3, x4], axis=-1)
        return self.conv4(x5)


### CBNet ###


class CBLinear(nn.Module):
    """CB Linear Block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: list[int],
        kernel_size: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.n_outs = len(out_channels)
        self.conv = nn.Conv2d(
            in_channels,
            sum(out_channels),
            kernel_size=kernel_size,
            stride=stride,
            padding=autopad(kernel_size),
            bias=True
        )

        splits = []
        idx = 0
        for c in out_channels[:-1]:
            splits.append(c + idx)
            idx += c

        self.splits = splits

    def __call__(self, x):
        x = self.conv(x)
        if self.n_outs <= 1:
            return [x]

        return mx.split(x, self.splits, axis=-1)



class CBFuse(nn.Module):
    """CB Fuse Block."""

    def __init__(
        self,
        index: int,
        scale_factors: list[float]
    ) -> None:
        self.index = index
        self.upscales = [nn.Upsample(scale_factor=s, mode="nearest") for s in scale_factors]

    def __call__(self, xs):
        outs = [
            upscale_fn(x[self.index])
            for x, upscale_fn in zip(xs[:-1], self.upscales)
        ]
        outs.append(xs[-1])

        return mx.sum(mx.stack(outs), axis=0)


### YOLO Modules


class DFL(nn.Module):
    """DFL Module Block."""
    def __init__(self, in_channels: int = 17) -> None:
        super().__init__()

        self.in_channels = in_channels
        # self.conv = nn.Conv2d(in_channels, 1, 1, bias=False)
        # self.conv.train(False) # Turn off grad
        # self.conv.load_weights([
        #     ("weight", mx.arange(in_channels, dtype=mx.float32).reshape(1, 1, 1, -1))
        # ])

        # Replace Original Conv2d kernel (1, 1) in original
        # with inner product
        self.weight = mx.arange(in_channels, dtype=mx.float32)

    def __call__(self, x):
        b, a, _ = x.shape # batch, anchors, channels
        out = x.reshape(b, a, 4, self.in_channels)
        out = nn.softmax(out, axis=-1)
        # out = self.conv(out)
        # return out.reshape(b, a, 4)
        return mx.inner(out, self.weight)


class DualDDetect(nn.Module):
    """YOLO Detect head for detection models.
    Used as final layer in dual model which includes
    one main branch and one auxilary branch.
    """

    def __init__(
        self,
        num_classes: int,
        channels: list[int] = [],
        stride: list[int] = [],
        inplace: bool = True,
    ) -> None:
        super().__init__()

        self._shape = None

        self.num_classes = num_classes
        self.num_layers = len(channels) // 2 # number of detection layers
        self.reg_max = 16

        self.num_outputs_per_anchor = num_classes + self.reg_max * 4
        self.inplace = inplace

        if stride:
            assert len(stride) == self.num_layers
            self.stride = mx.array(stride, dtype=mx.float32)
        else:
            self.stride = mx.zeros(self.num_layers, dtype=mx.float32)

        channels_2 = make_divisible(max((channels[0] // 4, self.reg_max * 4, 16)), 4)
        channels_3 = max((channels[0], min((self.num_classes * 2, 128))))
        channels_4 = make_divisible(max((channels[self.num_layers] // 4, self.reg_max * 4, 16)), 4)
        channels_5 = max((channels[self.num_layers], min((self.num_classes * 2, 128))))

        self.convs2 = [
            nn.Sequential(
                Conv(c, channels_2, 3, 1),
                Conv(channels_2, channels_2, 3, 1, groups=4),
                ExtendedConv2d(channels_2, 4 * self.reg_max, 1, groups=4)
            )
            for c in channels[:self.num_layers]
        ]
        self.convs3 = [
            nn.Sequential(
                Conv(c, channels_3, 3),
                Conv(channels_3, channels_3, 3),
                nn.Conv2d(channels_3, self.num_classes, 1)
            )
            for c in channels[:self.num_layers]
        ]
        self.convs4 = [
            nn.Sequential(
                Conv(c, channels_4, 3),
                Conv(channels_4, channels_4, 3, groups=4),
                ExtendedConv2d(channels_4, 4 * self.reg_max, 1, groups=4)
            )
            for c in channels[self.num_layers:]
        ]
        self.convs5 = [
            nn.Sequential(
                Conv(c, channels_5, 3),
                Conv(channels_5, channels_5, 3),
                nn.Conv2d(channels_5, self.num_classes, 1)
            )
            for c in channels[self.num_layers:]
        ]

        self.dfl1 = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)

    def __call__(self, x) -> tuple[mx.array|None, mx.array|None, mx.array, mx.array]:
        d1 = [
            mx.concatenate([conv2(xi), conv3(xi)], axis=-1)
            for conv2, conv3, xi in zip(self.convs2, self.convs3, x[:self.num_layers])
        ]

        d2 = [
            mx.concatenate([conv4(xi), conv5(xi)], axis=-1)
            for conv4, conv5, xi in zip(self.convs4, self.convs5, x[self.num_layers:])
        ]

        if self.training:
            return None, None, d1, d2

        # MLX use NxHxWxC layout versus NxCxHxW layout of pytorch
        # TODO: cache different cache size
        shape = x[0].shape #BHWC
        if self._shape != shape:
            anchors, strides = utils.make_anchors(d1, self.stride, 0.5)
            self._anchors = anchors
            self._strides = strides
            self._shape = shape

        a = mx.concatenate([d.reshape(shape[0], -1, self.num_outputs_per_anchor) for d in d1], axis=1)
        box1, cls1 = mx.split(a, [self.reg_max * 4], axis=-1)

        distance1 = self.dfl1(box1)
        dbox1 = utils.dist2bbox(distance1, self._anchors[None, :], xywh=True, axis=-1)
        dbox1 = dbox1 * self._strides

        a = mx.concatenate([d.reshape(shape[0], -1, self.num_outputs_per_anchor) for d in d2], axis=1)
        box2, cls2 = mx.split(a, [self.reg_max * 4], axis=-1)

        dbox2 = utils.dist2bbox(self.dfl2(box2), self._anchors[None, :], xywh=True, axis=-1)
        dbox2 = dbox2 * self._strides

        y1 = mx.concatenate([dbox1, mx.sigmoid(cls1)], axis=-1)
        y2 = mx.concatenate([dbox2, mx.sigmoid(cls2)], axis=-1)
        return y1, y2, d1, d2


class DDetect(nn.Module):
    """YOLO Detect head for detection models.
    Used as final layer in single model which
    includes only main branch after removing auxilary branch.
    """
    def __init__(
        self,
        num_classes: int,
        channels: list[int] = [],
        stride: list[int] = [],
        inplace: bool = True,
    ) -> None:
        super().__init__()

        self._shape = None

        self.num_classes = num_classes
        self.num_layers = len(channels) # number of detection layers
        self.reg_max = 16

        self.num_outputs_per_anchor = num_classes + self.reg_max * 4
        self.inplace = inplace

        if stride:
            assert len(stride) == self.num_layers
            self.stride = mx.array(stride, dtype=mx.float32)
        else:
            self.stride = mx.zeros(self.num_layers, dtype=mx.float32)

        channels_2 = make_divisible(max((channels[0] // 4, self.reg_max * 4, 16)), 4)
        channels_3 = max((channels[0], min((self.num_classes * 2, 128))))

        self.convs2 = [
            nn.Sequential(
                Conv(c, channels_2, 3, 1),
                Conv(channels_2, channels_2, 3, 1, groups=4),
                ExtendedConv2d(channels_2, 4 * self.reg_max, 1, groups=4)
            )
            for c in channels[:self.num_layers]
        ]
        self.convs3 = [
            nn.Sequential(
                Conv(c, channels_3, 3),
                Conv(channels_3, channels_3, 3),
                nn.Conv2d(channels_3, self.num_classes, 1)
            )
            for c in channels[:self.num_layers]
        ]

        self.dfl = DFL(self.reg_max)

    def __call__(self, x) -> tuple[mx.array | None, mx.array]:
        d = [
            mx.concatenate([conv2(xi), conv3(xi)], axis=-1)
            for conv2, conv3, xi in zip(self.convs2, self.convs3, x[:self.num_layers])
        ]

        if self.training:
            return None, d

        # TODO: cache different cache size
        shape = x[0].shape #BHWC
        if self._shape != shape:
            anchors, strides = utils.make_anchors(d, self.stride, 0.5)
            self._anchors = anchors
            self._strides = strides
            self._shape = shape

        a = mx.concatenate([di.reshape(shape[0], -1, self.num_outputs_per_anchor) for di in d], axis=1)
        box, cls_ = mx.split(a, [self.reg_max * 4], axis=-1)

        distance = self.dfl(box)
        dbox = utils.dist2bbox(distance, self._anchors[None, :], xywh=True, axis=-1)
        dbox = dbox * self._strides

        y = mx.concatenate([dbox, mx.sigmoid(cls_)], axis=-1)
        return y, d
