"""
Module help converting from original pytorch models to MLX models.
"""

from typing import Any
import argparse
import sys
import pathlib

import torch
import torch.nn
from mlx import core as mx, nn

from yolov9_mlx.models import yolo, modules


def convert_module_conv2d(conv, mod: nn.Conv2d):
    if conv.groups != 1:
        raise RuntimeError("Wrong num groups in conv module.")

    weights: list[tuple[str, mx.array]] = [
        ("weight", mx.array(conv.weight.detach().moveaxis(1, -1).numpy()))
    ]

    bias = conv.bias
    if bias is not None:
        weights.append(("bias", mx.array(conv.bias.detach().numpy())))

    mod.load_weights(weights)


def convert_module_extendedconv2d(conv, mod: modules.ExtendedConv2d):
    if conv.groups != mod.groups:
        raise RuntimeError("Mismatch number of groups in conv modules.")

    weight = mx.array(conv.weight.moveaxis(1, -1).detach().numpy())
    groups = conv.groups

    bias_weights: list[tuple[str, mx.array]] = []
    bias = conv.bias
    if bias is not None:
        bias_weights.append(("bias", mx.array(conv.bias.detach().numpy())))

    sub_weights = weight.split(groups, axis=0)

    sub_bias = [None] * groups
    if conv.bias is not None:
        sub_bias = mx.array(conv.bias.detach().numpy()).split(groups, axis=0)

    for w, b, l in zip(sub_weights, sub_bias, mod.convs):
        if b is not None:
            l.load_weights([("weight", w), ("bias", b)])
        else:
            l.load_weights([("weight", w)])


def convert_module_conv(tmodule, mod: modules.Conv):
    # batchnorm
    bn = tmodule.bn

    mod.batchnorm.load_weights([
        ("weight", mx.array(bn.weight.detach().numpy())),
        ("bias", mx.array(bn.bias.detach().numpy())),
        ("running_mean", mx.array(bn.running_mean.numpy())),
        ("running_var", mx.array(bn.running_var.numpy())),
    ])

    # conv2d
    if isinstance(mod.conv, nn.Conv2d):
        convert_module_conv2d(tmodule.conv, mod.conv)
    elif isinstance(mod.conv, modules.ExtendedConv2d):
        convert_module_extendedconv2d(tmodule.conv, mod.conv)


def convert_module_adown(tmodule, mod: modules.ADown):
    convert_module_conv(tmodule.cv1, mod.conv1)
    convert_module_conv(tmodule.cv2, mod.conv2)


def convert_module_repconvn(tmodule, mod: modules.RepConvN):
    convert_module_conv(tmodule.conv1, mod.conv1)
    convert_module_conv(tmodule.conv2, mod.conv2)


def convert_module_repnbottleneck(tmodule, mod: modules.RepNBottleneck):
    convert_module_repconvn(tmodule.cv1, mod.conv1)
    convert_module_conv(tmodule.cv2, mod.conv2)


def convert_module_repncsp(tmodule, mod: modules.RepNCSP):
    convert_module_conv(tmodule.cv1, mod.conv1)
    convert_module_conv(tmodule.cv2, mod.conv2)
    convert_module_conv(tmodule.cv3, mod.conv3)

    for t, m in zip(tmodule.m, mod.bottlenecks.layers):
        convert_module_repnbottleneck(t, m)


def convert_module_cblinear(tmodule, mod: modules.CBLinear):
    convert_module_conv2d(tmodule.conv, mod.conv)


def convert_module_sequential(tmodule: torch.nn.Sequential, mod: nn.Sequential):
    nlayers_torch = len(tmodule)
    nlayers_mx = len(mod.layers)

    if nlayers_torch != nlayers_mx:
        raise RuntimeError(f"Different layers in sequential {nlayers_torch} {nlayers_mx}")

    for tlayer, layer in zip(tmodule, mod.layers):
        if isinstance(layer, modules.Conv):
            convert_module_conv(tlayer, layer)
        elif isinstance(layer, modules.RepNCSP):
            convert_module_repncsp(tlayer, layer)
        elif isinstance(layer, nn.Conv2d):
            convert_module_conv2d(tlayer, layer)
        elif isinstance(layer, modules.ExtendedConv2d):
            convert_module_extendedconv2d(tlayer, layer)
        else:
            raise RuntimeError("Not implemented")


def convert_module_repncspelan4(tmodule, mod: modules.RepNCSPELAN4):
    convert_module_conv(tmodule.cv1, mod.conv1)
    convert_module_sequential(tmodule.cv2, mod.conv2)
    convert_module_sequential(tmodule.cv3, mod.conv3)
    convert_module_conv(tmodule.cv4, mod.conv4)


def convert_module_sppelan(tmodule, mod: modules.SPPELAN):
    convert_module_conv(tmodule.cv1, mod.conv1)
    convert_module_conv(tmodule.cv5, mod.conv2)


def convert_module_ddetect(detect, mod: modules.DDetect):
    for t, m in zip(detect.cv2, mod.convs2):
        convert_module_sequential(t, m)

    for t, m in zip(detect.cv3, mod.convs3):
        convert_module_sequential(t, m)


def convert_module_dualddetect(detect, mod: modules.DualDDetect):
    for t, m in zip(detect.cv2, mod.convs2):
        convert_module_sequential(t, m)

    for t, m in zip(detect.cv3, mod.convs3):
        convert_module_sequential(t, m)

    for t, m in zip(detect.cv4, mod.convs4):
        convert_module_sequential(t, m)

    for t, m in zip(detect.cv5, mod.convs5):
        convert_module_sequential(t, m)


def convert_weight_torch_to_mx(torch_model, mx_model: yolo.YoloBase, torch_silence: int = 1):
    """Converts Torch weights to MLX weights.

    Args:
        torch_model: Original Yolov9 model in pytorch.
        mx_model: Yolov9-MLX model.
        torch_silence: Usually, Yolov9 will have a silence layer at the start of models,
            our model will skip this (torch_silence=1).
            Except for Yolov9-C-Converted, which doesn't have Silence at start.
    """
    from models import yolo as org_yolo
    from models import common

    n_layers_torch = len(torch_model.model)
    n_layers = len(mx_model.layers)

    if n_layers_torch != n_layers + torch_silence:
        raise RuntimeError("Missmatch number of layers {} {}".format(n_layers_torch, n_layers))

    for tlayer, layer in zip(torch_model.model[torch_silence:], mx_model.layers):
        # Make weights conversion
        if isinstance(tlayer, common.Conv) and isinstance(layer, modules.Conv):
            convert_module_conv(tlayer, layer)
        elif isinstance(tlayer, common.RepNCSPELAN4) and isinstance(layer, modules.RepNCSPELAN4):
            convert_module_repncspelan4(tlayer, layer)
        elif isinstance(tlayer, common.ADown) and isinstance(layer, modules.ADown):
            convert_module_adown(tlayer, layer)
        elif isinstance(tlayer, common.SPPELAN) and isinstance(layer, modules.SPPELAN):
            convert_module_sppelan(tlayer, layer)
        elif isinstance(tlayer, common.CBLinear) and isinstance(layer, modules.CBLinear):
            convert_module_cblinear(tlayer, layer)
        elif isinstance(tlayer, common.CBFuse) and isinstance(layer, modules.CBFuse):
            continue
        elif isinstance(tlayer, org_yolo.DDetect) and isinstance(layer, modules.DDetect):
            convert_module_ddetect(tlayer, layer)
        elif isinstance(tlayer, org_yolo.DualDDetect) and isinstance(layer, modules.DualDDetect):
            convert_module_dualddetect(tlayer, layer)
        elif isinstance(tlayer, torch.nn.Upsample) and isinstance(layer, nn.Upsample):
            continue
        elif isinstance(tlayer, common.Concat) and isinstance(layer, modules.Concat):
            continue
        else:
            raise RuntimeError(f"Type {type(tlayer)} and {type(layer)} are not compatible.")


def tensor_torch_to_mx(t: torch.tensor, cpu_device: torch.device | None = None) -> mx.array:
    cpu = cpu_device if cpu_device else torch.device("cpu")
    a = t.moveaxis(1, -1).to(cpu)
    return mx.array(a.numpy())


def verify_loaded_model(torch_model, mx_model: yolo.YoloBase, is_dual: bool):
    device = torch.device("mps")
    cpu = torch.device("cpu")

    torch_model.eval().to(device)
    mx_model.eval()

    im = torch.rand(1, 3, 32, 32)
    im_mx = mx.array(im.moveaxis(1, -1).numpy())

    yt1, yt2, dt1, dt2 = None, None, None, None
    y1, y2, d1, d2 = None, None, None, None

    if is_dual:
        ys, ds = torch_model(im.to(device, dtype=torch.half))
        yt1, yt2 = ys
        dt1, dt2 = ds
        y1, y2, d1, d2 = mx_model(im_mx)
    else:
        yt1, dt1 = torch_model(im.to(device, dtype=torch.half))
        y1, d1 = mx_model(im_mx)

    # check result
    assert len(dt1) == len(d1)

    if is_dual:
        assert dt2 is not None
        assert d2 is not None
        assert len(dt2) == len(d2)

    for dt, d in zip(dt1 + dt2 if dt2 is not None else [], d1 + d2 if d2 is not None else []):
        t = tensor_torch_to_mx(dt)
        assert t.shape == d.shape
        assert mx.allclose(t, d, atol=0.1).item()

    yt1 = tensor_torch_to_mx(yt1.detach(), cpu)
    assert y1.shape == yt1.shape
    assert mx.allclose(y1, yt1, atol=0.4).item()

    if is_dual:
        yt2 = tensor_torch_to_mx(yt2.detach(), cpu)
        assert y2.shape == yt2.shape
        assert mx.allclose(y2, yt2, atol=0.4).item()

    print("Result verifed!")


def load_yolov9_model(model_name: str, model_pt: pathlib.Path) -> tuple[Any, yolo.YoloBase, bool]:
    """Loads original Yolov9 model in pytorch and
    corresponding Yolov9-MLX model.

    Args:
        model_name: Model name like yolov9-c or yolov9-e.
        model_pt: Original model checkpoint pt file.
    """
    torch_model = torch.load(model_pt)["model"]
    print(f"Loaded {model_name} torch model.")

    if model_name not in (
        "yolov9", "yolov9-c", "yolov9-e",
        "yolov9-c-converted", "yolov9-e-converted"
    ):
        raise ValueError("Invalid model name", model_name)

    torch_model_silence = 1
    is_dual = True

    if model_name == "yolov9-c":
        mx_model = yolo.Yolov9C()
    elif model_name == "yolov9-c-converted":
        mx_model = yolo.Yolov9CConverted()
        torch_model_silence = 0
        is_dual = False
    elif model_name == "yolov9-e":
        mx_model = yolo.Yolov9E()
    elif model_name == "yolov9-e-converted":
        mx_model = yolo.Yolov9EConverted()
        is_dual = False
    else:
        mx_model = yolo.Yolov9()

    convert_weight_torch_to_mx(torch_model, mx_model, torch_model_silence)
    print(f"Converted {model_name} to MLX model.")

    return torch_model, mx_model, is_dual


def save_model(model: yolo.YoloBase, out: str):
    model.save_weights(out)
    print("Model saved to " + out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolov9",
        type=str,
        required=True,
        help='path to yolov9 directory'
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=("yolov9", "yolov9-c", "yolov9-e", "yolov9-c-converted", "yolov9-e-converted"),
        help="Name of yolov9 model, yolov9-c, yolov9-e",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to yolov9 model pytorch pt file"
    )
    parser.add_argument(
        "--out",
        type=str,
        help="output safetensor file"
    )
    args = parser.parse_args()

    # Append Yolov9 Path to sys
    sys.path.append(str(pathlib.Path(args.yolov9).absolute()))

    torch_model, mx_model, is_dual = load_yolov9_model(args.model, pathlib.Path(args.checkpoint))
    verify_loaded_model(torch_model, mx_model, is_dual)
    save_model(mx_model, args.out or args.model + ".safetensors")


if __name__ == "__main__":
    main()
