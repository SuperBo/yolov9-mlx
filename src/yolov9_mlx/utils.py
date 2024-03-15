import mlx.core as mx
import numpy as np

def _broadcast_shape(*args):
    """Returns the shape of the arrays that would result from broadcasting the
    supplied arrays against each other.
    """
    # use the old-iterator because np.nditer does not handle size 0 arrays
    # consistently
    b = np.broadcast(*args[:32])
    # unfortunately, it cannot handle 32 or more arguments directly
    for pos in range(32, len(args), 31):
        # ironically, np.broadcast does not properly handle np.broadcast
        # objects (it treats them as scalars)
        # use broadcasting to avoid allocating the full array
        b = np.broadcast_to(0, b.shape)
        b = np.broadcast(b, *args[pos:(pos + 31)])
    return b.shape


def broadcast_arrays(*xs, stream: None | mx.Stream | mx.Device = None):
    """Broadcasts any number of arrays against each other.
    """
    shape = _broadcast_shape(*xs)

    if all(array.shape == shape for array in xs):
        # Common case where nothing needs to be broadcasted.
        return xs

    return tuple(mx.broadcast_to(x, shape, stream=stream) for x in xs)


def meshgrid(*xs, indexing='xy'):
    """Returns a tuple of coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.
    """
    ndim = len(xs)

    if indexing not in ("xy", "ij"):
        raise ValueError(
            "Valid values for `indexing` are 'xy' and 'ij'."
        )

    if ndim == 1:
        return xs[0]

    s0 = (1,) * ndim
    output = [
        x.reshape(s0[:i] + (-1,) + s0[i + 1:])
        for i, x in enumerate(xs)
    ]

    if indexing == 'xy' and ndim > 1:
        # switch first and second axis
        output[0] = output[0].reshape((1, -1) + s0[2:])
        output[1] = output[1].reshape((-1, 1) + s0[2:])

    # Return the full N-D matrix (not only the 1-D vector)
    output = broadcast_arrays(*output)

    return output


def make_anchors(features: list[mx.array], strides: list[int], grid_cell_offset: float = 0.5):
    """Generates anchors from list of features.

    Args:
        features: List of mx.array with shape NxHxWxC
        strides: List of strides to make anchors.
        grid_cell_offset: Offset between Grid Cells.

    Returns:
        A Tuple of (anchor_points, stride_tensor)
    """
    anchor_points, stride_list = [], []
    dtype = features[0].dtype
    for feat, stride in zip(features, strides):
        _, h, w, _ = feat.shape
        sx = mx.arange(w, dtype=dtype) + grid_cell_offset  # shift x
        sy = mx.arange(h, dtype=dtype) + grid_cell_offset  # shift y
        yv, xv = meshgrid(sy, sx, indexing="ij")

        anchor_points.append(mx.stack([xv, yv], axis=-1).reshape(-1, 2))
        stride_list.append(mx.full([h * w, 1], stride, dtype=dtype))

    anchors = mx.concatenate(anchor_points)
    stride_tensor = mx.concatenate(stride_list)

    return anchors, stride_tensor


def dist2bbox(
    distance: mx.array,
    anchor_points: mx.array,
    xywh: bool = True,
    axis: int = -1
) -> mx.array:
    """Transform distance(ltrb) to box(xywh or xyxy).

    Args:
        distance: Distance array.
        anchor_points: anchor points array.
        xywh: Use xywh layout.
        axis: Axis to transform.
    """
    lt, rb = mx.split(distance, 2, axis=axis)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb

    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return mx.concatenate([c_xy, wh], axis=axis)  # xywh bbox

    return mx.concatenate([x1y1, x2y2], axis=axis)  # xyxy bbox
