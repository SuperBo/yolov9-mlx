from mlx import core as mx
import numpy as np


def single_nms(
        detects: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
        max_detects: int = 1000,
        max_boxes: int = 30000
    ) -> tuple[np.ndarray, np.ndarray]:
    """Non-max-supression from Fast RCNN
    https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
    https://github.com/pytorch/vision/issues/942
    This works on a single image prediction.

    Args:
        detects: A numpy array of shape (num_detects, 4).
        scores: A numpy array of shape (num_detects,).
        iou_threshold: IOU threshold in supression.
        conf_threshold: Confidence threshold to ignore detections.
        max_detects: Maximum of detection need to be done.
        max_boxes: Maxium boxes to process in input.

    Return:
        A tuple of remaining boxes and indices to keep
    """
    order_org = np.argsort(scores)[::-1] # get boxes with more ious first
    order_org = order_org[:max_boxes] # limit nms boxes

    detects_sorted = detects[order_org]
    scores_sorted = scores[order_org]

    x1 = detects_sorted[:, 0]
    y1 = detects_sorted[:, 1]
    x2 = detects_sorted[:, 2]
    y2 = detects_sorted[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # materialize to loop over
    mx.eval(detects_sorted, scores_sorted, x1, y1, x2, y2, areas)

    # loop through dectects in sorted order
    order = np.arange(0, len(detects_sorted), dtype=np.uint32)
    keep = []

    while order.size > 0:
        i = order[0]

        keep.append(i)
        if len(keep) == max_detects:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        intersect = w * h
        overlap = intersect / (areas[i] + areas[order[1:]] - intersect)

        keep_indices = np.where(overlap <= iou_threshold)[0] + 1
        order = order[keep_indices]


    detects_remain = detects_sorted[keep]
    keep_org = order_org[keep]

    return detects_remain, keep_org


def xywh_to_xyxy(x: np.ndarray) -> np.ndarray:
    """Converts nx4 boxes from (x, y, w, h) to (x1, y1, x2, y2)
    where xy=center, x1y1=top-left, x2y2=bottom-right
    """
    xy = x[..., :2] # x, y
    b = x[..., 2:] / 2 # divide width, height by 2
    y = np.concatenate((xy - b, xy + b), axis=-1)
    return y


def batch_non_max_suppression(
    prediction: mx.array,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    classes=None,
    max_detects: int = 300,
    num_masks: int = 0,  # number of masks
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    This works on a batch of image predictions.

    Args:
        prediction: mx.array in shape (b, p, c).
        conf_threshold: Confidence threshold to filter prediction.
        iou_threshold: IOU threshold to suppress boxes.
        multi_label: Predict multi label for each boxes.

    Returns:
         A list of detections on (n,6) tensor per image [xyxy, conf, cls]
    """
    # batch_size = prediction.shape[0]  # batch size
    num_classes = prediction.shape[1] - num_masks - 4 # number of classes
    mask_index = 4 + num_classes # mask start index

    # Settings
    # max_wh = 7680 # (pixels) maximum box width and height
    max_nms = 30000 # maximum number of boxes into nms()

    # output = [mx.zeros((0, 6 + num_masks))] * batch_size

    # convert to numpy for better speed
    prediction = np.array(prediction)

    boxes, classes, _ = np.split(prediction, (4, mask_index), axis=-1)

    anchor_conf = classes.max(axis=-1)
    image_conf = anchor_conf.max(axis=-1)

    out_boxes = []
    out_classes = []

    # Iterate through image in batch
    # for box, cls_, iscore, ascore in zip(boxes, classes, image_conf, anchor_conf):
    for box, cls_, iscore, ascore in zip(boxes, classes, image_conf, anchor_conf):
        # Skip image if max confidence is < conf_threshold
        if iscore < conf_threshold:
            out_boxes.append(None)
            out_classes.append(None)
            continue

        # Filter low confidence box
        cond = ascore > conf_threshold

        box_xyxy = xywh_to_xyxy(box[cond])

        # Single NMS
        box_keep, keep = single_nms(box_xyxy, ascore[cond], iou_threshold, max_detects, max_nms)

        print(box_keep)
        out_boxes.append(box_keep)
        out_classes.append(cls_[cond][keep])

    return out_boxes, out_classes


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip boxes (xyxy) to image shape (height, width)
    In addition converting float box to integer box.

    Args:
        boxes: array with shape (x, 4)
        width: max width.
        height: max height.

    Returns:
        An array with width and height clipped.
    """
    amax = np.array([width, height, width, height])
    return np.clip(boxes, 0, amax).astype(np.uint32)


def scale_boxes(
    boxes: np.ndarray,
    w0: int,
    h0: int,
    w1: int,
    h1: int,
) -> np.ndarray:
    """Rescale boxes (xyxy) from (w0, h0) to (w1, h1).

    Args:
        boxes: Array with shape (x, 4)
        w0: Old image width.
        h0: Old image height.
        w1: New image 1 width.
        h1: New image 1 width.
    """
    gain = min(w0 / w1, h0 / h1)  # gain  = old / new
    pad_x = (w0 - w1 * gain) / 2 # wh padding
    pad_y = (h0 - h1 * gain) / 2 # wh padding

    new_boxes = (boxes - np.array([pad_x, pad_y, pad_x, pad_y])) / gain
    return clip_boxes(new_boxes, w1, h1)


def make_divisible_by_32(value: int) -> int:
    """Returns nearest value divisible by 32.
    Used for calculate resize size.
    """
    return ((value + 16) // 32) * 32
