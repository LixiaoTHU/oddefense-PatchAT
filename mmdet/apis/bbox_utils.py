import torch
from torch import Tensor

from .typing import _pair, _size_2_t

# This is the smallest subnormal positive of fp16,
# so this epsilon works with fp16 and above when dealing with div-by-zero.
_EPS = 6e-8


def bbox_scale(boxes: Tensor, img_size: _size_2_t):
    img_size = _pair(img_size)
    w, h = img_size
    scale = boxes.new_tensor((w, h, w, h))
    return boxes * scale


def bbox_to_x1y1x2y2(boxes: Tensor, format="center-size"):
    """Alternative is `mmdet.core.bbox.bbox_cxcywh_to_xyxy`."""
    assert boxes.shape[-1] >= 4

    if format in ("center-size", "YOLO", "cxcywh"):
        center, size = boxes[..., :2], boxes[..., 2:4]
        half_size = size / 2
        out_boxes = torch.cat((center - half_size, center + half_size), dim=-1)
    elif format in ("corner-size", "COCO", "mxmywh", "xywh"):
        corner, size = boxes[..., :2], boxes[..., 2:4]
        out_boxes = torch.cat((corner, corner + size), dim=-1)
    elif format in ("corner-corner", "PASCAL", "xyxy"):
        # For compatibility
        return boxes
    else:
        raise ValueError(format)

    if boxes.shape[-1] > 4:
        return torch.cat((out_boxes, boxes[..., 4:]), dim=-1)
    else:
        return out_boxes


def bbox_from_x1y1x2y2(boxes: Tensor, format="center-size"):
    """Alternative is `mmdet.core.bbox.bbox_xyxy_to_cxcywh`."""
    assert boxes.shape[-1] >= 4

    top_left, bottom_right = boxes[..., :2], boxes[..., 2:4]
    if format in ("center-size", "YOLO", "cxcywh"):
        size = bottom_right - top_left
        center = (top_left + bottom_right) / 2
        out_boxes = torch.cat((center, size), dim=-1)
    elif format in ("corner-size", "COCO", "mxmywh", "xywh"):
        size = bottom_right - top_left
        out_boxes = torch.cat((top_left, size), dim=-1)
    elif format in ("corner-corner", "PASCAL", "xyxy"):
        # For compatibility
        return boxes
    else:
        raise ValueError(format)

    if boxes.shape[-1] > 4:
        return torch.cat((out_boxes, boxes[..., 4:]), dim=-1)
    else:
        return out_boxes


def bbox_iou(
    boxes1: tuple[float, float, float, float], boxes2: tuple[float, float, float, float]
):
    """Compute iou of single box.

    Input boxes should be in x1y1x2y2 format."""
    x11, y11, x12, y12 = boxes1
    x21, y21, x22, y22 = boxes2
    xi1 = max(x11, x21)
    yi1 = max(y11, y21)
    xi2 = min(x12, x22)
    yi2 = min(y12, y22)
    overlap = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    a1 = (x12 - x11) * (y12 - y11)
    a2 = (x22 - x21) * (y22 - y21)
    union = a1 + a2 - overlap
    union = max(union, _EPS)
    return overlap / union


def bbox_ious(boxes1: Tensor, boxes2: Tensor):
    """Alternative is `mmdet.core.bbox.BboxOverlaps2D`.

    Input boxes should be in x1y1x2y2 format."""

    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    assert boxes1.shape[-1] >= 4

    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)
    assert boxes2.shape[-1] >= 4

    boxes1 = boxes1.unsqueeze(-2)  # B N 1 4
    boxes2 = boxes2.unsqueeze(-3)  # B 1 M 4

    top_left_i = torch.max(boxes1[..., :2], boxes2[..., :2])
    bottom_right_i = torch.min(boxes1[..., 2:4], boxes2[..., 2:4])
    size_i = torch.clamp_min(bottom_right_i - top_left_i, 0)
    overlap = size_i[..., 0] * size_i[..., 1]
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - overlap
    # union = union + _EPS
    union = union.clamp_min(_EPS)
    return overlap / union
