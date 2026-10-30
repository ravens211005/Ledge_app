"""Drawing helpers. All arrays are RGB uint8."""
import cv2
import numpy as np
import config

STAIR_COLOR = (0, 160, 255)
EDGE_COLOR = (255, 40, 40)


def _color(d):
    return EDGE_COLOR if d.class_name == config.EDGE_CLASS_NAME else STAIR_COLOR


def _contours(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cs


def _anchor(d):
    """Label position: mask centroid, staggered by detection index so labels don't overprint."""
    m = cv2.moments(d.mask.astype(np.uint8), binaryImage=True)
    if m["m00"] > 0:
        cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    else:
        cx, cy = int((d.box[0] + d.box[2]) / 2), int((d.box[1] + d.box[3]) / 2)
    return max(4, cx - 40), cy + 16 * (d.idx % 4)


def _label(img, text, x, y, color):
    cv2.putText(img, text, (x, max(14, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, max(14, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def segmentation_overlay(rgb, dets):
    layer = rgb.copy()
    for d in dets:
        layer[d.mask] = _color(d)
    out = cv2.addWeighted(layer, 0.4, rgb, 0.6, 0)
    for d in dets:
        cv2.drawContours(out, _contours(d.mask), -1, _color(d), 2)
        x, y = _anchor(d)
        _label(out, f"#{d.idx} {d.class_name} {d.conf:.2f}", x, y, (255, 255, 255))
    return out


def edge_view(rgb, dets):
    out = (rgb * 0.45).astype(np.uint8)
    for d in dets:
        if d.class_name == config.EDGE_CLASS_NAME:
            out[d.mask] = rgb[d.mask]
            cv2.drawContours(out, _contours(d.mask), -1, EDGE_COLOR, 3)
            x, y = _anchor(d)
            _label(out, f"#{d.idx} {d.conf:.2f}", x, y, (255, 255, 255))
        else:
            cv2.drawContours(out, _contours(d.mask), -1, STAIR_COLOR, 1)
    return out


def depth_colormap(depth_rel):
    u8 = (depth_rel * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def combined(rgb, dets, depth_rel):
    base = cv2.addWeighted(depth_colormap(depth_rel), 0.35, rgb, 0.65, 0)
    for d in dets:
        cv2.drawContours(base, _contours(d.mask), -1, _color(d), 2)
        x, y = _anchor(d)
        _label(base, f"#{d.idx} {d.class_name}", x, y, (255, 255, 255))
    return base