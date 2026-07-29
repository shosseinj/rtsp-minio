from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    x: float
    y: float
    width: float
    height: float


def clamp_bbox(box: BBox, width: float, height: float) -> BBox:
    x = min(max(box.x, 0.0), width)
    y = min(max(box.y, 0.0), height)
    right = min(max(box.x + box.width, x), width)
    bottom = min(max(box.y + box.height, y), height)
    return BBox(x, y, right - x, bottom - y)


def letterbox_to_source(
    box: BBox,
    source_width: float,
    source_height: float,
    target_width: float,
    target_height: float,
) -> BBox:
    scale = min(target_width / source_width, target_height / source_height)
    pad_x = (target_width - source_width * scale) / 2.0
    pad_y = (target_height - source_height * scale) / 2.0
    return clamp_bbox(
        BBox(
            (box.x - pad_x) / scale,
            (box.y - pad_y) / scale,
            box.width / scale,
            box.height / scale,
        ),
        source_width,
        source_height,
    )


def source_to_target(
    box: BBox,
    source_width: float,
    source_height: float,
    target_width: float,
    target_height: float,
    preserve_aspect_ratio: bool = True,
) -> BBox:
    if not preserve_aspect_ratio:
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        return clamp_bbox(
            BBox(
                box.x * scale_x,
                box.y * scale_y,
                box.width * scale_x,
                box.height * scale_y,
            ),
            target_width,
            target_height,
        )
    scale = min(target_width / source_width, target_height / source_height)
    pad_x = (target_width - source_width * scale) / 2.0
    pad_y = (target_height - source_height * scale) / 2.0
    return clamp_bbox(
        BBox(
            box.x * scale + pad_x,
            box.y * scale + pad_y,
            box.width * scale,
            box.height * scale,
        ),
        target_width,
        target_height,
    )


def mux_to_source(
    box: BBox,
    source_width: float,
    source_height: float,
    mux_width: float,
    mux_height: float,
    padded: bool,
) -> BBox:
    if padded:
        return letterbox_to_source(
            box, source_width, source_height, mux_width, mux_height
        )
    return source_to_target(
        box,
        mux_width,
        mux_height,
        source_width,
        source_height,
        preserve_aspect_ratio=False,
    )
